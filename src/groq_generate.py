"""Zero-shot Llama-3.3-70B-Instruct through the Groq API (reference for a large untuned model).

The API key is read from the GROQ_API_KEY environment variable (never hard-code it).
Because of rate limits, --subset 500 uses a fixed stratified 500-pair subset (100 per dialect).
Finished rows are cached, so an interrupted run resumes where it stopped.

  export GROQ_API_KEY=...
  python src/groq_generate.py --splits test --subset 500
"""
import argparse
import json
import os
import time
from pathlib import Path

from common import build_messages, clean_output, load_jsonl, stratified_subset, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--extra_test", nargs="*", default=[])
    ap.add_argument("--subset", type=int, default=None, help="stratified subset size, e.g. 500")
    ap.add_argument("--instruction", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--system", default="llama70b_groq_zeroshot")
    ap.add_argument("--run", default="p0")
    ap.add_argument("--sleep", type=float, default=2.1, help="seconds between calls (free-tier rate limit)")
    args = ap.parse_args()

    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    data = Path(args.data_dir)
    targets = [(sp, data / f"{sp}.jsonl") for sp in args.splits] + [(Path(p).stem, Path(p)) for p in args.extra_test]
    for name, path in targets:
        rows = load_jsonl(path)
        if args.subset:
            rows = stratified_subset(rows, args.subset, seed=0)
        cache_path = Path(args.out_dir) / args.system / args.run / f"{name}.cache.jsonl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = {}
        if cache_path.exists():
            cache = {r["id"]: r["raw"] for r in load_jsonl(cache_path)}
        with open(cache_path, "a", encoding="utf-8") as fh:
            for i, r in enumerate(rows):
                if r["id"] in cache:
                    continue
                for attempt in range(8):
                    try:
                        resp = client.chat.completions.create(
                            model=args.model, messages=build_messages(r, args.instruction),
                            temperature=0, max_tokens=128)
                        cache[r["id"]] = resp.choices[0].message.content or ""
                        break
                    except Exception as e:  # rate limit or transient error: back off
                        wait = min(120, 5 * 2 ** attempt)
                        print(f"  {type(e).__name__}: retry in {wait}s")
                        time.sleep(wait)
                else:
                    cache[r["id"]] = ""  # unusable output, scored 0
                fh.write(json.dumps({"id": r["id"], "raw": cache[r["id"]]}, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 50 == 0:
                    print(f"  {name}: {i}/{len(rows)}", flush=True)
                time.sleep(args.sleep)
        preds = [{**r, "raw": cache[r["id"]], "hyp": clean_output(cache[r["id"]])} for r in rows]
        write_jsonl(Path(args.out_dir) / args.system / args.run / f"{name}.jsonl", preds)
        print(f"wrote {name}: {len(preds)} rows")


if __name__ == "__main__":
    main()
