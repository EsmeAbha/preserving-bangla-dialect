"""Scores every system run and builds the result tables.

Reads outputs/<system>/<run>/<split>.jsonl (fields id, dialect, tgt, hyp) and writes to results/<split>/:
  runs.csv          one line per system, run and dialect (+ "all"): chrF, BLEU, LaBSE, WER, n, empty
  main_table.md     mean +- std over runs: chrF per dialect, overall chrF / BLEU / LaBSE
  significance.md   95% paired bootstrap CIs (1,000 resamples) of chrF differences between systems
  rq2.md            dialect distance (100 - copy chrF), copy WER, ranking, Spearman with bootstrap CI
  sentences.csv     per-sentence chrF (and LaBSE) for every run, used for human-eval / error analysis

Empty or unusable outputs stay in and get chrF 0 and LaBSE 0.

  python src/evaluate.py --split test --labse --compare qwen7b_qlora:qwen7b_zeroshot qwen7b_qlora:qwen7b_fewshot
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from scipy.stats import spearmanr

from common import DIALECTS, load_jsonl

_CHRF = CHRF()


# ---------- metrics ----------
def chrf_stats(hyps, refs):
    """Per-sentence chrF sufficient statistics (n x 18); corpus chrF = score of the column sums."""
    return np.array(_CHRF._extract_corpus_statistics(hyps, [refs]), dtype=float)


def chrf_from_stats(stats_sum):
    return _CHRF._compute_score_from_stats(list(stats_sum)).score


def corpus_chrf(hyps, refs):
    return _CHRF.corpus_score(hyps, [refs]).score


def sentence_chrf(hyps, refs):
    return [_CHRF.sentence_score(h, [r]).score for h, r in zip(hyps, refs)]


def corpus_bleu(hyps, refs):
    return BLEU().corpus_score(hyps, [refs]).score


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def corpus_wer(hyps, refs):
    errs = sum(edit_distance(h.split(), r.split()) for h, r in zip(hyps, refs))
    return 100 * errs / max(1, sum(len(r.split()) for r in refs))


class Labse:
    """Cosine similarity of LaBSE embeddings, cached per text; empty outputs get 0."""

    def __init__(self, name="sentence-transformers/LaBSE"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)
        self.cache = {}

    def embed(self, texts):
        new = [t for t in dict.fromkeys(texts) if t not in self.cache and t]
        if new:
            for t, e in zip(new, self.model.encode(new, batch_size=128, normalize_embeddings=True,
                                                   show_progress_bar=False)):
                self.cache[t] = e

    def scores(self, hyps, refs):
        self.embed(list(hyps) + list(refs))
        return [float(self.cache[h] @ self.cache[r]) if h and r else 0.0 for h, r in zip(hyps, refs)]


# ---------- loading ----------
def load_runs(out_dir, split):
    """{system: {run: rows}}"""
    systems = defaultdict(dict)
    for f in sorted(Path(out_dir).glob(f"*/*/{split}.jsonl")):
        systems[f.parent.parent.name][f.parent.name] = load_jsonl(f)
    return dict(systems)


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def fmt(m, s, show_std=True):
    if m is None:
        return "–"
    return f"{m:.2f} ± {s:.2f}" if show_std and s else f"{m:.2f}"


# ---------- bootstrap ----------
def paired_bootstrap(runs_a, runs_b, n_boot=1000, seed=12345):
    """CI of mean-over-runs corpus chrF(A) - chrF(B), resampling sentences shared by all runs."""
    ids = set.intersection(*[{r["id"] for r in run} for run in runs_a + runs_b])
    ids = sorted(ids)
    def stats(run):
        by = {r["id"]: r for r in run}
        return chrf_stats([by[i]["hyp"] for i in ids], [by[i]["tgt"] for i in ids])
    sa, sb = [stats(r) for r in runs_a], [stats(r) for r in runs_b]
    point = np.mean([chrf_from_stats(s.sum(0)) for s in sa]) - np.mean([chrf_from_stats(s.sum(0)) for s in sb])
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=n_boot).astype(float)
    diffs = []
    for c in counts:
        a = np.mean([chrf_from_stats(c @ s) for s in sa])
        b = np.mean([chrf_from_stats(c @ s) for s in sb])
        diffs.append(a - b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = float(np.mean(np.array(diffs) <= 0)) if point > 0 else float(np.mean(np.array(diffs) >= 0))
    return point, lo, hi, len(ids), p


def spearman_ci(x, y, n_boot=1000, seed=12345):
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", nargs="*", default=None, help="restrict / order the systems in the tables")
    ap.add_argument("--labse", action="store_true", help="compute LaBSE similarity (downloads the model)")
    ap.add_argument("--compare", nargs="*", default=[], help="pairs A:B for paired bootstrap")
    ap.add_argument("--main_system", default="qwen7b_qlora", help="system used for the RQ2 correlation")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    runs = load_runs(args.out_dir, args.split)
    if not runs:
        raise SystemExit(f"no outputs found in {args.out_dir}/*/*/{args.split}.jsonl")
    order = args.systems or sorted(runs)
    res = Path(args.results_dir) / args.split
    res.mkdir(parents=True, exist_ok=True)
    labse = Labse() if args.labse else None
    present = [d for d in DIALECTS if any(r["dialect"] == d for rr in runs.values() for run in rr.values() for r in run)]

    lines, sent_lines = [], []
    table = defaultdict(lambda: defaultdict(list))  # (system) -> (metric, dialect) -> [per run]
    for system in order:
        for run, rows in runs.get(system, {}).items():
            hyps, refs = [r["hyp"] or "" for r in rows], [r["tgt"] for r in rows]
            sent = sentence_chrf(hyps, refs)
            lab = labse.scores(hyps, refs) if labse else [None] * len(rows)
            for r, s, l in zip(rows, sent, lab):
                sent_lines.append([system, run, r["id"], r["dialect"], round(s, 2), None if l is None else round(l, 4)])
            for d in present + ["all"]:
                idx = [i for i, r in enumerate(rows) if d == "all" or r["dialect"] == d]
                if not idx:
                    continue
                h, t = [hyps[i] for i in idx], [refs[i] for i in idx]
                m = {"chrF": corpus_chrf(h, t), "BLEU": corpus_bleu(h, t), "WER": corpus_wer(h, t),
                     "LaBSE": float(np.mean([lab[i] for i in idx])) if labse else None,
                     "n": len(idx), "empty": sum(1 for x in h if not x.strip())}
                lines.append([system, run, d] + [None if m[k] is None else round(m[k], 4) if isinstance(m[k], float) else m[k]
                                                 for k in ("chrF", "BLEU", "LaBSE", "WER", "n", "empty")])
                for k in ("chrF", "BLEU", "LaBSE", "WER"):
                    table[system][(k, d)].append(m[k])
                table[system][("n", d)] = [len(idx)]

    with open(res / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "run", "dialect", "chrF", "BLEU", "LaBSE", "WER", "n", "empty"])
        w.writerows(lines)
    with open(res / "sentences.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "run", "id", "dialect", "chrF", "LaBSE"])
        w.writerows(sent_lines)

    # main table
    head = ["System", "Runs", "n"] + [f"chrF {d[:4].title()}" for d in present] + ["chrF all", "BLEU all", "LaBSE all"]
    md = [f"# Results on {args.split} (mean ± std over runs)\n", "| " + " | ".join(head) + " |",
          "|" + "---|" * len(head)]
    for system in order:
        if system not in table:
            continue
        t = table[system]
        cells = [system, str(len(runs[system])), str(t[("n", "all")][0])]
        cells += [fmt(*mean_std(t[("chrF", d)])) for d in present]
        cells += [fmt(*mean_std(t[("chrF", "all")])), fmt(*mean_std(t[("BLEU", "all")])),
                  fmt(*mean_std(t[("LaBSE", "all")]))]
        md.append("| " + " | ".join(cells) + " |")
    (res / "main_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # significance
    if args.compare:
        sig = [f"# Paired bootstrap on {args.split} ({args.n_boot} resamples, chrF, mean over runs)\n",
               "| A | B | A − B | 95% CI | n | p (one-sided) |", "|---|---|---|---|---|---|"]
        for pair in args.compare:
            a, b = pair.split(":")
            if a not in runs or b not in runs:
                print(f"skip {pair}: missing outputs")
                continue
            d, lo, hi, n, p = paired_bootstrap(list(runs[a].values()), list(runs[b].values()), args.n_boot)
            sig.append(f"| {a} | {b} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {n} | {p:.3f} |")
        (res / "significance.md").write_text("\n".join(sig) + "\n", encoding="utf-8")
        print("\n".join(sig))

    # RQ2: dialect distance
    if "copy" in runs:
        copy_rows = next(iter(runs["copy"].values()))
        rq2 = [f"# RQ2: dialect distance on {args.split}\n",
               "Distance = 100 − chrF of the copy baseline; copy WER as a check.\n",
               f"| Dialect | Distance | Copy WER | {args.main_system} chrF |", "|---|---|---|---|"]
        for d in present:
            dist = 100 - table["copy"][("chrF", d)][0]
            mm = mean_std(table[args.main_system][("chrF", d)])[0] if args.main_system in table else None
            rq2.append(f"| {d} | {dist:.2f} | {table['copy'][('WER', d)][0]:.2f} | {fmt(mm, 0)} |")
        if args.main_system in runs:
            dist = {r["id"]: 100 - s for r, s in zip(copy_rows, sentence_chrf([r["src"] for r in copy_rows],
                                                                              [r["tgt"] for r in copy_rows]))}
            per_run = [{r["id"]: s for r, s in zip(rows, sentence_chrf([r["hyp"] or "" for r in rows],
                                                                       [r["tgt"] for r in rows]))}
                       for rows in runs[args.main_system].values()]
            ids = sorted(set(dist).intersection(*per_run))
            x = [dist[i] for i in ids]
            y = [np.mean([pr[i] for pr in per_run]) for i in ids]
            rho, lo, hi = spearman_ci(x, y, args.n_boot)
            rq2.append(f"\nPer-sentence Spearman ρ (distance vs. {args.main_system} chrF, mean over runs, "
                       f"pooled, n={len(ids)}): {rho:.3f}, 95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
            dist_rank = sorted(present, key=lambda d: -(100 - table["copy"][("chrF", d)][0]))
            out_rank = sorted(present, key=lambda d: mean_std(table[args.main_system][("chrF", d)])[0])
            rq2.append(f"\nFarthest → closest dialect: {', '.join(dist_rank)}")
            rq2.append(f"\nHardest → easiest for {args.main_system}: {', '.join(out_rank)}")
        (res / "rq2.md").write_text("\n".join(rq2) + "\n", encoding="utf-8")
        print("\n".join(rq2))
    # RQ3: leave-one-dialect-out, systems named <base>_lodo_<dialect>
    lodo = {d: s for s in table for d in present if s.endswith(f"_lodo_{d}")}
    if lodo:
        full = args.main_system if args.main_system in table else None
        rq3 = [f"# RQ3: leave-one-dialect-out on {args.split} (chrF on the held-out dialect)\n",
               f"| Held-out dialect | Fold model | Not seen in training | {full or 'full model'} (seen) | Drop |",
               "|---|---|---|---|---|"]
        for d, s in lodo.items():
            unseen = mean_std(table[s][("chrF", d)])[0]
            seen = mean_std(table[full][("chrF", d)])[0] if full else None
            drop = f"{seen - unseen:.2f}" if seen is not None and unseen is not None else "–"
            rq3.append(f"| {d} | {s} | {fmt(unseen, 0)} | {fmt(seen, 0)} | {drop} |")
        (res / "rq3_lodo.md").write_text("\n".join(rq3) + "\n", encoding="utf-8")
        print("\n".join(rq3))
    print(f"\nwrote {res}/")


if __name__ == "__main__":
    main()
