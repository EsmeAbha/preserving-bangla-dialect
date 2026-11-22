"""Turns the markdown tables in results/ into the LaTeX tables included by report/report.tex.

  python src/make_paper_tables.py --ft qwen05b_lora     (writes report/tables/*.tex)
"""
import argparse
import re
from pathlib import Path

NAMES = {
    "copy": "Copy input", "dictionary": "Dictionary (per dialect)", "dictionary_pooled": "Dictionary (pooled)",
    "banglat5_cpu1ep": "BanglaT5, 1 epoch", "qwen3b_zeroshot": "Qwen2.5-3B zero-shot",
    "qwen3b_fewshot": "Qwen2.5-3B 5-shot",
}
DIAL = ["chittagong", "noakhali", "sylhet", "barishal", "mymensingh"]


def md_table(path, first_col=None):
    """Rows of the first markdown table in a file (or the table whose header starts with first_col)."""
    rows, header, active = [], None, False
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if header is None:
                if first_col and cells[0] != first_col:
                    continue
                header, active = cells, True
            elif set(line) <= set("|-: "):
                continue
            elif active:
                rows.append(dict(zip(header, cells)))
        elif header is not None and active and rows:
            break
    return rows


def num(s):
    s = s.replace("−", "-")
    m = re.match(r"\s*([-+]?\d+\.?\d*)(?:\s*±\s*(\d+\.?\d*))?", s)
    return (float(m.group(1)), float(m.group(2)) if m.group(2) else None) if m else (None, None)


def cell(s, bold=False, digits=1):
    v, sd = num(s)
    if v is None:
        return "--"
    t = f"{v:.{digits}f}"
    if bold:
        t = r"\textbf{" + t + "}"
    if sd is not None:
        t += r"\pmsd{" + f"{sd:.{digits}f}" + "}"
    return t


def results_table(path, systems, ft, cols, caption, label, short=False):
    rows = {r["System"]: r for r in md_table(path)}
    best = {c: max(num(rows[s][c])[0] for s in systems if s in rows and num(rows[s].get(c, "–"))[0] is not None)
            for c in cols}
    names = {**NAMES, ft: "Qwen2.5-0.5B + LoRA"}
    if short:
        names = {**names, ft: "Qwen-0.5B+LoRA", "banglat5_cpu1ep": "BanglaT5", "dictionary": "Dictionary", "copy": "Copy"}
    heads = {"chrF Chit": "Chit.", "chrF Noak": "Noak.", "chrF Sylh": "Sylh.", "chrF Bari": "Bari.",
             "chrF Myme": "Myme.", "chrF all": "chrF", "BLEU all": "BLEU", "LaBSE all": "LaBSE"}
    out = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}\\toprule",
           "System & " + " & ".join(heads[c] for c in cols) + r"\\\midrule"]
    for s in systems:
        if s not in rows:
            continue
        vals = []
        for c in cols:
            v = num(rows[s][c])[0]
            d = 2 if c.startswith("LaBSE") else 1
            vals.append(cell(rows[s][c], bold=v is not None and abs(v - best[c]) < 1e-9, digits=d))
        out.append(names.get(s, s) + " & " + " & ".join(vals) + r"\\")
        if s == "copy":
            out.append(r"\addlinespace[1pt]")
    out += [r"\bottomrule\end{tabular}", r"\caption{" + caption + r"}\label{" + label + "}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def sig_table(paths, ft, caption, label):
    names = {**NAMES, ft: "Qwen-0.5B+LoRA", "dictionary": "Dictionary", "dictionary_pooled": "Dict.\\ pooled",
             "banglat5_cpu1ep": "BanglaT5", "qwen3b_zeroshot": "3B zero-shot", "qwen3b_fewshot": "3B 5-shot",
             "copy": "Copy"}
    out = [r"\begin{table*}[t]\centering\small", r"\begin{tabular}{@{}llrcr@{}}\toprule",
           r"Set & Comparison & $\Delta$chrF & 95\% CI & $n$\\\midrule"]
    for tag, p in paths:
        for r in md_table(p):
            d = num(r["A − B"])[0]
            ci = r["95% CI"].replace("−", "-").replace("+", "").replace("-", "$-$")
            out.append(f"{tag} & {names.get(r['A'], r['A'])} $-$ {names.get(r['B'], r['B'])} & ${d:+.2f}$ & "
                       f"{ci} & {r['n']}" + r"\\")
        out.append(r"\addlinespace[2pt]")
    out[-1] = r"\bottomrule\end{tabular}"
    out += [r"\caption{" + caption + r"}\label{" + label + "}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def ci2(text):
    lo, hi = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", text.replace("−", "-"))]
    return f"$[{lo:.2f}, {hi:.2f}]$"


def rq2_table(path_full, path_sub, ft):
    names = {**NAMES, ft: "Qwen-0.5B + LoRA", "dictionary": "Dictionary", "dictionary_pooled": "Dict.\\ pooled",
             "banglat5_cpu1ep": "BanglaT5", "qwen3b_zeroshot": "3B zero-shot", "qwen3b_fewshot": "3B 5-shot"}
    out = [r"\begin{table}[t]\centering\footnotesize", r"\begin{tabular}{@{}lrcr@{}}\toprule",
           r"System & $\rho$ (sent.) & 95\% CI & $\rho$ (dial.)\\\midrule",
           r"\multicolumn{4}{@{}l}{\emph{Full test set} ($n=1{,}875$)}\\"]
    for r in md_table(path_full):
        out.append(f"{names.get(r['System'], r['System'])} & ${num(r['Sentence-level ρ'])[0]:.2f}$ & "
                   f"{ci2(r['95% CI'])} & ${num(r['Dialect-level ρ (5 dialects)'])[0]:.2f}$" + r"\\")
    out.append(r"\multicolumn{4}{@{}l}{\emph{500-pair subset}}\\")
    for r in md_table(path_sub):
        if r["System"] in ("dictionary", "banglat5_cpu1ep"):
            continue
        out.append(f"{names.get(r['System'], r['System'])} & ${num(r['Sentence-level ρ'])[0]:.2f}$ & "
                   f"{ci2(r['95% CI'])} & ${num(r['Dialect-level ρ (5 dialects)'])[0]:.2f}$" + r"\\")
    out += [r"\bottomrule\end{tabular}",
            r"\caption{RQ2: Spearman correlation between dialect distance ($100-$copy chrF) and chrF, per sentence "
            r"(with bootstrap CI) and over the five dialect means. More negative = difficulty follows distance "
            r"more closely.}\label{tab:rq2}", r"\end{table}"]
    return "\n".join(out) + "\n"


def lodo_table(path):
    rows = {r["System"]: r for r in md_table(path)}
    col = {"chittagong": "chrF Chit", "noakhali": "chrF Noak", "sylhet": "chrF Sylh", "barishal": "chrF Bari",
           "mymensingh": "chrF Myme"}
    out = [r"\begin{table}[t]\centering\small", r"\begin{tabular}{@{}lrrrr@{}}\toprule",
           r"Held-out & Copy & Unseen & Seen & Recovered\\\midrule"]
    rec = []
    for d in DIAL:
        c = num(rows["copy"][col[d]])[0]
        u = num(rows[f"dictionary_lodo_{d}"][col[d]])[0]
        s = num(rows["dictionary_pooled"][col[d]])[0]
        rec.append(100 * (u - c) / (s - c))
        out.append(f"{d.capitalize()} & {c:.1f} & {u:.1f} & {s:.1f} & {rec[-1]:.0f}\\%" + r"\\")
    out += [r"\bottomrule\end{tabular}",
            r"\caption{RQ3 with the pooled dictionary on test (chrF on the held-out dialect). \emph{Unseen}: learned "
            r"from the other four dialects; \emph{Seen}: learned from all five. Recovered = share of the gap between "
            r"copy and seen that is closed without any data from the dialect.}\label{tab:lodo}", r"\end{table}"]
    return "\n".join(out) + "\n", rec


def error_table(path, ft):
    names = {**NAMES, ft: "Qwen-0.5B + LoRA", "dictionary": "Dictionary", "banglat5_cpu1ep": "BanglaT5",
             "qwen3b_zeroshot": "3B zero-shot", "qwen3b_fewshot": "3B 5-shot", "copy": "Copy"}
    out = [r"\begin{table}[t]\centering\footnotesize", r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
           r"System & Exact & Kept & Produced & Novel & Neg.\\\midrule"]
    for r in md_table(path):
        out.append(names.get(r["System"], r["System"]) + " & " + " & ".join(
            r[k] for k in ["Exact %", "Dialect words kept %", "Needed words produced %", "Novel words %",
                           "Negation mismatch %"]) + r"\\")
    out += [r"\bottomrule\end{tabular}",
            r"\caption{Automatic error profile on the 500-pair test subset (\%). \emph{Exact}: output equals the "
            r"reference. \emph{Kept}: dialect words (source words absent from the reference) left unchanged. "
            r"\emph{Produced}: reference words absent from the source that the output contains. \emph{Novel}: "
            r"output words in neither source nor reference. \emph{Neg.}: output and reference disagree on "
            r"negation.}\label{tab:errors}", r"\end{table}"]
    return "\n".join(out) + "\n"


def qual_table(ft, dialects=("chittagong", "sylhet", "barishal")):
    """Real outputs for one sentence per dialect: the first test-subset sentence (by id) with 4-7 words where
    the four systems give four different outputs."""
    import json
    systems = [("qwen3b_fewshot", "3B 5-shot"), (ft, "0.5B+LoRA"), ("dictionary", "Dictionary"),
               ("banglat5_cpu1ep", "BanglaT5")]
    outs = {}
    for sysname, _ in systems:
        f = sorted(Path("outputs_sub500", sysname).glob("*/test.jsonl"))[0]
        outs[sysname] = {json.loads(l)["id"]: json.loads(l) for l in open(f, encoding="utf-8")}
    lines = [r"\begin{table}[t]\centering\footnotesize", r"\begin{tabular}{@{}l@{\hspace{5pt}}p{0.76\columnwidth}@{}}\toprule"]
    for d in dialects:
        cand = [i for i, r in sorted(outs["dictionary"].items()) if r["dialect"] == d
                and 4 <= len(r["tgt"].split()) <= 7 and len({outs[s][i]["hyp"] for s, _ in systems}) == 4]
        i = cand[0]
        r = outs["dictionary"][i]
        lines.append(r"\multicolumn{2}{@{}l}{\emph{" + d.capitalize() + r"}}\\")
        lines.append(r"Source & \bn{" + r["src"] + r"}\\")
        lines.append(r"Reference & \bn{" + r["tgt"] + r"}\\")
        for sysname, label in systems:
            lines.append(label + r" & \bn{" + outs[sysname][i]["hyp"] + r"}\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule\end{tabular}"
    lines += [r"\caption{Real test outputs, one sentence for three of the dialects (chosen automatically: the first "
              r"subset sentence of 4--7 words on which the four systems all differ).}\label{tab:qual}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", default="qwen05b_lora")
    ap.add_argument("--out", default="report/tables")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ft = args.ft
    allc = ["chrF Chit", "chrF Noak", "chrF Sylh", "chrF Bari", "chrF Myme", "chrF all", "BLEU all", "LaBSE all"]
    (out / "main.tex").write_text(results_table(
        "results/test/main_table.md", ["copy", "dictionary", "dictionary_pooled", "banglat5_cpu1ep", ft], ft, allc,
        r"Full Vashantor test set (1,875 pairs, 375 per dialect): chrF per dialect and overall, BLEU and LaBSE. "
        r"Best value per column in bold.", "tab:main"), encoding="utf-8")
    (out / "sub500.tex").write_text(results_table(
        "results/test_sub500/test/main_table.md",
        ["copy", "qwen3b_zeroshot", "qwen3b_fewshot", ft, "dictionary", "banglat5_cpu1ep"], ft, allc,
        r"RQ1 on the fixed 500-pair stratified test subset (100 per dialect): prompting versus fine-tuning. The "
        r"prompted model uses the best instruction and the best 5-shot draw on validation.", "tab:sub500"),
        encoding="utf-8")
    (out / "onubad.tex").write_text(results_table(
        "results/onubad_novel/main_table.md", ["copy", "dictionary", "banglat5_cpu1ep", ft], ft,
        ["chrF Chit", "chrF Sylh", "chrF Bari", "chrF all", "BLEU all", "LaBSE all"],
        r"Out of domain: the 2,437 novel ONUBAD pairs (no sentence shared with Vashantor training).",
        "tab:onubad", short=True).replace("table*", "table").replace(r"\centering\small", r"\centering\footnotesize").replace(
        r"\setlength", ""), encoding="utf-8")
    (out / "sig.tex").write_text(sig_table(
        [("Sub.", "results/test_sub500/test/significance.md"), ("Test", "results/test/significance.md"),
         ("ONU.", "results/onubad_novel/significance.md")], ft,
        r"Paired bootstrap differences in chrF (1,000 resamples) on the 500-pair subset (Sub.), the full test set "
        r"(Test) and novel ONUBAD (ONU.).", "tab:sig"), encoding="utf-8")
    (out / "rq2.tex").write_text(rq2_table("results/test/rq2_all_systems.md",
                                           "results/test_sub500/rq2_all_systems.md", ft), encoding="utf-8")
    lodo, rec = lodo_table("results/dict_ablations/test/main_table.md")
    (out / "lodo.tex").write_text(lodo, encoding="utf-8")
    (out / "errors.tex").write_text(error_table("results/test_sub500/error_profile.md", ft), encoding="utf-8")
    (out / "qual.tex").write_text(qual_table(ft), encoding="utf-8")
    print("wrote", sorted(p.name for p in out.iterdir()), "LODO recovered %:", [round(x) for x in rec])


if __name__ == "__main__":
    main()
