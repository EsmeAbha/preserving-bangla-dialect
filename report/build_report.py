"""Fills report/report_template.tex with the numbers in results/ and writes report/report.tex.

  python src/make_paper_tables.py --ft qwen05b_lora && python report/build_report.py --ft qwen05b_lora
  cd report && xelatex report.tex && xelatex report.tex
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_paper_tables import md_table, num  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def row(path, system):
    return {r["System"]: r for r in md_table(ROOT / path)}[system]


def sig(path, a, b):
    for r in md_table(ROOT / path):
        if r["A"] == a and r["B"] == b:
            lo, hi = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", r["95% CI"].replace("−", "-"))]
            return num(r["A − B"])[0], lo, hi
    raise KeyError((path, a, b))


def f1(x):
    return f"{x:.1f}"


def ci(t):
    d, lo, hi = t
    return f"${d:+.1f}$ chrF, CI $[{lo:+.1f}, {hi:+.1f}]$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", default="qwen05b_lora")
    args = ap.parse_args()
    ft = args.ft
    V = {}
    m = lambda sysname, col, p="results/test/main_table.md": num(row(p, sysname)[col])[0]
    sub = "results/test_sub500/test/main_table.md"
    V["ft_test"] = f1(m(ft, "chrF all"))
    V["ft_labse"] = f"{m(ft, 'LaBSE all'):.2f}"
    V["ft_bleu"] = f1(m(ft, "BLEU all"))
    V["ft_sub"] = f1(m(ft, "chrF all", sub))
    V["zs_sub"] = f1(m("qwen3b_zeroshot", "chrF all", sub))
    V["fs_sub"] = f1(m("qwen3b_fewshot", "chrF all", sub))
    V["copy_sub"] = f1(m("copy", "chrF all", sub))
    V["ft_sub_gain"] = f1(m(ft, "chrF all", sub) - m("qwen3b_fewshot", "chrF all", sub))
    V["dict_test"] = f1(m("dictionary", "chrF all"))
    V["pool_test"] = f1(m("dictionary_pooled", "chrF all"))
    V["bt5_test"] = f1(m("banglat5_cpu1ep", "chrF all"))
    V["copy_test"] = f1(m("copy", "chrF all"))
    V["ft_vs_fs"] = ci(sig("results/test_sub500/test/significance.md", ft, "qwen3b_fewshot"))
    V["ft_vs_zs"] = ci(sig("results/test_sub500/test/significance.md", ft, "qwen3b_zeroshot"))
    V["fs_vs_zs"] = ci(sig("results/test_sub500/test/significance.md", "qwen3b_fewshot", "qwen3b_zeroshot"))
    V["fs_vs_copy"] = ci(sig("results/test_sub500/test/significance.md", "qwen3b_fewshot", "copy"))
    V["zs_vs_copy"] = ci(sig("results/test_sub500/test/significance.md", "qwen3b_zeroshot", "copy"))
    V["ft_vs_dict"] = ci(sig("results/test/significance.md", ft, "dictionary"))
    V["ft_vs_bt5"] = ci(sig("results/test/significance.md", ft, "banglat5_cpu1ep"))
    V["bt5_vs_dict"] = ci(sig("results/test/significance.md", "banglat5_cpu1ep", "dictionary"))
    V["dict_vs_pool"] = ci(sig("results/test/significance.md", "dictionary", "dictionary_pooled"))
    on = "results/onubad_novel/main_table.md"
    V["ft_onu"] = f1(m(ft, "chrF all", on))
    V["dict_onu"] = f1(m("dictionary", "chrF all", on))
    V["bt5_onu"] = f1(m("banglat5_cpu1ep", "chrF all", on))
    V["copy_onu"] = f1(m("copy", "chrF all", on))
    V["ft_drop_onu"] = f1(m(ft, "chrF all") - m(ft, "chrF all", on))
    V["dict_drop_onu"] = f1(m("dictionary", "chrF all") - m("dictionary", "chrF all", on))
    V["bt5_drop_onu"] = f1(m("banglat5_cpu1ep", "chrF all") - m("banglat5_cpu1ep", "chrF all", on))
    V["bt5_vs_dict_onu"] = ci(sig("results/onubad_novel/significance.md", "banglat5_cpu1ep", "dictionary"))
    err = {r["System"]: r for r in md_table(ROOT / "results/test_sub500/error_profile.md")}
    for k, s in [("ft", ft), ("zs", "qwen3b_zeroshot"), ("fs", "qwen3b_fewshot"), ("dict", "dictionary"),
                 ("bt5", "banglat5_cpu1ep")]:
        V[f"{k}_kept"] = err[s]["Dialect words kept %"]
        V[f"{k}_prod"] = err[s]["Needed words produced %"]
        V[f"{k}_novel"] = err[s]["Novel words %"]
        V[f"{k}_neg"] = err[s]["Negation mismatch %"]
    rq2 = {r["System"]: r for r in md_table(ROOT / "results/test_sub500/rq2_all_systems.md")}
    rq2f = {r["System"]: r for r in md_table(ROOT / "results/test/rq2_all_systems.md")}
    V["rho_zs"] = f"{num(rq2['qwen3b_zeroshot']['Sentence-level ρ'])[0]:.2f}"
    V["rho_fs"] = f"{num(rq2['qwen3b_fewshot']['Sentence-level ρ'])[0]:.2f}"
    V["rho_ft"] = f"{num(rq2f[ft]['Sentence-level ρ'])[0]:.2f}"
    V["rho_dict"] = f"{num(rq2f['dictionary']['Sentence-level ρ'])[0]:.2f}"
    V["rho_bt5"] = f"{num(rq2f['banglat5_cpu1ep']['Sentence-level ρ'])[0]:.2f}"
    val = {r["System"]: r for r in md_table(ROOT / "results/val/main_table.md")}
    V["ft_val"] = f1(num(val[ft]["chrF all"])[0]) if ft in val else "--"
    log = ROOT / "checkpoints" / f"{ft}_s1" / "log.json"
    if ft.endswith("_ep05") or not log.exists():
        V["ft_epochs"] = ("the checkpoint after half an epoch (293 optimiser steps, 2 hours); the planned "
                          "two-epoch run was cut short by the compute budget")
        V["ft_short"] = "half an epoch of training"
    else:
        L = json.loads(log.read_text())
        best = max(L["log"], key=lambda e: e["val_chrf"])
        V["ft_epochs"] = (f"2 epochs (1,172 optimiser steps, {L['minutes'] / 60:.1f} hours); the best "
                          f"checkpoint was at epoch {best['epoch']}")
        V["ft_short"] = "two epochs of training"
    t = (ROOT / "report/report_template.tex").read_text(encoding="utf-8")
    for _ in range(3):  # prose blocks contain placeholders themselves
        for k, v in V.items():
            t = t.replace(f"«{k}»", v)
    left = sorted(set(re.findall(r"«([a-z_0-9]+)»", t)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    (ROOT / "report/report.tex").write_text(t, encoding="utf-8")
    print("wrote report/report.tex")


if __name__ == "__main__":
    main()
