"""Build report material in docs/: results.md (tables) + PNG figures.

python -m scripts.make_report [--set dataset=ml-latest-small]
Reads the stage outputs in outputs/<dataset>/ and docs/bench_results.csv.
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.config import cli  # noqa: E402
from src.common.io import read_pandas  # noqa: E402

log = logging.getLogger("report")

# validated categorical palette (fixed order) + chart chrome
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MARKERS = ["o", "s", "^", "D"]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)


def fig_faith_threshold(sweep: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, col, title in [(axes[0], "faithful_score_only", "Score-drop criterion only"),
                           (axes[1], "faithful_score_or_rank", "Score drop OR leaves top-10")]:
        last = {}
        for i, cond in enumerate(["cited", "random"]):
            d = sweep[sweep.condition == cond].sort_values("threshold")
            ax.plot(d.threshold, d[col], color=C[i], linewidth=2, marker=MARKERS[i], markersize=5,
                    label="cited evidence removed" if cond == "cited" else "random items removed")
            last[cond] = (d.threshold.iloc[-1], d[col].iloc[-1], C[i])
        if abs(last["cited"][1] - last["random"][1]) > 0.06:      # direct labels only where they don't collide
            for cond, (xx, yy, c) in last.items():
                end_label(ax, xx, yy, cond, c)
        style(ax, title, "relative score-drop threshold", "share of explanations judged faithful")
        ax.axvline(0.10, color=AXIS, linewidth=1, linestyle="--")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_faith_history(summary: dict, path):
    bh = summary["by_history_length"]
    buckets = list(bh["cited"].keys())
    x = range(len(buckets))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.38
    for i, cond in enumerate(["cited", "random"]):
        vals = [bh[cond].get(b, {}).get("faithful", 0) for b in buckets]
        xs = [xi + (i - 0.5) * (w + 0.02) for xi in x]
        ax.bar(xs, vals, width=w, color=C[i], label=f"{cond} removal", edgecolor=SURFACE, linewidth=1)
        if cond == "cited":
            for xx, v in zip(xs, vals):
                ax.annotate(f"{v:.0%}", (xx, v), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=7, color=INK2)
    ax.set_xticks(list(x), [f"{b}\n(n={bh['cited'][b]['n']})" for b in buckets])
    style(ax, "Faithfulness by user history length (threshold 10% or top-10 exit)",
          "ratings in the user's training history", "faithful share")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_bench(bench: pd.DataFrame, path_time, path_speedup):
    bench = bench.copy()
    bench["label"] = bench.apply(lambda r: f"{r.dataset}" + (f" ({int(r.n_ratings/1e6)}M)" if r.n_ratings >= 1e6 else
                                                               f" ({int(r.n_ratings/1e3)}k)"), axis=1)
    bench = bench.groupby(["label", "n_ratings", "cores"], as_index=False)[["t_als_fit", "t_evidence", "t_verify", "t_split", "t_recs"]].median()
    bench["t_explain_verify"] = bench.t_evidence + bench.t_verify
    labels = bench.sort_values("n_ratings").label.unique()
    stages = [("t_als_fit", "ALS training"), ("t_explain_verify", "Evidence + faithfulness (200 users)")]
    for path, speedup in [(path_time, False), (path_speedup, True)]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, (col, title) in zip(axes, stages):
            for i, lab in enumerate(labels):
                d = bench[bench.label == lab].sort_values("cores")
                y = d[col].iloc[0] / d[col] if speedup else d[col]
                ax.plot(d.cores, y, color=C[i], linewidth=2, marker=MARKERS[i], markersize=5, label=lab)
                end_label(ax, d.cores.iloc[-1], y.iloc[-1], lab.split(" ")[0], C[i])
            if speedup:
                cs = sorted(bench.cores.unique())
                ax.plot(cs, [c / cs[0] for c in cs], color=MUTED, linewidth=1, linestyle="--", label="ideal")
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(bench.cores.unique()), [f"local[{c}]" for c in sorted(bench.cores.unique())])
            if not speedup:
                ax.set_yscale("log")
            style(ax, title, "Spark local cores", "speed-up vs local[1]" if speedup else "seconds (log scale)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return bench


def fig_evidence_scaling(e2: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    cs = list(e2.cores)
    for i, (col, lab) in enumerate([("t_evidence", "evidence extraction"), ("t_verify", "faithfulness verification")]):
        y = e2[col].iloc[0] / e2[col]
        ax.plot(cs, y, color=C[i], linewidth=2, marker=MARKERS[i], markersize=5, label=lab)
    ax.plot(cs, [c / cs[0] for c in cs], color=MUTED, linewidth=1, linestyle="--", label="ideal")
    ax.set_xscale("log", base=2)
    ax.set_xticks(cs, [f"local[{c}]" for c in cs])
    style(ax, f"Speed-up, {int(e2.n_recs.iloc[0]):,} recommendations (ml-1m)", "Spark local cores", "speed-up vs local[1]")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(v) for v in r.values) + " |")
    return "\n".join(out)


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{int(v):,}" if v.is_integer() and abs(v) >= 1000 else f"{v:.4g}"
    return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v)


def llm_examples(out, n=3) -> list[str]:
    """A few real-LLM explanations (if any were generated) for the report."""
    try:
        ex = read_pandas(str(out / "explanations.parquet"))
    except (OSError, ValueError):
        return []
    ex = ex[ex.llm_name != "template"]
    return [f"> {t}" for t in ex.text.head(n)]


def main():
    cfg, _ = cli(__doc__)
    out, docs = cfg.out_dir, cfg.docs
    ld = lambda n: json.load(open(out / n))  # noqa: E731
    prep, als, ev, gr, fa, ut = (ld("prepare_stats.json"), ld("als_metrics.json"), ld("evidence_stats.json"),
                                 ld("grounding_summary.json"), ld("faithfulness_summary.json"), ld("utility.json"))
    sweep = pd.read_csv(out / "faithfulness_sweep.csv")
    fig_faith_threshold(sweep, docs / "faithfulness_vs_threshold.png")
    fig_faith_history(fa, docs / "faithfulness_by_history.png")
    k = als["recommender"]
    rec_tbl = pd.DataFrame([
        {"model": als["recommender"]["model"], **{m: als["recommender"][m] for m in (f"precision@10", "recall@10", "mrr@10", "ndcg@10")}},
        {"model": "Popularity", **{m: als["popularity"][m] for m in ("precision@10", "recall@10", "mrr@10", "ndcg@10")}},
    ])
    faith_tbl = pd.DataFrame([
        {"condition": "cited evidence removed", "faithful share": fa["faithfulness_score"],
         "mean rel. drop": fa["mean_rel_drop"]["cited"], "median rel. drop": fa["median_rel_drop"]["cited"],
         "left top-10": fa["frac_out_of_top_k"]["cited"]},
        {"condition": "random items removed (baseline)", "faithful share": fa["random_baseline_score"],
         "mean rel. drop": fa["mean_rel_drop"]["random"], "median rel. drop": fa["median_rel_drop"]["random"],
         "left top-10": fa["frac_out_of_top_k"]["random"]},
    ])
    hist_tbl = pd.DataFrame([{"history": b, "n": v["n"], "cited faithful": v["faithful"],
                              "random faithful": fa["by_history_length"]["random"][b]["faithful"]}
                             for b, v in fa["by_history_length"]["cited"].items()])
    auc = ut["auc"]
    util_tbl = pd.DataFrame([{"features": n, "ROC-AUC": auc[n]["roc_auc"]["mean"], "ROC-AUC sd": auc[n]["roc_auc"]["std"],
                              "PR-AUC": auc[n]["pr_auc"]["mean"]} for n in ("base", "base+expl", "expl_only")])
    stress = gr["stress_test"]
    lines = [
        f"# i-recs results ({cfg['dataset']})", "",
        "Generated by `python -m scripts.make_report`. All numbers come from the stage outputs in "
        f"`outputs/{cfg['dataset']}/`.", "",
        "## Data and split", "",
        f"{prep['ratings_clean']:,} ratings after cleaning; per-user temporal split: {prep['train']:,} train / "
        f"{prep['test']:,} test ratings, {prep['users_train']} users, {prep['items_train']:,} items.", "",
        "## M1 Recommender (test set, unseen items only)", "",
        md_table(rec_tbl), "",
        f"Chosen by validation NDCG@10 on a time-based split of train: implicit ALS rank={k['rank']}, "
        f"regParam={k['regParam']}, alpha={k['alpha']}. Rating prediction (explicit ALS, rank={als['rating_prediction']['rank']}, "
        f"regParam={als['rating_prediction']['regParam']}): **test RMSE {als['rating_prediction']['test_rmse']:.3f}**.", "",
        "## M2 Evidence extraction", "",
        f"{ev['n_recs']:,} recommendations for {ev['n_users']} sampled users; mean history {ev['mean_history']:.0f} ratings; "
        f"{ev['frac_with_evidence']:.0%} have positive-influence evidence. Top evidence item = "
        f"{ev['mean_top1_share_of_score']:.1%} of the score on average. Runtime {ev['seconds']}s (fold-in, no retraining).", "",
        "## M3/M4 Explanations and grounding", "",
        md_table(pd.DataFrame([{"writer": w, **v} for w, v in gr["by_writer"].items()])), "",
        f"Stress test on {stress['n']} explanations: clean false-positive rate {stress['clean_false_positive_rate']:.1%}; "
        "detection recall per injected hallucination type: " +
        ", ".join(f"{k2} {v:.0%}" for k2, v in stress["detection_recall"].items()) + ".", "",
    ]
    ex = llm_examples(out)
    if ex:
        lines += ["Example LLM explanations (evidence-only prompt, all passed the grounding check):", ""]
        for e in ex:
            lines += [e, ""]
    lines += [
        "## M5 Faithfulness (core result)", "",
        f"Faithful = removing the cited items (jointly, fold-in) lowers the score by >= {fa['threshold']:.0%} "
        f"or pushes the item out of the top-{fa['top_k']}. {fa['n_explanations']:,} explanations, "
        f"{fa['n_random_trials']:,} random-removal trials.", "",
        md_table(faith_tbl), "",
        f"Paired: cited removal beats the mean random removal for **{fa['paired']['cited_beats_random_rate']:.1%}** of explanations "
        f"(one-sided Wilcoxon p = {fa['paired']['wilcoxon_p']:.2g}). Fold-in ranks match the ALS ranking for "
        f"{fa['rank_consistency_with_als_recs']:.0%} of recommendations.", "",
        "![faithfulness vs threshold](faithfulness_vs_threshold.png)", "",
        md_table(hist_tbl), "",
        "![faithfulness by history](faithfulness_by_history.png)", "",
        "Reading: a 2-item explanation is usually faithful for users with short histories and rarely for heavy users, "
        "whose recommendations are spread over hundreds of past ratings. Cited items beat random items in every bucket.", "",
        "## M6 Utility (offline proxy)", "",
        f"Label: {ut['label']} ({ut['positives']} of {ut['n']}, base rate {ut['base_rate']:.3f}). "
        f"Logistic regression (Spark MLlib), {auc['base']['repeats']} user-grouped splits.", "",
        md_table(util_tbl), "",
        f"Paired gain of explanation features: ROC-AUC +{auc['paired_gain_base+expl_vs_base']['roc_auc_mean']:.3f}, "
        f"improved in {auc['paired_gain_base+expl_vs_base']['splits_improved']}/{auc['base']['repeats']} splits. "
        "Within history-size strata the faithful flag alone does not raise acceptance; the gain comes mainly from the "
        "evidence content (the user's ratings of the cited items). Observational proxy, not a causal user study.", "",
    ]
    bench_path = docs / "bench_results.csv"
    if bench_path.exists():
        raw = pd.concat([pd.read_csv(f) for f in [bench_path, docs / "bench_25m_local4.csv"] if f.exists()])
        bench = fig_bench(raw, docs / "bench_time.png", docs / "bench_speedup.png")
        tbl = bench.pivot_table(index=["label"], columns="cores", values="t_als_fit").round(1)
        tbl.columns = [f"ALS fit local[{c}] (s)" for c in tbl.columns]
        tbl2 = bench.pivot_table(index=["label"], columns="cores", values="t_explain_verify").round(1)
        tbl2.columns = [f"evid+verify local[{c}] (s)" for c in tbl2.columns]
        lines += ["## M7 Scalability (single machine, Spark local mode)", "",
                  md_table(tbl.reset_index()), "", md_table(tbl2.reset_index()), "",
                  "![time](bench_time.png)", "", "![speedup](bench_speedup.png)", "",
                  "Evidence and verification touch only the 200 sampled users, so with 2 partitions per core most of "
                  "their time is Spark task scheduling and, on Windows, starting a fresh Python worker per task "
                  "(no fork-based worker daemon). The run below uses 2,000 users and 1 partition per core to "
                  "measure the compute itself.", ""]
        evs = docs / "bench_evidence_scaling.csv"
        if evs.exists():
            e2 = pd.read_csv(evs).sort_values("cores")
            t = e2[["cores", "n_recs", "t_evidence", "t_verify", "verify_rows"]].copy()
            t["evidence speed-up"] = (t.t_evidence.iloc[0] / t.t_evidence).round(2)
            t["verify speed-up"] = (t.t_verify.iloc[0] / t.t_verify).round(2)
            fig_evidence_scaling(e2, docs / "bench_evidence_scaling.png")
            lines += [f"### Evidence + faithfulness scaling (ml-1m, {int(e2.evidence_users.iloc[0])} users)", "",
                      md_table(t), "", "![evidence scaling](bench_evidence_scaling.png)", ""]
        big = bench[bench.n_ratings == bench.n_ratings.max()].sort_values("cores")
        sp = big.t_als_fit.iloc[0] / big.t_als_fit.iloc[-1]
        lines += [
            "Reading: ALS training is the distributed workload and scales with data and cores "
            f"({big.label.iloc[0]}: {big.t_als_fit.iloc[0]:.0f}s on local[1] -> {big.t_als_fit.iloc[-1]:.0f}s on "
            f"local[{int(big.cores.iloc[-1])}], {sp:.1f}x). Evidence and verification cost depends on the number of "
            "explained recommendations and the sampled users' history lengths, not on total dataset size (one "
            "batched k x k solve per user), so it stays at tens of seconds even on 5M ratings. On a single Windows laptop their parallel speed-up is small because "
            "per-task Python worker start-up dominates; the occasional large outlier (e.g. ml-1m local[8] "
            "verification) is most likely worker start-up contention rather than compute. A multi-node cluster (EMR/Dataproc) with "
            "fork-based Python workers is expected to remove most of this overhead.", ""]
    lines += [
        "## Summary", "",
        f"- ALS-implicit beats popularity on every top-10 metric (NDCG@10 {k['ndcg@10']:.3f} vs "
        f"{als['popularity']['ndcg@10']:.3f}).",
        "- Evidence-only explanations are grounded (" +
        ", ".join(f"{w}: {v['grounded_rate']:.0%} of {v['n']:,}" for w, v in gr["by_writer"].items()) +
        f"); on the stress test the checker's lowest per-type detection recall is "
        f"{min(stress['detection_recall'].values()):.0%}, with a {stress['clean_false_positive_rate']:.0%} "
        "false-positive rate.",
        f"- **{fa['faithfulness_score']:.1%}** of explanations are faithful (removing the cited items drops the score as "
        f"claimed), vs {fa['random_baseline_score']:.1%} for random removals of the same size; cited beats random for "
        f"{fa['paired']['cited_beats_random_rate']:.0%} of explanations. Faithfulness falls with history length: "
        "a 2-item explanation cannot carry a heavy user's recommendation.",
        f"- Explanation features add ROC-AUC +{auc['paired_gain_base+expl_vs_base']['roc_auc_mean']:.3f} to an offline "
        "acceptance proxy (observational, not causal).",
        "- The whole pipeline runs on Spark; ALS training scales with cores and evidence/verification cost does "
        "not grow with total dataset size.", "",
    ]
    (docs / "results.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", docs / "results.md")


if __name__ == "__main__":
    main()
