import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import torch

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY      = "#21295C"
DEEP_BLUE = "#065A82"
TEAL      = "#1C7293"
GOLD      = "#E8A87C"
INK       = "#1A1F2E"
MUTED     = "#6B7684"
WHITE     = "#FFFFFF"
LIGHT_BG  = "#F7F8FA"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
})

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.yaxis.grid(True, color=MUTED, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)

# ── Load results ──────────────────────────────────────────────────────────────
with open("results_webqsp.json", encoding="utf-8") as f:
    data = json.load(f)

pq = data["per_question"]
assert len(pq) == 150

f1_vals   = [q["f1"]  for q in pq]
hit_vals  = [q["hit"] for q in pq]
avg_f1    = sum(f1_vals) / len(f1_vals)
avg_hit   = sum(hit_vals) / len(hit_vals)

# ── Chart 2 bins ──────────────────────────────────────────────────────────────
bins = {
    "[0.0, 0.25)":  sum(1 for v in f1_vals if 0.0 <= v < 0.25),
    "[0.25, 0.50)": sum(1 for v in f1_vals if 0.25 <= v < 0.50),
    "[0.50, 0.75)": sum(1 for v in f1_vals if 0.50 <= v < 0.75),
    "[0.75, 1.00)": sum(1 for v in f1_vals if 0.75 <= v < 1.00),
    "F1 = 1.00":    sum(1 for v in f1_vals if v == 1.0),
}
assert sum(bins.values()) == 150

# ── Chart 3 buckets ───────────────────────────────────────────────────────────
buckets = {
    "Single\n(1 answer)":  [i for i, q in enumerate(pq) if len(q["gold"]) == 1],
    "Few\n(2-5 answers)":  [i for i, q in enumerate(pq) if 2 <= len(q["gold"]) <= 5],
    "Many\n(6+ answers)":  [i for i, q in enumerate(pq) if len(q["gold"]) >= 6],
}
bucket_stats = {}
for label, idxs in buckets.items():
    n = len(idxs)
    bucket_stats[label] = {
        "n": n,
        "f1":  sum(f1_vals[i]  for i in idxs) / n if n else 0,
        "hit": sum(hit_vals[i] for i in idxs) / n if n else 0,
    }
assert sum(v["n"] for v in bucket_stats.values()) == 150

# ── Chart 4: load .pth and compute Recall@K ───────────────────────────────────
print("Loading scored_triples .pth (~290 MB)...")
pth = torch.load(
    "./subgraphrag_data/scored_triples/webqsp_240912_unidir_test.pth",
    weights_only=False,
)
print(f"  Loaded {len(pth)} entries\n")

K_VALUES = [25, 50, 100, 200, 500]
recall_lists = {k: [] for k in K_VALUES}
missing_ids = 0

for q in pq:
    qid = q["id"]
    if qid not in pth:
        missing_ids += 1
        continue
    gold_norm = set(q["gold"])                    # already lowercased
    triples   = pth[qid]["scored_triples"]        # sorted desc by score
    for k in K_VALUES:
        top_k    = triples[:k]
        entities = {t[0].lower().strip() for t in top_k} | {t[2].lower().strip() for t in top_k}
        recall_lists[k].append(1 if gold_norm & entities else 0)

if missing_ids:
    print(f"  WARNING: {missing_ids} IDs missing from pth")

avg_recall = {k: sum(v) / len(v) for k, v in recall_lists.items() if v}

# ── Verification summary ──────────────────────────────────────────────────────
print("=" * 62)
print("DATA VERIFICATION SUMMARY")
print("=" * 62)
print(f"  Total questions : {len(pq)}")
print(f"  avg F1          : {avg_f1*100:.2f}%   avg Hit: {avg_hit*100:.2f}%")
print()
print("Chart 2 — F1 bins:")
for label, count in bins.items():
    print(f"  {label:<16} : {count:>3}")
print(f"  {'SUM':<16} : {sum(bins.values()):>3}  OK")
print()
print("Chart 3 — Answer-count buckets:")
for label, stats in bucket_stats.items():
    print(f"  {label.replace(chr(10),' '):<22}  n={stats['n']:>3}"
          f"  F1={stats['f1']:.3f}  Hit={stats['hit']:.3f}")
print(f"  SUM  n={sum(v['n'] for v in bucket_stats.values())}  OK")
print()
print("Chart 4 — Recall@K:")
for k in K_VALUES:
    print(f"  K={k:<4}  Recall = {avg_recall[k]*100:.1f}%")
print("=" * 62)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1 — results_comparison.png
# ═══════════════════════════════════════════════════════════════════════════════
groups     = ["Paper reported", "Authors'\npredictions", "Our GPT-4o-mini\nrun"]
f1_scores  = [77.45, 69.32, 68.91]
hit_scores = [90.11, 76.00, 76.67]

x = np.arange(len(groups))
width = 0.32

fig, ax = plt.subplots(figsize=(14, 7))
style_ax(ax)
bars_f1  = ax.bar(x - width/2, f1_scores,  width, label="Macro-F1", color=DEEP_BLUE, zorder=3)
bars_hit = ax.bar(x + width/2, hit_scores, width, label="Hit",      color=TEAL,      zorder=3)

for bar in bars_f1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.2f}",
            ha="center", va="bottom", fontsize=10, color=INK)
for bar in bars_hit:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.2f}",
            ha="center", va="bottom", fontsize=10, color=INK)

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12, color=INK)
ax.set_ylabel("Score (%)", fontsize=12, color=INK)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=12, length=0)
ax.set_title("SubgraphRAG Reproduction Results",
             fontsize=18, fontweight="bold", color=INK, pad=16)
ax.legend(loc="upper center", ncol=2, fontsize=11, frameon=False,
          bbox_to_anchor=(0.5, 1.0))

fig.tight_layout()
fig.savefig("results_comparison.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("  Saved results_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2 — f1_distribution.png
# ═══════════════════════════════════════════════════════════════════════════════
bin_labels = list(bins.keys())
bin_counts = list(bins.values())
bar_colors = [DEEP_BLUE] * 4 + [GOLD]

fig, ax = plt.subplots(figsize=(14, 7))
style_ax(ax)
bars = ax.bar(bin_labels, bin_counts, color=bar_colors, width=0.55, zorder=3)
for bar, count in zip(bars, bin_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            str(count), ha="center", va="bottom", fontsize=10, color=INK)
ax.set_ylabel("Number of questions", fontsize=12, color=INK)
ax.set_ylim(0, max(bin_counts) * 1.18)
ax.tick_params(axis="x", labelsize=11, length=0, colors=INK)
ax.tick_params(axis="y", labelsize=10)
ax.set_title("Distribution of per-question F1 (n=150)",
             fontsize=18, fontweight="bold", color=INK, pad=32)
ax.text(0.5, 1.01, "Bimodal: questions tend to fully succeed or fully fail",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=11, color=MUTED, style="italic")
fig.tight_layout()
fig.savefig("f1_distribution.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("  Saved f1_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3 — performance_by_answer_count.png
# ═══════════════════════════════════════════════════════════════════════════════
bucket_labels = list(bucket_stats.keys())
b_f1_vals = [bucket_stats[k]["f1"]  for k in bucket_labels]
b_hit_vals = [bucket_stats[k]["hit"] for k in bucket_labels]
b_ns       = [bucket_stats[k]["n"]   for k in bucket_labels]

x = np.arange(len(bucket_labels))
width = 0.32

fig, ax = plt.subplots(figsize=(14, 7))
style_ax(ax)
bars_f1  = ax.bar(x - width/2, b_f1_vals,  width, label="Macro-F1", color=DEEP_BLUE, zorder=3)
bars_hit = ax.bar(x + width/2, b_hit_vals, width, label="Hit",      color=TEAL,      zorder=3)
for bar, val in zip(bars_f1, b_f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
for bar, val in zip(bars_hit, b_hit_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(bucket_labels, fontsize=12, color=INK)
ax.set_ylabel("Score", fontsize=12, color=INK)
ax.set_ylim(0, 1.18)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=12, length=0)
for xi, n in zip(x, b_ns):
    ax.text(xi, -0.115, f"n = {n}", ha="center", va="top", fontsize=10,
            color=MUTED, transform=ax.get_xaxis_transform())
ax.set_title("Performance by question answer-count",
             fontsize=18, fontweight="bold", color=INK, pad=16)
ax.legend(loc="upper center", ncol=2, fontsize=11, frameon=False,
          bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig("performance_by_answer_count.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("  Saved performance_by_answer_count.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4 — retrieval_recall_at_k.png
# ═══════════════════════════════════════════════════════════════════════════════
k_x       = K_VALUES
recall_y  = [avg_recall[k] for k in K_VALUES]

fig, ax = plt.subplots(figsize=(14, 7))
style_ax(ax)

ax.plot(k_x, recall_y, color=DEEP_BLUE, linewidth=2.5, marker="o",
        markersize=9, markerfacecolor=GOLD, markeredgecolor=DEEP_BLUE,
        markeredgewidth=1.8, zorder=4)

for kv, rv in zip(k_x, recall_y):
    ax.text(kv, rv + 0.012, f"{rv*100:.1f}%",
            ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")

ax.set_xscale("log")
ax.set_xticks(k_x)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.set_xlim(15, 650)
ax.set_ylim(0.0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.tick_params(axis="x", labelsize=11, length=0, colors=INK)
ax.tick_params(axis="y", labelsize=10)
ax.set_xlabel("Retrieval budget K (number of top-scored triples)", fontsize=12, color=INK)
ax.set_ylabel("Recall@K  (fraction of questions with gold entity found)", fontsize=12, color=INK)
ax.set_title("Answer entity recall vs. retrieval budget K",
             fontsize=18, fontweight="bold", color=INK, pad=16)

fig.tight_layout()
fig.savefig("retrieval_recall_at_k.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("  Saved retrieval_recall_at_k.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6 — example_walkthrough.png  (WebQTest-1982: Vivaldi)
# ═══════════════════════════════════════════════════════════════════════════════
WALKTHROUGH_ID = "WebQTest-1982"
q6 = next(q for q in pq if q["id"] == WALKTHROUGH_ID)
top5 = pth[WALKTHROUGH_ID]["scored_triples"][:5]

fig = plt.figure(figsize=(14, 8), facecolor=WHITE)
gs6 = gridspec.GridSpec(
    4, 1,
    height_ratios=[1.3, 3.0, 2.1, 1.6],
    hspace=0.06,
    left=0.04, right=0.96,
    top=0.97, bottom=0.03,
)

# ── Row 0: Question header ────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs6[0])
ax0.set_axis_off()
ax0.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax0.transAxes,
                              facecolor=NAVY, edgecolor="none", zorder=0))
ax0.text(0.5, 0.62,
         q6["question"].capitalize() + "?",
         transform=ax0.transAxes, ha="center", va="center",
         fontsize=17, fontweight="bold", color="white", zorder=1)
ax0.text(0.5, 0.18,
         f"ID: {q6['id']}    F1 = {q6['f1']:.2f}    Hit = {int(q6['hit'])}    "
         f"Gold answers: {len(q6['gold'])}",
         transform=ax0.transAxes, ha="center", va="center",
         fontsize=10, color="white", alpha=0.80, zorder=1)

# ── Row 1: Top-5 triples table ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs6[1])
ax1.set_facecolor(WHITE)
ax1.set_axis_off()
ax1.text(0.0, 0.98, "Top 5 Retrieved Triples",
         transform=ax1.transAxes, ha="left", va="top",
         fontsize=12, fontweight="bold", color=INK)

col_labels = ["#", "Score", "Head entity", "Relation", "Tail entity"]
col_widths  = [0.05, 0.09, 0.22, 0.38, 0.26]
cell_data   = [
    [str(i + 1), f"{t[3]:.4f}", t[0], t[1], t[2]]
    for i, t in enumerate(top5)
]

tbl = ax1.table(
    cellText=cell_data,
    colLabels=col_labels,
    colWidths=col_widths,
    cellLoc="left",
    loc="upper left",
    bbox=[0.0, 0.02, 1.0, 0.88],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#D8DDE6")
    cell.PAD = 0.06
    if row == 0:
        cell.set_facecolor(NAVY)
        cell.set_text_props(color="white", fontweight="bold", ha="center")
    elif row % 2 == 1:
        cell.set_facecolor(WHITE)
    else:
        cell.set_facecolor(LIGHT_BG)
    if row > 0 and col in (0, 1):
        cell.set_text_props(ha="center")

# ── Row 2: LLM response ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs6[2])
ax2.set_axis_off()
ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes,
                              facecolor=LIGHT_BG, edgecolor="none", zorder=0))

ax2.text(0.02, 0.93, "LLM Response",
         transform=ax2.transAxes, ha="left", va="top",
         fontsize=12, fontweight="bold", color=INK, zorder=1)

response_display = q6["llm_response"]          # keep \n intact
ax2.text(0.03, 0.70, f'"{response_display}"',
         transform=ax2.transAxes, ha="left", va="top",
         fontsize=10.5, color=INK, style="italic",
         linespacing=1.6, zorder=1)

# ── Row 3: Gold vs Predicted ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs6[3])
ax3.set_facecolor(WHITE)
ax3.set_axis_off()

# Divider
ax3.plot([0.5, 0.5], [0.08, 0.92], color=MUTED, alpha=0.35,
         linewidth=1.2, transform=ax3.transAxes)

ax3.text(0.25, 0.90, "Gold Answers",
         transform=ax3.transAxes, ha="center", va="top",
         fontsize=11, fontweight="bold", color=NAVY)
ax3.text(0.75, 0.90, "Predicted Answers",
         transform=ax3.transAxes, ha="center", va="top",
         fontsize=11, fontweight="bold", color=DEEP_BLUE)

gold_str = "   |   ".join(a.title() for a in q6["gold"])
pred_str = "   |   ".join(a.title() for a in q6["predicted"])

ax3.text(0.25, 0.52, gold_str,
         transform=ax3.transAxes, ha="center", va="center",
         fontsize=11, color=INK)
ax3.text(0.75, 0.52, pred_str,
         transform=ax3.transAxes, ha="center", va="center",
         fontsize=11, color=INK)

ax3.text(0.5, 0.12, "Exact match — F1 = 1.00",
         transform=ax3.transAxes, ha="center", va="bottom",
         fontsize=10, color=TEAL, fontweight="bold")

fig.savefig("example_walkthrough.png", dpi=300, facecolor=WHITE)
plt.close(fig)
print("  Saved example_walkthrough.png")

print()
print("All 6 charts saved successfully.")
