import json
import re
import random
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from openai import OpenAI
from tqdm import tqdm

# ── Palette (matches compute_and_plot.py) ────────────────────────────────────
NAVY      = "#21295C"
DEEP_BLUE = "#065A82"
TEAL      = "#1C7293"
GOLD      = "#E8A87C"
INK       = "#1A1F2E"
MUTED     = "#6B7684"
WHITE     = "#FFFFFF"

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

# ── Shared helpers ────────────────────────────────────────────────────────────
SYS_PROMPT = (
    'Based on the triplets retrieved from a knowledge graph, please answer '
    'the question. Please return formatted answers as a list, each prefixed '
    'with "ans:".'
)

ICL_USER = """Triplets:
(San Francisco Giants,sports.sports_team.team_mascot,Lou Seal)
(San Francisco Giants,sports.sports_team.championships,2010 World Series)
(San Francisco Giants,sports.sports_team.championships,2012 World Series)
(San Francisco Giants,sports.sports_team.championships,2014 World Series)

Question:
What year did the team with mascot named Lou Seal win the World Series?"""

ICL_ASSISTANT = (
    "The team with mascot named Lou Seal is the San Francisco Giants.\n"
    "The San Francisco Giants won the World Series in the following years:\n"
    "ans: 2010\nans: 2012\nans: 2014"
)

def normalize(s: str) -> str:
    return " ".join(s.lower().split())

def parse_answers(text: str) -> list:
    matches = re.findall(r"ans:\s*(.+?)(?:\n|$)", text)
    return [normalize(m.strip()) for m in matches if m.strip()]

def compute_f1(pred_set: set, gold_set: set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(gold_set)
    return 2 * p * r / (p + r)

# ── Load the 150 results ──────────────────────────────────────────────────────
with open("results_webqsp.json", encoding="utf-8") as f:
    data = json.load(f)
pq = data["per_question"]
assert len(pq) == 150

# ── Sample 30 from the 150 ───────────────────────────────────────────────────
random.seed(42)
sample_30 = random.sample(pq, 30)
sample_ids = {q["id"] for q in sample_30}
print(f"Sampled 30 questions (seed=42) from 150: {[q['id'] for q in sample_30[:5]]}...")

# ── With-retrieval baseline (same 30, from existing results) ─────────────────
wr_f1s  = [q["f1"]  for q in sample_30]
wr_hits = [q["hit"] for q in sample_30]
wr_avg_f1  = sum(wr_f1s)  / len(wr_f1s)
wr_avg_hit = sum(wr_hits) / len(wr_hits)
print(f"With-retrieval (same 30): F1={wr_avg_f1*100:.2f}%  Hit={wr_avg_hit*100:.2f}%")

# ── No-retrieval inference ────────────────────────────────────────────────────
if os.path.exists("results_no_retrieval.json"):
    print("\nresults_no_retrieval.json already exists — loading cached results.")
    with open("results_no_retrieval.json", encoding="utf-8") as f:
        nr_data = json.load(f)
    nr_results = nr_data["per_question"]
else:
    print("\nRunning no-retrieval GPT-4o-mini inference on 30 questions...")
    client = OpenAI()
    nr_results = []
    errors = 0

    for q in tqdm(sample_30, desc="No-retrieval inference"):
        user_msg = f"Question:\n{q['question']}"
        llm_response = ""
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",    "content": SYS_PROMPT},
                    {"role": "user",      "content": ICL_USER},
                    {"role": "assistant", "content": ICL_ASSISTANT},
                    {"role": "user",      "content": user_msg},
                ],
                temperature=0,
                seed=42,
                max_tokens=256,
            )
            llm_response = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"\n  ERROR on {q['id']}: {e}")
            errors += 1

        predicted = parse_answers(llm_response)
        gold      = q["gold"]
        pred_set  = set(predicted)
        gold_set  = set(gold)
        f1        = compute_f1(pred_set, gold_set)
        hit       = 1 if pred_set & gold_set else 0

        nr_results.append({
            "id": q["id"], "question": q["question"],
            "gold": gold, "predicted": predicted,
            "f1": f1, "hit": hit, "llm_response": llm_response,
        })

    nr_output = {
        "config": {"model": "gpt-4o-mini", "mode": "no_retrieval",
                   "sample_size": 30, "seed": 42, "api_errors": errors},
        "per_question": nr_results,
    }
    with open("results_no_retrieval.json", "w", encoding="utf-8") as f:
        json.dump(nr_output, f, indent=2)
    print(f"  Saved results_no_retrieval.json  (errors: {errors})")

# ── Compute no-retrieval metrics ──────────────────────────────────────────────
nr_f1s  = [r["f1"]  for r in nr_results]
nr_hits = [r["hit"] for r in nr_results]
nr_avg_f1  = sum(nr_f1s)  / len(nr_f1s)
nr_avg_hit = sum(nr_hits) / len(nr_hits)

print(f"\n{'='*50}")
print("ABLATION RESULTS")
print(f"{'='*50}")
print(f"  {'Condition':<22}  {'F1':>8}  {'Hit':>8}")
print(f"  {'-'*40}")
print(f"  {'With retrieval':<22}  {wr_avg_f1*100:>7.2f}%  {wr_avg_hit*100:>7.2f}%")
print(f"  {'No retrieval':<22}  {nr_avg_f1*100:>7.2f}%  {nr_avg_hit*100:>7.2f}%")
print(f"  {'Delta':<22}  {(wr_avg_f1-nr_avg_f1)*100:>+7.2f}pp {(wr_avg_hit-nr_avg_hit)*100:>+7.2f}pp")
print(f"{'='*50}\n")

# ── Chart 5: ablation_no_retrieval.png ───────────────────────────────────────
groups = ["With retrieval", "No retrieval"]
f1_vals_chart  = [wr_avg_f1,  nr_avg_f1]
hit_vals_chart = [wr_avg_hit, nr_avg_hit]

x = np.arange(len(groups))
width = 0.30

fig, ax = plt.subplots(figsize=(14, 7))
style_ax(ax)

bars_f1  = ax.bar(x - width/2, f1_vals_chart,  width, label="Macro-F1", color=DEEP_BLUE, zorder=3)
bars_hit = ax.bar(x + width/2, hit_vals_chart, width, label="Hit",      color=TEAL,      zorder=3)

for bar, val in zip(bars_f1, f1_vals_chart):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
            f"{val:.2f}", ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")

for bar, val in zip(bars_hit, hit_vals_chart):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
            f"{val:.2f}", ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")

# Delta annotations
delta_f1  = wr_avg_f1  - nr_avg_f1
delta_hit = wr_avg_hit - nr_avg_hit
ax.annotate(
    f"+{delta_f1*100:.1f}pp\nfrom KG",
    xy=(x[0] - width/2, wr_avg_f1),
    xytext=(x[0] - width/2 - 0.28, wr_avg_f1 + 0.06),
    fontsize=9, color=NAVY, ha="center",
    arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
)
ax.annotate(
    f"+{delta_hit*100:.1f}pp\nfrom KG",
    xy=(x[0] + width/2, wr_avg_hit),
    xytext=(x[0] + width/2 + 0.32, wr_avg_hit + 0.06),
    fontsize=9, color=TEAL, ha="center",
    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2),
)

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=13, color=INK)
ax.set_ylabel("Score", fontsize=12, color=INK)
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=13, length=0)

ax.set_title("Ablation: effect of KG retrieval (n=30)",
             fontsize=18, fontweight="bold", color=INK, pad=16)
ax.legend(loc="upper center", ncol=2, fontsize=11, frameon=False,
          bbox_to_anchor=(0.5, 1.0))

fig.tight_layout()
fig.savefig("ablation_no_retrieval.png", dpi=300, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("  Saved ablation_no_retrieval.png")
