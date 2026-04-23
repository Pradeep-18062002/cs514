import json
import re
import random
import os
from openai import OpenAI
from tqdm import tqdm

PREDICTIONS_FILE = (
    "./subgraphrag_data/results/KGQA/webqsp/SubgraphRAG/"
    "gpt-4o-mini/scored_100-sys_icl_dc-0-thres_0.0-test-predictions.jsonl"
)
RESULTS_FILE = "results_webqsp.json"
SAMPLE_SIZE = 150
TOP_K = 100

# Paper Appendix E Figure 8 ICL example
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
    "ans: 2010\n"
    "ans: 2012\n"
    "ans: 2014"
)


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def compute_f1(pred_set: set, gold_set: set) -> float:
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def parse_answers(text: str) -> list[str]:
    matches = re.findall(r"ans:\s*(.+?)(?:\n|$)", text)
    return [normalize(m.strip()) for m in matches if m.strip()]


def main():
    if not os.path.exists(PREDICTIONS_FILE):
        raise FileNotFoundError(
            f"Expected predictions file not found: {PREDICTIONS_FILE}\n"
            "Run download.py first."
        )

    print(f"Loading records from JSONL...")
    records = []
    with open(PREDICTIONS_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  Loaded {len(records)} records")

    random.seed(42)
    sample = random.sample(records, SAMPLE_SIZE)
    sys_prompt = records[0]["sys_query"]

    print(f"  System prompt: {sys_prompt[:80]}...")
    print(f"  Sampled {SAMPLE_SIZE} questions (seed=42)\n")

    client = OpenAI()
    results = []
    errors = 0

    for rec in tqdm(sample, desc="GPT-4o-mini inference"):
        qid = rec["id"]
        question = rec["question"]
        gold_raw = rec["a_entity"]
        gold = [normalize(a) for a in gold_raw]
        user_query = rec["user_query"]

        llm_response = ""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": ICL_USER},
                    {"role": "assistant", "content": ICL_ASSISTANT},
                    {"role": "user", "content": user_query},
                ],
                temperature=0,
                seed=42,
                max_tokens=512,
            )
            llm_response = response.choices[0].message.content or ""
        except Exception as e:
            print(f"\n  ERROR on {qid}: {e}")
            errors += 1

        predicted = parse_answers(llm_response)
        pred_set = set(predicted)
        gold_set = set(gold)
        f1 = compute_f1(pred_set, gold_set)
        hit = 1 if pred_set & gold_set else 0

        results.append({
            "id": qid,
            "question": question,
            "gold": gold,
            "predicted": predicted,
            "f1": f1,
            "hit": hit,
            "llm_response": llm_response,
        })

    macro_f1 = sum(r["f1"] for r in results) / len(results) * 100
    hit_rate = sum(r["hit"] for r in results) / len(results) * 100

    output = {
        "config": {
            "model": "gpt-4o-mini",
            "top_k": TOP_K,
            "sample_size": SAMPLE_SIZE,
            "seed": 42,
            "dataset": "WebQSP test",
            "api_errors": errors,
        },
        "summary": {"macro_f1": round(macro_f1, 4), "hit": round(hit_rate, 4)},
        "per_question": results,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Macro-F1 : {macro_f1:.2f}%  (paper: 77.45%)")
    print(f"  Hit      : {hit_rate:.2f}%  (paper: 90.11%)")
    print(f"  Sample   : {len(results)} questions  |  errors: {errors}")
    print(f"  Saved    : {RESULTS_FILE}")

    # Print 2 correct + 1 incorrect examples for slides
    correct = sorted(
        [r for r in results if r["hit"] == 1 and r["f1"] >= 0.9],
        key=lambda r: -r["f1"],
    )
    incorrect = [r for r in results if r["hit"] == 0 and r["predicted"]]

    print(f"\n{'='*60}")
    print("EXAMPLE OUTPUTS  (copy to slides)")
    print(f"{'='*60}")

    for i, r in enumerate(correct[:2], 1):
        print(f"\n[CORRECT {i}]  {r['id']}")
        print(f"  Question : {r['question']}")
        print(f"  Gold     : {r['gold']}")
        print(f"  Predicted: {r['predicted']}")
        print(f"  F1={r['f1']:.2f}  Hit={r['hit']}")

    if incorrect:
        r = incorrect[0]
        print(f"\n[INCORRECT]  {r['id']}")
        print(f"  Question : {r['question']}")
        print(f"  Gold     : {r['gold']}")
        print(f"  Predicted: {r['predicted']}")
        print(f"  F1={r['f1']:.2f}  Hit={r['hit']}")


if __name__ == "__main__":
    main()
