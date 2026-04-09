"""
LexGuard — Unified End-to-End Pipeline Evaluation
===================================================
Tests the COMPLETE Streamlit pipeline against CUAD ground truth:
  1. BERT Risk Clause Extraction (12 clauses) — does BERT detect the clause correctly?
  2. V4 Contract Brief (8 metadata entities) — does the full-doc LLM extract correctly?
  3. Hybrid Search Retrieval — can the retriever find the relevant chunks?

This gives ONE unified accuracy report for the entire system.
"""

import os
import sys
import glob
import pandas as pd
import ast
import json
import time
from dotenv import load_dotenv
from google import genai

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from local_store import LocalStore
from tools import extract_clause_with_bert, CUAD_PROMPTS, extract_contract_brief

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ═══════════════════════════════════════════════
# BERT Clause Type → CSV Column mapping
# ═══════════════════════════════════════════════
BERT_CLAUSE_MAP = {
    "Non-Compete":                  {"yes_no": "Non-Compete-Answer",                 "text": "Non-Compete"},
    "Governing Law":                {"yes_no": "Governing Law-Answer",               "text": "Governing Law"},
    "Audit Rights":                 {"yes_no": "Audit Rights-Answer",                "text": "Audit Rights"},
    "Change of Control":            {"yes_no": "Change Of Control-Answer",           "text": "Change Of Control"},
    "Effective Date":               {"yes_no": "Effective Date-Answer",              "text": "Effective Date"},
    "Uncapped Liability":           {"yes_no": "Uncapped Liability-Answer",          "text": "Uncapped Liability"},
    "Cap on Liability":             {"yes_no": "Cap On Liability-Answer",            "text": "Cap On Liability"},
    "Exclusivity":                  {"yes_no": "Exclusivity-Answer",                 "text": "Exclusivity"},
    "Liquidated Damages":           {"yes_no": "Liquidated Damages-Answer",          "text": "Liquidated Damages"},
    "Termination for Convenience":  {"yes_no": "Termination For Convenience-Answer", "text": "Termination For Convenience"},
    "IP Ownership Assignment":      {"yes_no": "Ip Ownership Assignment-Answer",     "text": "Ip Ownership Assignment"},
    "Source Code Escrow":           {"yes_no": "Source Code Escrow-Answer",           "text": "Source Code Escrow"},
}

# V4 Entity mapping (reused from evaluate_extraction.py)
V4_ENTITIES = [
    {"name": "Document Name",               "col": "Document Name-Answer"},
    {"name": "Parties",                      "col": "Parties-Answer"},
    {"name": "Agreement Date",              "col": "Agreement Date-Answer"},
    {"name": "Effective Date",              "col": "Effective Date-Answer"},
    {"name": "Expiration Date",             "col": "Expiration Date-Answer"},
    {"name": "Renewal Term",                "col": "Renewal Term-Answer"},
    {"name": "Notice to Terminate Renewal", "col": "Notice Period To Terminate Renewal- Answer"},
    {"name": "Governing Law",               "col": "Governing Law-Answer"},
]


def llm_grade(predicted: str, ground_truth: str) -> bool:
    """Semantic LLM grading — returns True if predicted matches ground truth."""
    if not predicted or predicted.upper() == "NOT FOUND":
        return False
    prompt = f"""You are a lenient but accurate grader for contract metadata extraction.
Does the PREDICTED VALUE contain the same core factual information as the GROUND TRUTH VALUE?

Grading rules:
- Date formats may differ (5/8/2014 vs May 8, 2014). All equivalent.
- Party names may omit suffixes or aliases. Still a match.
- Extra detail in prediction is fine.
- "State of Nevada" matches "Nevada". "Ontario" matches "Ontario, Canada".
- "one (1) year" matches "successive 1 year".
- "Perpetual" and "until terminated" are equivalent.
- For renewal terms with dates vs durations, mark YES if logically consistent.

PREDICTED VALUE: {predicted}
GROUND TRUTH VALUE: {ground_truth}

Answer strictly "YES" or "NO"."""
    try:
        time.sleep(0.5)
        resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return "YES" in resp.text.strip().upper()
    except Exception as e:
        print(f"  Grading Error: {e}")
        return False


def get_ground_truth_text(row, col):
    """Extract the first valid ground truth string from a CSV cell."""
    val = row.get(col, None)
    if pd.isna(val) or str(val).strip() in ['[]', '', 'nan']:
        return None
    try:
        if str(val).startswith('['):
            arr = ast.literal_eval(str(val))
            valid = [str(a).strip() for a in arr if len(str(a).strip()) > 0]
            return valid[0] if valid else None
        return str(val).strip()
    except:
        return str(val).strip()


def main():
    print("=" * 70)
    print("  LexGuard — Unified End-to-End Pipeline Evaluation")
    print("=" * 70)

    df = pd.read_csv("./data_test/master_clauses.csv")

    # Match .txt files to ground truth
    txt_files = {os.path.basename(f): f for f in glob.glob("./data_test/*.txt")}
    pdf_to_txt = {}
    for txt_name, txt_path in txt_files.items():
        pdf_name = txt_name.replace(".txt", ".pdf")
        if pdf_name in df['Filename'].values:
            pdf_to_txt[pdf_name] = txt_path

    print(f"\n📄 Found {len(pdf_to_txt)} contracts with ground truth.\n")

    # ═══════════════════════════════════════════════
    # Results accumulators
    # ═══════════════════════════════════════════════
    bert_results = {ct: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for ct in BERT_CLAUSE_MAP}
    v4_results = {e["name"]: {"hits": 0, "total": 0} for e in V4_ENTITIES}
    hybrid_results = {"found": 0, "total": 0}

    # Init hybrid search store
    store = LocalStore(working_dir="./evaluate_store_all_MiniLM_L6_v2", dense_model="all-MiniLM-L6-v2")

    for doc_idx, (pdf_name, txt_path) in enumerate(sorted(pdf_to_txt.items()), 1):
        print(f"\n{'='*60}")
        print(f"[{doc_idx}/{len(pdf_to_txt)}] {pdf_name}")
        print(f"{'='*60}")

        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()

        row = df[df['Filename'] == pdf_name].iloc[0]

        # ═══════════════════════════════════════════════
        # PHASE 1: BERT Risk Clause Detection
        # ═══════════════════════════════════════════════
        print("\n  --- PHASE 1: BERT Risk Clause Detection ---")
        for clause_type, cols in BERT_CLAUSE_MAP.items():
            gt_yes_no = str(row.get(cols["yes_no"], "No")).strip()
            ground_truth_present = gt_yes_no.lower() == "yes"

            bert_output = extract_clause_with_bert(clause_type, full_text)
            bert_detected = "No underlying risks detected" not in bert_output and "Error" not in bert_output

            if ground_truth_present and bert_detected:
                bert_results[clause_type]["tp"] += 1
                status = "✅ TP"
            elif not ground_truth_present and not bert_detected:
                bert_results[clause_type]["tn"] += 1
                status = "✅ TN"
            elif not ground_truth_present and bert_detected:
                bert_results[clause_type]["fp"] += 1
                status = "⚠️ FP"
            else:
                bert_results[clause_type]["fn"] += 1
                status = "❌ FN"

            print(f"    {status} {clause_type} | GT={gt_yes_no} | BERT={'Detected' if bert_detected else 'Not Found'}")

        # ═══════════════════════════════════════════════
        # PHASE 2: V4 Contract Brief (Full-Doc LLM)
        # ═══════════════════════════════════════════════
        print("\n  --- PHASE 2: V4 Contract Brief ---")
        brief = extract_contract_brief(full_text)

        if brief:
            for entity in V4_ENTITIES:
                gt = get_ground_truth_text(row, entity["col"])
                if not gt:
                    continue

                pred = str(brief.get(entity["name"], "NOT FOUND")).strip()
                is_match = llm_grade(pred, gt)

                v4_results[entity["name"]]["total"] += 1
                if is_match:
                    v4_results[entity["name"]]["hits"] += 1
                    print(f"    ✅ {entity['name']} | Pred: {pred[:60]} | Truth: {gt[:60]}")
                else:
                    print(f"    ❌ {entity['name']} | Pred: {pred[:60]} | Truth: {gt[:60]}")

        # ═══════════════════════════════════════════════
        # PHASE 3: Hybrid Search Retrieval
        # ═══════════════════════════════════════════════
        print("\n  --- PHASE 3: Hybrid Search Retrieval ---")
        # Test retrieval for clauses that exist in this doc
        for clause_type, cols in BERT_CLAUSE_MAP.items():
            gt_text = get_ground_truth_text(row, cols["text"])
            if not gt_text:
                continue

            results = store.search_hybrid(clause_type, top_k=5)
            doc_results = [r for r in results if r['doc_name'] == pdf_name]

            hybrid_results["total"] += 1
            if doc_results:
                # Check if any retrieved chunk semantically contains the ground truth
                combined = " ".join([r['text'][:500] for r in doc_results])
                is_found = llm_grade(combined[:2000], gt_text[:500])
                if is_found:
                    hybrid_results["found"] += 1
                    print(f"    ✅ Hybrid found '{clause_type}' ({len(doc_results)} chunks)")
                else:
                    print(f"    ❌ Hybrid missed '{clause_type}' (chunks found but no match)")
            else:
                print(f"    ❌ Hybrid missed '{clause_type}' (0 chunks for this doc)")

    # ═══════════════════════════════════════════════
    # FINAL UNIFIED REPORT
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  UNIFIED END-TO-END PIPELINE EVALUATION RESULTS")
    print("=" * 70)

    # BERT Summary
    print("\n📊 PHASE 1: BERT Risk Clause Detection (12 clauses)")
    bert_summary = []
    for ct, stats in bert_results.items():
        total = stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"]
        if total > 0:
            accuracy = ((stats["tp"] + stats["tn"]) / total) * 100
            precision = (stats["tp"] / (stats["tp"] + stats["fp"]) * 100) if (stats["tp"] + stats["fp"]) > 0 else 0
            recall = (stats["tp"] / (stats["tp"] + stats["fn"]) * 100) if (stats["tp"] + stats["fn"]) > 0 else 0
            bert_summary.append({
                "Clause": ct,
                "Accuracy": f"{accuracy:.0f}%",
                "Precision": f"{precision:.0f}%",
                "Recall": f"{recall:.0f}%",
                "TP": stats["tp"], "TN": stats["tn"],
                "FP": stats["fp"], "FN": stats["fn"]
            })
    print(pd.DataFrame(bert_summary).to_markdown(index=False))

    # Overall BERT
    total_bert = sum(s["tp"]+s["tn"]+s["fp"]+s["fn"] for s in bert_results.values())
    correct_bert = sum(s["tp"]+s["tn"] for s in bert_results.values())
    bert_acc = (correct_bert / total_bert * 100) if total_bert > 0 else 0
    print(f"\n🎯 Overall BERT Accuracy: {bert_acc:.1f}% ({correct_bert}/{total_bert})")

    # V4 Summary
    print("\n📊 PHASE 2: V4 Contract Brief (8 entities)")
    v4_summary = []
    for e in V4_ENTITIES:
        s = v4_results[e["name"]]
        if s["total"] > 0:
            acc = s["hits"] / s["total"] * 100
            v4_summary.append({"Entity": e["name"], "Accuracy": f"{acc:.1f}%", "Detail": f"{s['hits']}/{s['total']}"})
    print(pd.DataFrame(v4_summary).to_markdown(index=False))

    # Hybrid Summary
    print(f"\n📊 PHASE 3: Hybrid Search Retrieval")
    hybrid_acc = (hybrid_results["found"] / hybrid_results["total"] * 100) if hybrid_results["total"] > 0 else 0
    print(f"🎯 Hybrid Retrieval Recall: {hybrid_acc:.1f}% ({hybrid_results['found']}/{hybrid_results['total']})")

    # GRAND TOTAL
    print("\n" + "=" * 70)
    total_all = total_bert + sum(s["total"] for s in v4_results.values()) + hybrid_results["total"]
    correct_all = correct_bert + sum(s["hits"] for s in v4_results.values()) + hybrid_results["found"]
    grand_acc = (correct_all / total_all * 100) if total_all > 0 else 0
    print(f"🏆 GRAND TOTAL PIPELINE ACCURACY: {grand_acc:.1f}% ({correct_all}/{total_all})")
    print("=" * 70)


if __name__ == "__main__":
    main()
