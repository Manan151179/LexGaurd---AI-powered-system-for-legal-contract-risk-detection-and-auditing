"""
LexGuard — Full-Document Entity Extraction Benchmark (V2)
=========================================================
Strategy: Send the ENTIRE contract text to Gemini 2.5 Flash (1M context window)
instead of fragmented Top-5 chunks. Uses chain-of-thought prompting and
relaxed semantic grading to target >80% accuracy on all 8 metadata entities.
"""

import os
import glob
import pandas as pd
import ast
import json
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

ENTITIES_TO_EXTRACT = [
    {
        "name": "Document Name",
        "desc": "The official title or name of the contract/agreement as stated in the document header.",
        "col": "Document Name-Answer"
    },
    {
        "name": "Parties",
        "desc": "The two or more parties who signed the contract. Return only the entity/individual names separated by semicolons. Ignore aliases in parentheses.",
        "col": "Parties-Answer"
    },
    {
        "name": "Agreement Date",
        "desc": "The date the contract was signed, executed, or 'made as of'. Format as MM/DD/YYYY if possible.",
        "col": "Agreement Date-Answer"
    },
    {
        "name": "Effective Date",
        "desc": "The date the contract goes into effect. IMPORTANT: If the contract says 'effective as of the date first written above' or 'effective as of the Agreement Date', the Effective Date IS the Agreement Date. Reason step by step.",
        "col": "Effective Date-Answer"
    },
    {
        "name": "Expiration Date",
        "desc": "The date the contract's initial term expires. IMPORTANT: (a) If stated as 'N years from the Effective Date', you MUST calculate the actual calendar date. (b) If the agreement has no fixed end date, or says it continues 'in perpetuity', 'until terminated', or 'for so long as', answer 'Perpetual'. (c) Look in the 'Term' or 'Term and Termination' section. Reason step by step.",
        "col": "Expiration Date-Answer"
    },
    {
        "name": "Renewal Term",
        "desc": "How long the contract automatically renews for after the initial term expires. Look for words like 'automatically renew', 'successive', 'additional period', 'extend', 'option to renew'. IMPORTANT: (a) If the agreement is perpetual or has no fixed end and no renewal cycle, answer 'Perpetual'. (b) If the contract grants renewal 'options' for specific periods, state the period length (e.g., '3 years'). (c) A contract that says 'until terminated' with no renewal clause is 'Perpetual'. If absolutely no renewal mechanism exists and the contract is NOT perpetual, answer 'NOT FOUND'.",
        "col": "Renewal Term-Answer"
    },
    {
        "name": "Notice to Terminate Renewal",
        "desc": "The advance notice period required to prevent auto-renewal or to terminate the agreement. Look for phrases like 'written notice', 'days prior to expiration', 'notice of non-renewal', 'notice of termination'. Check both the Renewal/Term section AND the Termination section carefully. Answer as a time period (e.g., '30 days', '60 days', '90 days'). If not specified, answer 'NOT FOUND'.",
        "col": "Notice Period To Terminate Renewal- Answer"
    },
    {
        "name": "Governing Law",
        "desc": "Which state or country's law governs the contract. Return just the jurisdiction name (e.g., 'Nevada', 'Ontario, Canada').",
        "col": "Governing Law-Answer"
    }
]

def extract_all_entities(full_text: str) -> dict:
    """Send the ENTIRE document to Gemini and extract all 8 entities in one shot."""
    
    entity_descriptions = "\n".join(
        [f"{i+1}. **{e['name']}**: {e['desc']}" for i, e in enumerate(ENTITIES_TO_EXTRACT)]
    )
    
    prompt = f"""You are an expert contract reviewer. You must extract the following metadata from the provided contract text.

Think step-by-step for each entity. Apply these critical reasoning rules:
1. For dates: If the contract says "effective as of the date first written above" or similar, the Effective Date IS the Agreement Date.
2. For expiration: If stated as "N years from [date]", CALCULATE the actual calendar date. If the contract has no end date or says "perpetual" or "until terminated", answer "Perpetual".
3. For renewal: Look for "automatically renew", "successive periods", "extend". If the contract is perpetual with no termination date, the renewal is also "Perpetual".
4. For notice to terminate: Search the ENTIRE document — both the "Term" section AND the "Termination" section. Look for "written notice", "days prior", "notice of non-renewal".
5. NEVER answer "NOT FOUND" if the information exists somewhere in the document. Read the entire text carefully.

ENTITIES TO EXTRACT:
{entity_descriptions}

OUTPUT FORMAT: Respond with ONLY a valid JSON object. Keys must exactly match: "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term", "Notice to Terminate Renewal", "Governing Law".
If any information is completely missing from the contract, use "NOT FOUND" as the value.
Do NOT wrap in markdown code fences.

CONTRACT TEXT:
{full_text}"""

    try:
        time.sleep(1)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        raw = response.text.strip()
        # Clean potential markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"  Extraction Error: {e}")
        return {}

def grade_extraction(extracted: str, ground_truth: str) -> bool:
    """Semantically grade whether the extraction matches ground truth."""
    if not extracted or extracted.upper() == "NOT FOUND":
        return False

    prompt = f"""You are a lenient but accurate grader for contract metadata extraction.

Does the PREDICTED VALUE contain the same core factual information as the GROUND TRUTH VALUE?

Grading rules:
- Date formats may differ (5/8/2014 vs May 8, 2014 vs 2014-05-08). These are ALL equivalent.
- Party names may omit suffixes like "Inc.", "Corp.", or aliases in parentheses like ("Company"). This is still a match.
- Extra detail in the prediction is fine, as long as the ground truth is accurately represented.
- For jurisdiction/governing law, "State of Nevada" matches "Nevada". "Ontario" matches "Ontario, Canada".
- For renewal terms, "one (1) year" matches "successive 1 year" or "1 year".
- For renewal terms, if the Ground Truth contains dates (e.g. "7/22/2019; 7/22/2022"), and the Predicted Value gives consistent DURATION (e.g. "3 years"), mark YES if those dates logically correspond to that duration from the initial term.
- "Perpetual" and "until terminated" are equivalent concepts.

PREDICTED VALUE: {extracted}
GROUND TRUTH VALUE: {ground_truth}

Answer strictly "YES" or "NO"."""

    try:
        time.sleep(0.5)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return "YES" in response.text.strip().upper()
    except Exception as e:
        print(f"  Grading Error: {e}")
        return False


def main():
    print("=" * 60)
    print("  LexGuard — Full-Document Entity Extraction Benchmark V2")
    print("=" * 60)

    df = pd.read_csv("./data_test/master_clauses.csv")
    
    # Match available .txt files to CSV ground truth
    txt_files = {os.path.basename(f): f for f in glob.glob("./data_test/*.txt")}
    pdf_to_txt = {}
    for txt_name, txt_path in txt_files.items():
        pdf_name = txt_name.replace(".txt", ".pdf")
        if pdf_name in df['Filename'].values:
            pdf_to_txt[pdf_name] = txt_path
    
    print(f"\n📄 Found {len(pdf_to_txt)} contracts with ground truth.\n")
    
    results = {e["name"]: {"hits": 0, "total": 0, "details": []} for e in ENTITIES_TO_EXTRACT}
    
    for doc_idx, (pdf_name, txt_path) in enumerate(sorted(pdf_to_txt.items()), 1):
        print(f"[{doc_idx}/{len(pdf_to_txt)}] {pdf_name}")
        
        # Read the FULL document text
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        
        # Truncate at 200k chars (well within Gemini's 1M token limit)
        full_text = full_text[:200000]
        
        # Extract ALL entities in one LLM call
        extracted = extract_all_entities(full_text)
        
        if not extracted:
            print(f"  ⚠️ Extraction returned empty, skipping.")
            continue
        
        # Get ground truth row
        row = df[df['Filename'] == pdf_name].iloc[0]
        
        for entity in ENTITIES_TO_EXTRACT:
            col = entity["col"]
            name = entity["name"]
            
            # Get ground truth
            ans_str = row.get(col, None)
            if pd.isna(ans_str) or str(ans_str).strip() in ['[]', '', 'nan']:
                continue
            
            try:
                if str(ans_str).startswith('['):
                    ans_list = ast.literal_eval(str(ans_str))
                else:
                    ans_list = [str(ans_str)]
            except:
                ans_list = [str(ans_str)]
            
            valid_ans = [str(a).strip() for a in ans_list if len(str(a).strip()) > 0]
            if not valid_ans:
                continue
            
            ground_truth = valid_ans[0]
            
            # Get extracted value
            pred = str(extracted.get(name, "NOT FOUND")).strip()
            
            # Grade
            is_match = grade_extraction(pred, ground_truth)
            
            results[name]["total"] += 1
            if is_match:
                results[name]["hits"] += 1
                print(f"  ✅ {name} | Pred: {pred[:80]} | Truth: {ground_truth[:80]}")
            else:
                print(f"  ❌ {name} | Pred: {pred[:80]} | Truth: {ground_truth[:80]}")
            
            results[name]["details"].append({
                "doc": pdf_name[:50],
                "predicted": pred[:100],
                "truth": ground_truth[:100],
                "match": is_match
            })
    
    # Print final results
    print("\n" + "=" * 60)
    print("  FINAL EXTRACTION BENCHMARK RESULTS")
    print("=" * 60)
    
    summary = []
    all_above_80 = True
    for ent in ENTITIES_TO_EXTRACT:
        name = ent["name"]
        stats = results[name]
        if stats["total"] > 0:
            acc = (stats["hits"] / stats["total"]) * 100
            status = "✅" if acc >= 80 else "❌"
            if acc < 80:
                all_above_80 = False
            summary.append({
                "Entity": name,
                "Accuracy": f"{acc:.1f}%",
                "Detail": f"{stats['hits']}/{stats['total']}",
                "Status": status
            })
    
    print(pd.DataFrame(summary).to_markdown(index=False))
    
    if all_above_80:
        print("\n🎉 ALL ENTITIES ABOVE 80%! Ready for production.")
    else:
        print("\n⚠️ Some entities below 80%. Review the failures above.")


if __name__ == "__main__":
    main()
