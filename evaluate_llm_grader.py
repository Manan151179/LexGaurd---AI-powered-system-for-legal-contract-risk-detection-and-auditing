import os
import glob
import pandas as pd
import ast
import uuid
import sys
import time
from datetime import datetime, timezone
from google import genai
from google.genai import types
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_store import LocalStore

load_dotenv()

# Initialize Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env, grading will fail.")

client = genai.Client(api_key=GEMINI_API_KEY)

def ask_llm(ground_truth, chunk_text):
    prompt = f"""You are an expert legal annotator.
You must determine if a given CONTRACT CHUNK contains the equivalent legal provision as the REQUIRED GROUND TRUTH snippet.

REQUIRED GROUND TRUTH:
"{ground_truth}"

CONTRACT CHUNK:
"{chunk_text}"

Answer strictly "YES" if the chunk covers the same specific legal provision or contains the same factual restrictions/terms as the ground truth. It does not need to be a word-for-word match if OCR/formatting is messy.
Answer strictly "NO" if it does not contain the target information.
Do not provide any explanation. Output only YES or NO."""

    try:
        time.sleep(1) # Rate limit protection
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        ans = response.text.strip().upper()
        return "YES" in ans
    except Exception as e:
        print(f"LLM Error: {e}")
        return False

def get_chunks(text, doc_name, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end]
        chunks.append({
            "CHUNK_ID": str(uuid.uuid4()),
            "DOC_NAME": doc_name.replace(".txt", ".pdf"),
            "CHUNK_TEXT": chunk_str.strip(),
            "METADATA": "{}",
            "UPLOAD_TIMESTAMP": datetime.now(timezone.utc).isoformat()
        })
        start += chunk_size - overlap
    return chunks

EXPANDED_QUERIES = {
    "Governing Law": "Governing Law, state jurisdiction, applicable laws, court location, governed by the laws of",
    "Non-Compete": "Non-Compete, Restrictive Covenants, competing business, covenant not to compete",
    "Change Of Control": "Change Of Control, merger, acquisition, voting securities, transfer of assets"
}

def main():
    print("Loading test data...")
    txt_files = glob.glob("./data_test/*.txt")
    if not txt_files:
        print("No txt files found.")
        return
        
    all_chunks = []
    for txt in txt_files:
        with open(txt, "r", encoding="utf-8") as f:
            text = f.read()
        doc_name = os.path.basename(txt)
        all_chunks.extend(get_chunks(text, doc_name))

    print(f"Total chunks generated: {len(all_chunks)}")
    
    # We use MiniLM Baseline + Query Expansion + Top-3 as the experiment
    store_path = "./evaluate_store_all_MiniLM_L6_v2"
    store = LocalStore(working_dir=store_path, dense_model="all-MiniLM-L6-v2")
    # Ingestion already happened in previous script, but we ensure it's loaded
    if len(store.chunks) == 0:
        store.ingest(all_chunks)

    df = pd.read_csv("./data_test/master_clauses.csv")
    
    clauses_to_test = [
        ("Governing Law", "Governing Law-Answer"),
        ("Non-Compete", "Non-Compete-Answer"),
        ("Change Of Control", "Change Of Control-Answer")
    ]
    
    total_queries = 0
    topk_hits = 0
    
    print("\n--- STARTING LLM GRADING ---")
    
    for _, row in df.iterrows():
        filename = row['Filename']
        doc_name = filename.strip()
        
        if not any(c['DOC_NAME'] == doc_name for c in all_chunks):
            continue

        for cls_name, cls_ans_col in clauses_to_test:
            ans_str = row[cls_ans_col]
            if pd.isna(ans_str) or ans_str == '[]':
                continue
                
            try:
                if ans_str.startswith('['):
                    ans_list = ast.literal_eval(ans_str)
                else:
                    ans_list = [ans_str]
            except:
                ans_list = [ans_str]
                
            query = EXPANDED_QUERIES[cls_name]
            all_results = store.search_hybrid(query, top_k=333)
            doc_results = [r for r in all_results if r['doc_name'] == doc_name][:3] # Top-3 for the doc
            
            # Since answers can be lists, we check if the LLM validates *any* required answer part over *any* of the 3 chunks
            # To save tokens, we just take the first substantial answer
            valid_ans = [str(a) for a in ans_list if len(str(a).strip()) > 10]
            if not valid_ans:
                continue
            
            ground_truth = valid_ans[0]
                
            found = False
            for i, res in enumerate(doc_results):
                chunk_text = res['text']
                is_match = ask_llm(ground_truth, chunk_text)
                if is_match:
                    found = True
                    break
                    
            total_queries += 1
            if found:
                topk_hits += 1
                print(f"[{total_queries}] ✅ MATCH FOUND: {doc_name} -> {cls_name}")
            else:
                print(f"[{total_queries}] ❌ MISSED: {doc_name} -> {cls_name}")
                
    recall = (topk_hits / max(1, total_queries)) * 100
    print("\n--- LLM EVALUATION RESULTS ---")
    print(f"Total Valid Queries: {total_queries}")
    print(f"LLM-Graded Top-3 Recall: {topk_hits}/{total_queries} ({recall:.1f}%)")

if __name__ == "__main__":
    main()
