import os
import glob
import pandas as pd
import ast
import uuid
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from local_store import LocalStore

def get_chunks(text, doc_name, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_str = text[start:end]
        chunks.append({
            "CHUNK_ID": str(uuid.uuid4()),
            "DOC_NAME": doc_name.replace(".txt", ".pdf"), # Match the csv
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

def evaluate_experiment(store, df, all_chunks, top_k, use_expanded):
    clauses_to_test = [
        ("Governing Law", "Governing Law-Answer"),
        ("Non-Compete", "Non-Compete-Answer"),
        ("Change Of Control", "Change Of Control-Answer")
    ]
    
    total_queries = 0
    topk_hits = 0
    
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
                
            query = EXPANDED_QUERIES[cls_name] if use_expanded else cls_name
            # If BGE model is used natively without instruct prompt, it's ok, but BGE-small works best with "Represent this sentence for searching relevant passages: "
            # However, for hybrid, we keep it simple here.
            
            all_results = store.search_hybrid(query, top_k=333)
            doc_results = [r for r in all_results if r['doc_name'] == doc_name][:top_k]
            
            found = False
            for ans in ans_list:
                ans_clean = str(ans).lower().strip()
                if len(ans_clean) < 10:
                    continue
                
                ans_parts = [p.strip() for p in ans_clean.split('<omitted>') if len(p.strip()) > 5]
                if not ans_parts:
                    ans_parts = [ans_clean]

                for res in doc_results:
                    chunk_text = res['text'].lower()
                    chunk_words = set(chunk_text.split())
                    ans_words = set(ans_clean.replace('<omitted>', '').split())
                    
                    overlap_ratio = len(chunk_words.intersection(ans_words)) / max(1, len(ans_words))
                    part_found = any(part in chunk_text for part in ans_parts)
                    
                    if part_found or overlap_ratio > 0.4:
                        found = True
                        break
                if found: break
                
            total_queries += 1
            if found:
                topk_hits += 1
                
    return (topk_hits / max(1, total_queries)) * 100

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
    df = pd.read_csv("./data_test/master_clauses.csv")
    
    experiments = [
        {"name": "Baseline", "model": "all-MiniLM-L6-v2", "top_k": 3, "use_expanded": False},
        {"name": "Increased Top-K", "model": "all-MiniLM-L6-v2", "top_k": 10, "use_expanded": False},
        {"name": "Query Expansion", "model": "all-MiniLM-L6-v2", "top_k": 3, "use_expanded": True},
        {"name": "Advanced Model", "model": "BAAI/bge-small-en-v1.5", "top_k": 3, "use_expanded": False},
        {"name": "Ultimate Combo", "model": "BAAI/bge-small-en-v1.5", "top_k": 10, "use_expanded": True},
    ]
    
    # Pre-build stores for unique models so we don't re-embed
    stores = {}
    unique_models = set(exp["model"] for exp in experiments)
    for model in unique_models:
        safe_model_dir = model.replace("/", "_").replace("-", "_")
        store_path = f"./evaluate_store_{safe_model_dir}"
        print(f"\nInitializing LocalStore for model '{model}' at '{store_path}'...")
        store = LocalStore(working_dir=store_path, dense_model=model)
        store.ingest(all_chunks)
        stores[model] = store
        
    results_table = []
    print("\n--- RUNNING EXPERIMENTS ---")
    for exp in experiments:
        print(f"Running: {exp['name']}...")
        store = stores[exp["model"]]
        recall = evaluate_experiment(store, df, all_chunks, exp["top_k"], exp["use_expanded"])
        results_table.append({
            "Experiment": exp["name"],
            "Model": exp["model"],
            "Top-K": exp["top_k"],
            "Expanded Query": "Yes" if exp["use_expanded"] else "No",
            "Recall": f"{recall:.1f}%"
        })
        
    print("\n--- FINAL RESULTS ---")
    print(pd.DataFrame(results_table).to_markdown(index=False))

if __name__ == "__main__":
    main()
