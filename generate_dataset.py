import pandas as pd
import json
import ast
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def parse_clause(clause_str):
    """
    CUAD text clauses are stored as string representations of lists, 
    e.g., "['clause text here']". We need to safely evaluate them and extract the text.
    """
    if pd.isna(clause_str) or not isinstance(clause_str, str):
        return None
    try:
        parsed = ast.literal_eval(clause_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            return " ".join(parsed)  # Combine if multiple strings
    except (ValueError, SyntaxError):
        pass
    
    # Fallback if evaluation fails but it looks like a list
    if clause_str.startswith("['") and clause_str.endswith("']"):
        return clause_str[2:-2]
        
    return str(clause_str)

def generate_dataset(csv_path="master_clauses.csv", output_path="instruction_dataset.json"):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    dataset = []
    
    high_cols = {
        "Uncapped Liability": "uncapped liability",
        "Non-Compete": "strict non-compete terms",
        "Exclusivity": "exclusive dealing requirements",
        "Most Favored Nation": "most favored nation pricing"
    }
    
    medium_cols = {
        "Liquidated Damages": "liquidated damages or penalties",
        "Audit Rights": "strict audit requirements",
        "Change Of Control": "change of control conditions",
        "Minimum Commitment": "minimum order or revenue commitments"
    }
    
    low_cols = {
        "Governing Law": "standard governing law provision",
    }
    
    instruction = "Analyze the legal clause to determine if it contains high-risk language. Classify the risk as 'High Risk', 'Medium Risk', or 'Low Risk' and provide the reasoning."
    
    counts = {"High Risk": 0, "Medium Risk": 0, "Low Risk": 0}
    target_per_class = 17 # Aim for ~50 total balanced examples
    
    def generate_gemini_explanation(clause_text, risk_level, reason):
        prompt = f"""You are an expert legal auditor. I am creating a fine-tuning dataset to teach a small language model how to reason like you.

I have a contract clause that falls under the category of '{risk_level}' because it contains '{reason}'.

Clause Text: 
"{clause_text}"

Write a detailed, step-by-step instructional explanation explaining why this clause is {risk_level}. 
You must quote specific parts of the text to support your reasoning.
Do not use markdown formatting. Keep the tone analytical and professional.
Format your final output starting with: '{risk_level}: '"""
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text.strip().replace("**", "").replace("_", "")
        except Exception as e:
            print(f"Error generating reasoning: {e}")
            return f"{risk_level}: This clause introduces {reason}. (Fallback: API Error)"

    print("Generating verbose reasoning targets via Gemini API. This may take a minute...")
    for idx, row in df.iterrows():
        # High Risk
        for col, reason in high_cols.items():
            ans_col = f"{col}-Answer"
            if counts["High Risk"] < target_per_class and ans_col in df.columns and str(row[ans_col]).strip() == "Yes":
                text = parse_clause(row[col])
                if text:
                    gemini_reasoning = generate_gemini_explanation(text, "High Risk", reason)
                    dataset.append({
                        "instruction": instruction,
                        "input": text,
                        "output": gemini_reasoning
                    })
                    counts["High Risk"] += 1
                    
        # Medium Risk
        for col, reason in medium_cols.items():
            ans_col = f"{col}-Answer"
            if counts["Medium Risk"] < target_per_class and ans_col in df.columns and str(row[ans_col]).strip() == "Yes":
                text = parse_clause(row[col])
                if text:
                    gemini_reasoning = generate_gemini_explanation(text, "Medium Risk", reason)
                    dataset.append({
                        "instruction": instruction,
                        "input": text,
                        "output": gemini_reasoning
                    })
                    counts["Medium Risk"] += 1
                    
        # Low Risk
        for col, reason in low_cols.items():
            ans_col = f"{col}-Answer"
            if counts["Low Risk"] < target_per_class and ans_col in df.columns:
                ans_str = str(row[ans_col]).strip()
                if ans_str != "nan" and ans_str != "[]" and ans_str != "":
                    text = parse_clause(row[col])
                    if text:
                        gemini_reasoning = generate_gemini_explanation(text, "Low Risk", reason)
                        dataset.append({
                            "instruction": instruction,
                            "input": text,
                            "output": gemini_reasoning
                        })
                        counts["Low Risk"] += 1
                        
        if len(dataset) >= 50:
            break

    print(f"Dataset generated with {len(dataset)} examples. (High: {counts['High Risk']}, Medium: {counts['Medium Risk']}, Low: {counts['Low Risk']})")
    
    with open(output_path, "w") as f:
        json.dump(dataset[:50], f, indent=4)
        
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_dataset()
