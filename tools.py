import os
import streamlit as st
import snowflake.connector

import config  # noqa: F401 — seeds randomness on import
from local_store import LocalStore
from transformers import pipeline

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Global variable to cache the pipeline
_bert_extractor = None

def get_bert_extractor():
    global _bert_extractor
    if _bert_extractor is None:
        print("🧠 Loading LexGuard-CUAD-BERT model locally...")
        _bert_extractor = pipeline(
            "question-answering",
            model="doandune/LexGuard-CUAD-BERT",
            handle_impossible_answer=True
        )
    return _bert_extractor

def expand_fragment_to_sentence(chunk: str, fragment_start: int, fragment_end: int) -> str:
    """Takes the start/end character indices from BERT and returns the complete sentence."""
    tokenizer = PunktSentenceTokenizer()
    spans = list(tokenizer.span_tokenize(chunk))
    for start, end in spans:
        if start <= fragment_start <= end:
            return chunk[start:end].strip()
    return chunk[fragment_start:fragment_end].strip()

# Removed externalbrowser to use standard password auth that worked for ingest.py

# Define the exact CUAD prompts mapping — all 41 CUAD clause categories
CUAD_PROMPTS = {
    # ── Original 12 clauses ──
    "Effective Date": 'Highlight the parts (if any) of this contract related to "Effective Date" that should be reviewed by a lawyer. Details: The date when the contract is effective',
    "Governing Law": 'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer. Details: Which state/country\'s law governs the interpretation of the contract?',
    "Uncapped Liability": 'Highlight the parts (if any) of this contract related to "Uncapped Liability" that should be reviewed by a lawyer. Details: Is a party\'s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation.',
    "Cap on Liability": 'Highlight the parts (if any) of this contract related to "Cap On Liability" that should be reviewed by a lawyer. Details: Does the contract include a cap on liability upon the breach of a party\'s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.',
    "Non-Compete": 'Highlight the parts (if any) of this contract related to "Non-Compete" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?',
    "Exclusivity": 'Highlight the parts (if any) of this contract related to "Exclusivity" that should be reviewed by a lawyer. Details: Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure all \u201crequirements\u201d from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on collaborating or working with other parties), whether during the contract or after the contract ends (or both).',
    "Audit Rights": 'Highlight the parts (if any) of this contract related to "Audit Rights" that should be reviewed by a lawyer. Details: Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?',
    "Change of Control": 'Highlight the parts (if any) of this contract related to "Change Of Control" that should be reviewed by a lawyer. Details: Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?',
    "Liquidated Damages": 'Highlight the parts (if any) of this contract related to "Liquidated Damages" that should be reviewed by a lawyer. Details: Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?',
    "Termination for Convenience": 'Highlight the parts (if any) of this contract related to "Termination For Convenience" that should be reviewed by a lawyer. Details: Can a party terminate this contract without cause (solely by giving a notice and allowing a waiting period to expire)?',
    "IP Ownership Assignment": 'Highlight the parts (if any) of this contract related to "Ip Ownership Assignment" that should be reviewed by a lawyer. Details: Does intellectual property created by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?',
    "Source Code Escrow": 'Highlight the parts (if any) of this contract related to "Source Code Escrow" that should be reviewed by a lawyer. Details: Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy, insolvency, etc.)?',
    # ── 29 additional CUAD clauses ──
    "Document Name": 'Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract',
    "Parties": 'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer. Details: The two or more parties who signed the contract',
    "Agreement Date": 'Highlight the parts (if any) of this contract related to "Agreement Date" that should be reviewed by a lawyer. Details: The date of the contract',
    "Expiration Date": 'Highlight the parts (if any) of this contract related to "Expiration Date" that should be reviewed by a lawyer. Details: On what date will the contract\'s initial term expire?',
    "Renewal Term": 'Highlight the parts (if any) of this contract related to "Renewal Term" that should be reviewed by a lawyer. Details: What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice.',
    "Notice Period to Terminate Renewal": 'Highlight the parts (if any) of this contract related to "Notice Period To Terminate Renewal" that should be reviewed by a lawyer. Details: What is the notice period required to terminate renewal?',
    "Most Favored Nation": 'Highlight the parts (if any) of this contract related to "Most Favored Nation" that should be reviewed by a lawyer. Details: Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?',
    "No-Solicit of Customers": 'Highlight the parts (if any) of this contract related to "No-Solicit Of Customers" that should be reviewed by a lawyer. Details: Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?',
    "Competitive Restriction Exception": 'Highlight the parts (if any) of this contract related to "Competitive Restriction Exception" that should be reviewed by a lawyer. Details: This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above.',
    "No-Solicit of Employees": 'Highlight the parts (if any) of this contract related to "No-Solicit Of Employees" that should be reviewed by a lawyer. Details: Is there a restriction on a party\'s soliciting or hiring employees and/or contractors from the counterparty, whether during the contract or after the contract ends (or both)?',
    "Non-Disparagement": 'Highlight the parts (if any) of this contract related to "Non-Disparagement" that should be reviewed by a lawyer. Details: Is there a requirement on a party not to disparage the counterparty?',
    "ROFR/ROFO/ROFN": 'Highlight the parts (if any) of this contract related to "Rofr/Rofo/Rofn" that should be reviewed by a lawyer. Details: Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?',
    "Anti-Assignment": 'Highlight the parts (if any) of this contract related to "Anti-Assignment" that should be reviewed by a lawyer. Details: Is consent or notice required of a party if the contract is assigned to a third party?',
    "Revenue/Profit Sharing": 'Highlight the parts (if any) of this contract related to "Revenue/Profit Sharing" that should be reviewed by a lawyer. Details: Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?',
    "Price Restrictions": 'Highlight the parts (if any) of this contract related to "Price Restrictions" that should be reviewed by a lawyer. Details: Is there a restriction on the ability of a party to raise or reduce prices of technology, goods, or services provided?',
    "Minimum Commitment": 'Highlight the parts (if any) of this contract related to "Minimum Commitment" that should be reviewed by a lawyer. Details: Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?',
    "Volume Restriction": 'Highlight the parts (if any) of this contract related to "Volume Restriction" that should be reviewed by a lawyer. Details: Is there a fee increase or consent requirement, etc. if one party\'s use of the product/services exceeds certain threshold?',
    "Joint IP Ownership": 'Highlight the parts (if any) of this contract related to "Joint Ip Ownership" that should be reviewed by a lawyer. Details: Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?',
    "License Grant": 'Highlight the parts (if any) of this contract related to "License Grant" that should be reviewed by a lawyer. Details: Does the contract contain a license granted by one party to its counterparty?',
    "Non-Transferable License": 'Highlight the parts (if any) of this contract related to "Non-Transferable License" that should be reviewed by a lawyer. Details: Does the contract limit the ability of a party to transfer the license being granted to a third party?',
    "Affiliate License-Licensor": 'Highlight the parts (if any) of this contract related to "Affiliate License-Licensor" that should be reviewed by a lawyer. Details: Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?',
    "Affiliate License-Licensee": 'Highlight the parts (if any) of this contract related to "Affiliate License-Licensee" that should be reviewed by a lawyer. Details: Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?',
    "Unlimited/All-You-Can-Eat License": 'Highlight the parts (if any) of this contract related to "Unlimited/All-You-Can-Eat-License" that should be reviewed by a lawyer. Details: Is there a clause granting one party an \u201centerprise,\u201d \u201call you can eat\u201d or unlimited usage license?',
    "Irrevocable or Perpetual License": 'Highlight the parts (if any) of this contract related to "Irrevocable Or Perpetual License" that should be reviewed by a lawyer. Details: Does the contract contain a license grant that is irrevocable or perpetual?',
    "Post-Termination Services": 'Highlight the parts (if any) of this contract related to "Post-Termination Services" that should be reviewed by a lawyer. Details: Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?',
    "Warranty Duration": 'Highlight the parts (if any) of this contract related to "Warranty Duration" that should be reviewed by a lawyer. Details: What is the duration of any warranty against defects or errors in technology, products, or services provided under the contract?',
    "Insurance": 'Highlight the parts (if any) of this contract related to "Insurance" that should be reviewed by a lawyer. Details: Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?',
    "Covenant Not to Sue": 'Highlight the parts (if any) of this contract related to "Covenant Not To Sue" that should be reviewed by a lawyer. Details: Is a party restricted from contesting the validity of the counterparty\'s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?',
    "Third Party Beneficiary": 'Highlight the parts (if any) of this contract related to "Third Party Beneficiary" that should be reviewed by a lawyer. Details: Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?',
}

def extract_clause_with_bert(clause_type: str, context_text: str) -> str:
    """
    Extracts specific legal clauses from a given contract text using a fine-tuned BERT model.
    """
    print(f"🔧 Tool Invoked: BERT extraction for '{clause_type}'...")
    try:
        if clause_type not in CUAD_PROMPTS:
            return f"Error: '{clause_type}' is not a valid clause type. Valid types are: {', '.join(CUAD_PROMPTS.keys())}."
            
        extractor = get_bert_extractor()
        prompt = CUAD_PROMPTS[clause_type]
        
        context_capped = context_text[:50000]
        results = extractor(question=prompt, context=context_capped, top_k=3) 
        
        if isinstance(results, list):
            best_guess = results[0]
        else:
            best_guess = results
            
        score = best_guess.get('score', 0)
        answer = best_guess.get('answer', '').strip()
        start_idx = best_guess.get('start', 0)
        end_idx = best_guess.get('end', 0)
        
        if score < 0.0004 or not answer:
            return f"Status: No underlying risks detected by BERT (Confidence too low)."
            
        expanded_sentence = expand_fragment_to_sentence(context_capped, start_idx, end_idx)
        return f"Status: Extracted successfully\nConfidence: {score:.4f}\nExtracted Text: '{expanded_sentence}'"

    except Exception as e:
        print(f"\n❌ BERT ERROR: {str(e)}\n")
        return f"Error extracting clause: {str(e)}"

@st.cache_resource(ttl=3600)
def get_snowflake_connection():
    print("❄️ Opening persistent Snowflake connection...")

    return snowflake.connector.connect(
        user=os.getenv("SNOW_USER"),
        password=os.getenv("SNOW_PASS"),
        account=os.getenv("SNOW_ACCOUNT"),
        role=os.getenv("SNOW_ROLE", "TRAINING_ROLE"),
        warehouse=os.getenv("SNOW_WH", "COMPUTE_WH"),
        database=os.getenv("SNOW_DB", "LEXGUARD_DB"),
        schema=os.getenv("SNOW_SCHEMA", "CONTRACT_DATA")
    )

def retrieve_contract_clauses(search_term: str) -> str:
    """
    Searches the Snowflake database for specific legal contract clauses based on a keyword.
    Use this tool whenever the user asks about the contents of the contracts.

    Args:
        search_term: A specific keyword or short phrase to search for (e.g., "termination", "liability").

    Returns:
        A string containing the retrieved contract chunks, or an error message if the search fails.
    """
    print(f"🔧 Tool Invoked: Searching Snowflake for '{search_term}'...")
    
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        query = f"""
            SELECT CHUNK_ID, DOC_NAME, CHUNK_TEXT 
            FROM CONTRACT_CHUNKS 
            WHERE CHUNK_TEXT ILIKE '%{search_term}%'
            LIMIT 5;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            return f"No evidence found in the contracts for '{search_term}'."
            
        evidence = []
        for row in results:
            evidence.append(f"[Source: {row[1]} | Chunk ID: {row[0]}]\n{row[2]}")
            
        return "\n\n---\n\n".join(evidence)
        
    except Exception as e:
        # Added this so we can see the exact raw error in your Mac terminal
        print(f"\n❌ RAW SNOWFLAKE ERROR: {str(e)}\n") 
        return f"Database error: {str(e)}"

def calculate_risk_level(clause_text: str) -> str:
    """
    Analyzes a specific contract clause to determine if it contains high-risk language.
    Use this tool if the user asks to evaluate risk or danger in a clause.

    Args:
        clause_text: The exact text of the legal clause to evaluate.

    Returns:
        A string indicating 'High Risk', 'Medium Risk', or 'Low Risk' with a brief reason.
    """
    print("🔧 Tool Invoked: Calculating risk level...")
    
    text_lower = clause_text.lower()
    if "indemnify" in text_lower or "immediate termination" in text_lower:
        return "High Risk: Contains indemnification or immediate termination clauses."
    elif "penalty" in text_lower or "breach" in text_lower:
        return "Medium Risk: Mentions penalties or breach conditions."
    else:
        return "Low Risk: No standard high-risk keywords detected."


def retrieve_local_clauses(search_term: str, top_k: int = 5) -> str:
    """
    Searches the local separated JSON stores for contract clauses by keyword.
    This is the offline/local alternative to retrieve_contract_clauses(), inspired
    by HyperGraphRAG's separated entity_vdb + hyperedge_vdb retrieval pattern.

    Args:
        search_term: A keyword or phrase to search the clause index for.
        top_k: Maximum number of results to return.

    Returns:
        A string containing the matched contract chunks, or a not-found message.
    """
    print(f"🔧 Tool Invoked: Searching local store for '{search_term}'...")

    try:
        store = LocalStore(working_dir=config.HYPERPARAMS["working_dir"])
        results = store.search_hybrid(search_term, top_k=top_k)

        if not results:
            print(f"   ⚠️ Hard fallback to document opening...")
            all_chunks = store.get_all_chunks()
            if all_chunks:
                results = all_chunks[:top_k]

        if not results:
            return "" # Return empty string instead of english message so agent handles it

        evidence = []
        for r in results:
            source_tag = r.get("source", "Fallback")
            chunk_id = r.get("chunk_id", "Unknown")
            evidence.append(f"[Source: {r['doc_name']} | Chunk ID: {chunk_id} | Retriever: {source_tag}]\n{r['text']}")

        return "\n\n---\n\n".join(evidence)

    except Exception as e:
        print(f"\n❌ LOCAL STORE ERROR: {str(e)}\n")
        return ""

def extract_risk_clauses_llm(context: str) -> dict:
    """
    Full-document LLM-based risk clause detection with source text annotations.
    Replaces BERT for the 41 CUAD clause types. Returns structured results with
    exact excerpts the LLM used to make its determination.
    """
    from google import genai
    import json

    clause_descriptions = """
1. **Non-Compete**: Is there a restriction on a party's ability to compete with the counterparty?
2. **Governing Law**: Which state/country's law governs the contract?
3. **Audit Rights**: Does a party have the right to audit the counterparty's books/records?
4. **Change of Control**: Are there provisions for what happens during a merger, acquisition, or change of ownership?
5. **Effective Date**: What is the date the contract goes into effect?
6. **Uncapped Liability**: Is any party's liability uncapped upon breach?
7. **Cap on Liability**: Is there a maximum cap on liability or time limitation for claims?
8. **Exclusivity**: Is either party required to deal exclusively with the other?
9. **Liquidated Damages**: Are there pre-determined penalty amounts for breach?
10. **Termination for Convenience**: Can either party terminate without cause by giving notice?
11. **IP Ownership Assignment**: Is there any IP assignment, transfer, or sale obligation?
12. **Source Code Escrow**: Is source code required to be deposited in escrow?
13. **Document Name**: What is the official name or title of this contract?
14. **Parties**: Who are the two or more parties who signed the contract?
15. **Agreement Date**: What is the date the contract was signed or executed?
16. **Expiration Date**: On what date will the contract's initial term expire?
17. **Renewal Term**: What is the renewal term after the initial term expires?
18. **Notice Period to Terminate Renewal**: What is the notice period required to terminate renewal?
19. **Most Favored Nation**: Is there a most favored nation clause entitling a party to better terms given to third parties?
20. **No-Solicit of Customers**: Is a party restricted from soliciting customers or partners of the counterparty?
21. **Competitive Restriction Exception**: Are there exceptions or carveouts to Non-Compete, Exclusivity, or No-Solicit clauses?
22. **No-Solicit of Employees**: Is there a restriction on soliciting or hiring employees/contractors of the counterparty?
23. **Non-Disparagement**: Is there a requirement on a party not to disparage the counterparty?
24. **ROFR/ROFO/ROFN**: Is there a right of first refusal, offer, or negotiation clause?
25. **Anti-Assignment**: Is consent or notice required if the contract is assigned to a third party?
26. **Revenue/Profit Sharing**: Is one party required to share revenue or profit with the counterparty?
27. **Price Restrictions**: Is there a restriction on the ability to raise or reduce prices of goods/services?
28. **Minimum Commitment**: Is there a minimum order size or amount that must be purchased?
29. **Volume Restriction**: Is there a fee increase or consent requirement if usage exceeds a threshold?
30. **Joint IP Ownership**: Is there joint or shared ownership of intellectual property between the parties?
31. **License Grant**: Does the contract contain a license granted by one party to its counterparty?
32. **Non-Transferable License**: Does the contract limit the ability to transfer the license to a third party?
33. **Affiliate License-Licensor**: Does the contract include a license grant by affiliates of the licensor?
34. **Affiliate License-Licensee**: Does the contract include a license grant to the licensee's affiliates?
35. **Unlimited/All-You-Can-Eat License**: Is there an enterprise or unlimited usage license?
36. **Irrevocable or Perpetual License**: Does the contract contain a license that is irrevocable or perpetual?
37. **Post-Termination Services**: Is a party subject to obligations after termination or expiration?
38. **Warranty Duration**: What is the duration of any warranty against defects or errors?
39. **Insurance**: Is there an insurance requirement for one party's benefit?
40. **Covenant Not to Sue**: Is a party restricted from contesting the counterparty's IP ownership or bringing unrelated claims?
41. **Third Party Beneficiary**: Is there a non-contracting party who is a beneficiary and can enforce rights?"""

    prompt = f"""You are an expert contract risk auditor. Analyze this contract for the following 41 risk clause categories.

For EACH clause type, determine:
- Whether it is present in the contract (true/false)
- The risk level ("High", "Medium", "Low", or "None")
- The EXACT verbatim excerpt from the contract that contains this clause (copy-paste the literal text)
- The section or paragraph where you found it (e.g., "Section 8.3", "Article IV", "Paragraph 12")

CLAUSE CATEGORIES:
{clause_descriptions}

CRITICAL RULES:
1. The "excerpt" MUST be a DIRECT QUOTE from the contract — do not paraphrase or summarize.
2. Copy at least 1-3 complete sentences so the user can verify.
3. If the clause is NOT present, set detected=false, excerpt=null, section=null, risk_level="None".
4. Be thorough — search the ENTIRE document including appendices and exhibits.

OUTPUT FORMAT: Respond with ONLY a valid JSON object. Each key is the clause name. Example:
{{
    "Non-Compete": {{
        "detected": true,
        "risk_level": "High",
        "excerpt": "Employee shall not, during the Term...",
        "section": "Section 8.3"
    }},
    "Governing Law": {{
        "detected": true,
        "risk_level": "Low",
        "excerpt": "This Agreement shall be governed by the laws of the State of California...",
        "section": "Section 15"
    }}
}}
Do NOT wrap in markdown code fences.

CONTRACT TEXT:
{context[:200000]}"""

    try:
        print(f"🔧 Tool Invoked: LLM Risk Clause Scan (41 clauses, full-document)...")
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"\n❌ LLM CLAUSE SCAN ERROR: {str(e)}\n")
        return {}


def extract_contract_brief(context: str) -> dict:
    """
    V4 Full-Document Entity Extraction — extracts all 8 core CUAD metadata entities
    using chain-of-thought prompting for multi-hop date reasoning and perpetual term detection.
    Benchmarked at 88-100% accuracy across all entities.
    """
    from google import genai
    import json
    
    prompt = """You are an expert contract reviewer. Extract the following metadata from the provided contract text.

Think step-by-step for each entity. Apply these critical reasoning rules:
1. For dates: If the contract says "effective as of the date first written above" or similar, the Effective Date IS the Agreement Date.
2. For expiration: If stated as "N years from [date]", CALCULATE the actual calendar date. If the contract has no end date or says "perpetual" or "until terminated", answer "Perpetual".
3. For renewal: Look for "automatically renew", "successive periods", "extend". If the contract is perpetual with no termination date, the renewal is also "Perpetual".
4. For notice to terminate: Search the ENTIRE document — both the "Term" section AND the "Termination" section. Look for "written notice", "days prior", "notice of non-renewal".
5. NEVER answer "NOT FOUND" if the information exists somewhere in the document.

ENTITIES TO EXTRACT:
1. **Document Name**: The official title or name of the contract/agreement as stated in the document header.
2. **Parties**: The two or more parties who signed the contract. Return only entity/individual names separated by semicolons.
3. **Agreement Date**: The date the contract was signed, executed, or 'made as of'. Format as MM/DD/YYYY.
4. **Effective Date**: The date the contract goes into effect. If 'effective as of the Agreement Date', use that date.
5. **Expiration Date**: The date the initial term expires. Calculate if needed. If perpetual, answer 'Perpetual'.
6. **Renewal Term**: How long the contract auto-renews (e.g., '1 year'). If perpetual, answer 'Perpetual'. If none, 'NOT FOUND'.
7. **Notice to Terminate Renewal**: Advance notice period to prevent auto-renewal (e.g., '30 days', '60 days'). If none, 'NOT FOUND'.
8. **Governing Law**: Which state or country's law governs the contract.

OUTPUT FORMAT: Respond with ONLY a valid JSON object with these exact keys:
"Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term", "Notice to Terminate Renewal", "Governing Law"
If any information is completely missing, use "NOT FOUND".
Do NOT wrap in markdown code fences.

CONTRACT TEXT:
""" + context[:200000]

    try:
        print(f"🔧 Tool Invoked: Generating V4 Full Contract Brief (8 entities)...")
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"\n❌ BRIEF EXTRACTION ERROR: {str(e)}\n")
        return {}