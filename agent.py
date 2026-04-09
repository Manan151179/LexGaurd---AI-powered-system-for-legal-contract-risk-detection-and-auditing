import os
import time
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import the tools we defined in tools.py
from tools import retrieve_contract_clauses, calculate_risk_level

# Configure logging for the Streamlit terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key from .env
load_dotenv()
# Initialize the NEW Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Removed global SYSTEM_PROMPT. Will be generated dynamically.

# Map string names to the actual Python functions so the loop can call them
AVAILABLE_TOOLS = {
    "retrieve_contract_clauses": retrieve_contract_clauses,
    "calculate_risk_level": calculate_risk_level
}

# 2. The Execution Loop
def run_lexguard_agent(user_query: str, local_context: str = None, pre_extracted_clauses: dict = None) -> dict:
    """
    Runs the multi-step reasoning loop for LexGuard.
    Handles tool routing, captures errors, and prevents infinite loops.

    Returns:
        dict with keys: response (str), trace (list[dict]), tool_calls (list[str]),
                        retrieval_count (int), risk_level (str), success (bool)
    """

    trace = []       # Structured execution trace
    tool_names = []  # Names of tools called
    retrieval_count = 0
    risk_level = "N/A"

    # Define tools dynamically so they can access the local context without LLM passing it
    def extract_local_clause(clause_type: str) -> str:
        """
        Extracts specific legal clauses from the user's uploaded local contract document using a BERT model.
        Use this tool when you need to find where a specific clause exists in the provided upload.

        Args:
            clause_type: The clause type. Valid options: 'Non-Compete', 'Governing Law', 'Audit Rights', 'Change of Control', 'Effective Date', 'Uncapped Liability', 'Cap on Liability', 'Exclusivity', 'Liquidated Damages', 'Termination for Convenience', 'IP Ownership Assignment', 'Source Code Escrow'.
        """
        if pre_extracted_clauses and clause_type in pre_extracted_clauses:
            return pre_extracted_clauses[clause_type]
            
        if not local_context:
            return "Error: No local document provided by the user."
        from tools import extract_clause_with_bert
        return extract_clause_with_bert(clause_type, local_context)

    def get_all_extracted_clauses() -> str:
        """
        Retrieves ALL 41 pre-extracted risk clauses from the contract at once.
        Use this tool when the user asks to "list all risks", "show all extracted clauses", or requests a comprehensive risk profile.
        """
        if not pre_extracted_clauses:
            return "Error: No pre-extracted clauses available. Ensure a document was uploaded and processed."
        
        report = []
        for clause, text in pre_extracted_clauses.items():
            report.append(f"--- {clause} ---\n{text}")
        return "\n".join(report)

    def answer_general_contract_question(question: str) -> str:
        """
        Answers general questions or generates summaries about the uploaded contract.
        CRITICAL: DO NOT use this tool if the user's question relates to ANY of the specific clauses supported by `extract_local_clause` (e.g. Non-Compete, Terminations, Liability). 
        You must only use this tool for global summaries, main purposes, or extremely broad questions.
        
        Args:
            question: The global question the user is asking. e.g. "What is the summary of this contract?"
        """
        if not local_context:
            return "Error: No local document provided by the user."
            
        print(f"🌍 Path B Triggered: Generating global answer for '{question}'")
        strict_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        strict_prompt = f"""You are a strict legal assistant evaluating a document.
Your ONLY job is to answer the user's question based strictly on the provided contract text below.

STRICT RULES:
1. Do NOT use outside knowledge. If the answer is not in the text, you MUST reply: "The contract does not specify this."
2. Do NOT give legal advice.
3. Keep the summary high-level. Do NOT extract specific liability caps or dates if the user didn't explicitly ask for them. If a clause is highly complex, simply state that it requires human review.

User's Question: {question}

Contract Text:
{local_context[:50000]}
"""
        try:
            response = strict_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=strict_prompt,
            )
            return response.text
        except Exception as e:
            return f"Error generating general answer: {str(e)}"

    if local_context:
        SYSTEM_PROMPT = """You are LexGuard, a Neuro-Symbolic Compliance Auditor using a Dual-Path Architecture.
The user has UPLOADED A LOCAL CONTRACT. You must analyze the user's request and route it:

PATH A (The Surgical Strike):
If the user asks about specific risks, clauses, or liabilities, YOU MUST USE 'extract_local_clause'. 
DO NOT use 'answer_general_contract_question' for extracting specific risks or identifying specific terms if they are listed as valid options in 'extract_local_clause'.
1. Route to 'extract_local_clause' to find precise, verbatim legal clauses.
2. Calculate any risks using common sense or the 'calculate_risk_level' tool.

PATH B (The Global Summary):
ONLY use 'answer_general_contract_question' if the user asks a broad global question (e.g., "Summarize this", "What is the main purpose?"). 
If the user's query asks about risks that are not included in 'extract_local_clause' (e.g. "Indemnification"), only then may you use this global tool to evaluate the risk.

PATH C (The Comprehensive Audit):
If the user asks to "list all risks", "show all extractions", or "give a full risk profile":
1. Route to 'get_all_extracted_clauses' to fetch the complete pre-computed BERT extractions.
2. Present the findings clearly to the user, calling out anything you identify as particularly high-risk.

AUDIT TRAIL REQUIRED:
Whatever path you take, you MUST explicitly cite the exact source text, Document Name, or Chunk ID that you relied on for your answer. Do not hide the raw text from the user. Offer it as a "Citation" so they can double-check your work.
"""
        available_tools = {
            "extract_local_clause": extract_local_clause,
            "get_all_extracted_clauses": get_all_extracted_clauses,
            "answer_general_contract_question": answer_general_contract_question,
            "calculate_risk_level": calculate_risk_level
        }
    else:
        SYSTEM_PROMPT = """
You are LexGuard, a Neuro-Symbolic Compliance Auditor. Your job is to audit Residential Lease Agreements against strict compliance rules.
You operate on a 'Recall-Then-Reason' pipeline:
1. First, use 'retrieve_contract_clauses' to search the local index for specific legal clauses by keyword.
2. Second, use 'calculate_risk_level' to evaluate whether a clause contains high-risk language.

Never guess or assume contract details. Always use your tools. If a tool returns an error, read the error and try a different search term or query. Once you have gathered all necessary context, provide a final compliance verdict explaining if the clause passes or fails.

**AUDIT TRAIL REQUIRED:**
You MUST explicitly cite the exact `Chunk ID`, `Source`, and a snippet of the raw literal text you relied upon at the end of every answer under a "Citations:" header. The user relies on this to independently verify your claims.
"""
        available_tools = AVAILABLE_TOOLS

    # Start a chat session using the new SDK configuration
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=list(available_tools.values()),
            temperature=0.1, # Keep temperature low for analytical tasks
        )
    )

    max_steps = 5  # Prevent infinite loops
    current_step = 0

    print(f"\n🧠 LexGuard starting audit for query: '{user_query}'")
    trace.append({"step": "start", "detail": f"Query: {user_query}"})

    # The initial prompt is the user's question
    prompt = user_query

    while current_step < max_steps:
        current_step += 1
        step_start = time.time()

        try:
            # Send the prompt (or the tool results) to Gemini
            response = chat.send_message(prompt)
        except Exception as e:
            error_msg = f"API Error communicating with Gemini: {str(e)}"
            logger.error(error_msg)
            trace.append({"step": "error", "detail": error_msg, "time": 0})
            return {"response": error_msg, "trace": trace, "tool_calls": tool_names,
                    "retrieval_count": retrieval_count, "risk_level": risk_level, "success": False}

        step_elapsed = round(time.time() - step_start, 2)

        # Check if Gemini decided it needs to use a tool (New SDK syntax)
        if response.function_calls:
            tool_responses = [] # We need to collect the results to send back

            for tool_call in response.function_calls:
                tool_name = tool_call.name
                tool_args = tool_call.args

                print(f"🛠️ Agent called tool: '{tool_name}' with arguments: {tool_args}")

                # Execute the tool and catch any errors
                tool_start = time.time()
                if tool_name in AVAILABLE_TOOLS:
                    try:
                        tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    except Exception as e:
                        tool_result = f"Error executing {tool_name}: {str(e)}"
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."

                tool_elapsed = round(time.time() - tool_start, 2)
                tool_names.append(tool_name)

                # Track retrieval count
                if tool_name == "retrieve_contract_clauses":
                    retrieval_count += str(tool_result).count("[Source:")
                # Track risk level
                if tool_name == "calculate_risk_level":
                    if "High Risk" in str(tool_result):
                        risk_level = "High"
                    elif "Medium Risk" in str(tool_result):
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"

                trace.append({
                    "step": f"tool_call",
                    "tool": tool_name,
                    "args": str(tool_args),
                    "result_preview": str(tool_result)[:150],
                    "time": tool_elapsed
                })

                # Format the response exactly how the new SDK requires it
                tool_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": str(tool_result)}
                    )
                )

            # Set the prompt for the next loop iteration to be the tool results
            prompt = tool_responses

        else:
            # If there are no function calls, Gemini gave us the final text!
            print("\n✅ Final Verdict Reached.")
            trace.append({"step": "response", "detail": "Final verdict generated", "time": step_elapsed})
            return {"response": response.text, "trace": trace, "tool_calls": tool_names,
                    "retrieval_count": retrieval_count, "risk_level": risk_level, "success": True}

    # If it hits max steps without returning, force it to stop
    timeout_msg = "⚠️ LexGuard Audit failed: Maximum reasoning steps exceeded."
    logger.error(timeout_msg)
    trace.append({"step": "timeout", "detail": timeout_msg, "time": 0})
    return {"response": timeout_msg, "trace": trace, "tool_calls": tool_names,
            "retrieval_count": retrieval_count, "risk_level": risk_level, "success": False}

# Test the loop directly in the terminal
if __name__ == "__main__":
    test_query = "Audit the lease agreements to see if the pet deposit amount is compliant."
    result = run_lexguard_agent(test_query)
    print(f"\n[FINAL OUTPUT]\n{result['response']}")
    print(f"\n[TRACE]\n{result['trace']}")