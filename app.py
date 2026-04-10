import streamlit as st
import os
import time
import fitz  # PyMuPDF

# Import our working agent loops
from agent import run_lexguard_agent as run_baseline_agent
from monitor import MetricsCollector
import chat_history

# Init local SQLite chat history tables on startup
chat_history.init_tables()

# ═══════════════════════════════════════════════
# 1. Page Configuration & Custom CSS
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="LexGuard Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme State ──
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

is_dark = st.session_state.theme == "dark"

# Theme-aware CSS with variables
if is_dark:
    theme_css = """
    :root {
        --bg-primary: #0F172A;
        --bg-card: rgba(30, 41, 59, 0.6);
        --bg-trace: rgba(15, 23, 42, 0.5);
        --text-primary: #E2E8F0;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --border-color: rgba(148, 163, 184, 0.15);
        --accent-purple: #7C3AED;
        --accent-blue: #3B82F6;
        --history-bg: rgba(30, 41, 59, 0.4);
        --history-text: #CBD5E1;
        --doc-viewer-bg: rgba(15, 23, 42, 0.4);
    }
    """
else:
    theme_css = """
    :root {
        --bg-primary: #FFFFFF;
        --bg-card: rgba(241, 245, 249, 0.8);
        --bg-trace: rgba(241, 245, 249, 0.6);
        --text-primary: #1E293B;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --border-color: rgba(148, 163, 184, 0.25);
        --accent-purple: #7C3AED;
        --accent-blue: #3B82F6;
        --history-bg: rgba(241, 245, 249, 0.6);
        --history-text: #334155;
        --doc-viewer-bg: rgba(248, 250, 252, 0.9);
    }
    .stApp { background: #F8FAFC !important; }

    /* Streamlit header bar — match light theme */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    [data-testid="stAppHeader"] { background: #F8FAFC !important; }
    [data-testid="stToolbar"] { background: transparent !important; }
    [data-testid="stToolbar"] button { color: #475569 !important; }
    [data-testid="stDecoration"] { background: transparent !important; display: none !important; }

    /* Sidebar top area — remove dark decoration */
    [data-testid="stSidebar"]::before { background: transparent !important; }
    [data-testid="stSidebarHeader"],
    [data-testid="stSidebar"] > div:first-child { background: #F1F5F9 !important; }
    [data-testid="stSidebarCollapsedControl"] button { color: #475569 !important; background: #F1F5F9 !important; }

    /* Chat message containers — white background */
    .stChatMessage { color: #1E293B !important; }
    [data-testid="stChatMessage"] { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 12px !important; }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] div { color: #1E293B !important; }
    [data-testid="stChatMessage"] code { background: #F1F5F9 !important; color: #7C3AED !important; }
    [data-testid="stChatMessage"] pre { background: #F1F5F9 !important; }

    /* Chat input box — white background */
    [data-testid="stChatInput"] { background: transparent !important; }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input { background: #FFFFFF !important; color: #1E293B !important; border: 1px solid #CBD5E1 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #94A3B8 !important; }
    .stChatInput > div { background: #FFFFFF !important; }

    /* Bottom chat input container bar */
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"],
    .stBottom > div,
    [data-testid="stBottom"] > div { background: #F8FAFC !important; }
    section[data-testid="stBottom"] { background: #F8FAFC !important; }

    /* File uploader */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] { background: #FFFFFF !important; border-color: #CBD5E1 !important; color: #1E293B !important; }
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] p { color: #475569 !important; }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"] { background: linear-gradient(135deg, #7C3AED, #3B82F6) !important; color: #FFFFFF !important; border: none !important; border-radius: 8px !important; font-weight: 500 !important; }
    [data-testid="stFileUploaderDropzone"] button:hover { opacity: 0.9 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #F1F5F9 !important; }
    [data-testid="stSidebar"] * { color: #1E293B !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #FFFFFF !important; }

    /* General text */
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #1E293B !important; }
    .stExpander { border-color: var(--border-color) !important; }
    [data-testid="stExpander"] { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px !important; }

    /* Spinners and alerts */
    .stSpinner > div { color: #1E293B !important; }
    .stAlert { background: #FFFFFF !important; color: #1E293B !important; border-color: #E2E8F0 !important; }

    /* Select boxes, toggles, radios */
    [data-testid="stWidgetLabel"] { color: #1E293B !important; }
    .stSelectbox > div > div { background: #FFFFFF !important; color: #1E293B !important; }
    .stToggle label span { color: #1E293B !important; }

    /* Main block container backgrounds */
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"] { background: transparent !important; }

    /* Table styling */
    table { color: #1E293B !important; }
    th { background: #F1F5F9 !important; color: #1E293B !important; }
    td { color: #334155 !important; }
    """

st.markdown(f"""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    {theme_css}

    /* Global */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* Hide Streamlit branding but keep sidebar toggle */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Main header */
    .main-header {{
        background: linear-gradient(135deg, #7C3AED 0%, #3B82F6 50%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
        letter-spacing: -0.5px;
    }}

    .sub-header {{
        color: var(--text-secondary);
        font-size: 0.95rem;
        font-weight: 300;
        margin-top: -8px;
        margin-bottom: 24px;
    }}

    /* Glass cards */
    .glass-card {{
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.3s ease;
        color: var(--text-primary);
    }}
    .glass-card:hover {{
        border-color: rgba(124, 58, 237, 0.4);
    }}

    /* Metric badges */
    .metric-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }}
    .badge-purple {{ background: rgba(124, 58, 237, 0.2); color: #A78BFA; }}
    .badge-blue   {{ background: rgba(59, 130, 246, 0.2); color: #93C5FD; }}
    .badge-green  {{ background: rgba(16, 185, 129, 0.2); color: #6EE7B7; }}
    .badge-amber  {{ background: rgba(245, 158, 11, 0.2); color: #FCD34D; }}
    .badge-red    {{ background: rgba(239, 68, 68, 0.2);  color: #FCA5A5; }}

    /* Latency tag */
    .latency-tag {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        background: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        margin-left: 8px;
    }}

    /* Risk badges */
    .risk-high   {{ background: rgba(239, 68, 68, 0.15); color: {'#FCA5A5' if is_dark else '#DC2626'}; padding: 2px 10px; border-radius: 6px; font-weight: 600; }}
    .risk-medium {{ background: rgba(245, 158, 11, 0.15); color: {'#FCD34D' if is_dark else '#D97706'}; padding: 2px 10px; border-radius: 6px; font-weight: 600; }}
    .risk-low    {{ background: rgba(16, 185, 129, 0.15); color: {'#6EE7B7' if is_dark else '#059669'}; padding: 2px 10px; border-radius: 6px; font-weight: 600; }}

    /* Trace log styling */
    .trace-step {{
        padding: 6px 12px;
        border-left: 3px solid #7C3AED;
        margin: 6px 0;
        font-size: 0.8rem;
        background: var(--bg-trace);
        border-radius: 0 6px 6px 0;
        color: var(--text-primary);
    }}

    /* Query history items */
    .history-item {{
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        cursor: pointer;
        font-size: 0.8rem;
        background: var(--history-bg);
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
        color: var(--history-text);
    }}
    .history-item:hover {{
        background: rgba(124, 58, 237, 0.15);
        border-color: rgba(124, 58, 237, 0.3);
    }}

    /* Status indicators */
    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .status-online  {{ background: #10B981; box-shadow: 0 0 6px #10B981; }}
    .status-offline {{ background: #EF4444; box-shadow: 0 0 6px #EF4444; }}

    /* Sidebar section titles */
    .sidebar-title {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-muted);
        margin-top: 16px;
        margin-bottom: 8px;
    }}

    /* Streamlit expander override */
    .streamlit-expanderHeader {{
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }}

    /* Animated gradient border effect for chat input */
    .stChatInput > div {{
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 12px !important;
    }}
    .stChatInput > div:focus-within {{
        border-color: #7C3AED !important;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.15) !important;
    }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 2. Session State Initialization
# ═══════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am **LexGuard** ⚖️. What contract clauses would you like me to audit today?",
         "trace": None, "latency": None, "risk": None}
    ]

if "collector" not in st.session_state:
    st.session_state.collector = MetricsCollector()

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = chat_history.new_session_id()

if "local_context" not in st.session_state:
    st.session_state.local_context = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

def parse_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# ═══════════════════════════════════════════════
# 3. Sidebar
# ═══════════════════════════════════════════════
with st.sidebar:
    # ── File Upload (Local Mode) ──
    st.markdown('<div class="sidebar-title">📄 Upload Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a Contract (PDF/TXT)", type=["pdf", "txt"], help="If uploaded, LexGuard will analyze this document locally.")
    
    if uploaded_file is not None:
        if st.session_state.uploaded_filename != uploaded_file.name:
            with st.spinner("Parsing document..."):
                if uploaded_file.name.lower().endswith(".pdf"):
                    st.session_state.local_context = parse_pdf(uploaded_file)
                else:
                    st.session_state.local_context = uploaded_file.read().decode("utf-8")
                st.session_state.uploaded_filename = uploaded_file.name
                
            with st.spinner("🧠 Scanning document for 41 critical legal risks (LLM Full-Doc)..."):
                from tools import extract_risk_clauses_llm
                risk_data = extract_risk_clauses_llm(st.session_state.local_context)
                st.session_state.local_extracted_clauses = risk_data

            # Phase 3: BERT Cross-Validation
            bert_data = {}
            with st.spinner("🔬 Cross-validating with BERT model (41 clauses)..."):
                from tools import batch_bert_extraction, cross_validate_results
                bert_data = batch_bert_extraction(st.session_state.local_context)

            # Merge and cross-validate
            if risk_data and bert_data:
                merged_data = cross_validate_results(risk_data, bert_data)
                st.session_state.local_extracted_clauses = merged_data
            elif risk_data:
                merged_data = risk_data
            else:
                merged_data = {}

            if merged_data:
                    # Build annotated risk report with cross-validation badges
                    risk_md = f"**⚠️ Risk Clause Audit: {uploaded_file.name}**\n\n"
                    
                    detected_clauses = []
                    not_found_clauses = []
                    review_clauses = []
                    ref_counter = 0
                    
                    for clause_name, info in merged_data.items():
                        if isinstance(info, dict):
                            llm_det = info.get("llm_detected", info.get("detected", False))
                            bert_det = info.get("bert_detected", False)
                            if llm_det or bert_det:
                                ref_counter += 1
                                detected_clauses.append((clause_name, info, ref_counter))
                                if info.get("needs_review", False):
                                    review_clauses.append(clause_name)
                            else:
                                not_found_clauses.append(clause_name)
                    
                    if detected_clauses:
                        # Risk summary table with cross-validation badges
                        risk_md += "| # | Clause | Risk | Verification | BERT Conf. | Section |\n|:---|:---|:---|:---|:---|:---|\n"
                        for clause_name, info, ref_num in detected_clauses:
                            risk_level = info.get("risk_level", "Unknown")
                            section = info.get("section", "—")
                            risk_icon = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                            
                            # Cross-validation badge
                            agreement = info.get("agreement", "")
                            if agreement == "agreed":
                                verify_badge = "✅ Verified"
                            elif agreement == "disagreement":
                                verify_badge = "⚠️ Review"
                            elif agreement == "llm_only":
                                verify_badge = "🤖 LLM-Only"
                            elif agreement == "bert_only":
                                verify_badge = "🔬 BERT-Only"
                            else:
                                verify_badge = "—"
                            
                            bert_conf = info.get("bert_confidence", 0)
                            conf_str = f"{bert_conf:.2%}" if bert_conf > 0 else "—"
                            
                            risk_md += f"| [{ref_num}] | {risk_icon} **{clause_name}** | {risk_level} | {verify_badge} | {conf_str} | {section} |\n"
                        
                        # Summary stats
                        agreed_count = sum(1 for _, i, _ in detected_clauses if i.get("agreement") == "agreed")
                        review_count = len(review_clauses)
                        risk_md += f"\n✅ **{len(detected_clauses)}** risks detected, **{agreed_count}** verified by both models"
                        if review_count > 0:
                            risk_md += f", **{review_count}** flagged for human review"
                        risk_md += f", **{len(not_found_clauses)}** clauses not found.\n"
                    else:
                        risk_md += "✅ No critical risk clauses detected in this contract.\n"
                    
                    # Append risk summary as an assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": risk_md,
                        "trace": None, "latency": None, "risk": "High" if detected_clauses else "Low",
                        "annotations": detected_clauses  # Store for expandable detail
                    })
                    chat_history.save_message(st.session_state.session_id, "assistant", risk_md, risk_level="High" if detected_clauses else "Low", annotations=detected_clauses if detected_clauses else None)
                
            with st.spinner("📑 Generating Contract Metadata Brief (V4 — 8 entities)..."):
                from tools import extract_contract_brief
                brief_data = extract_contract_brief(st.session_state.local_context)
                
                if brief_data:
                    # Build a rich formatted brief with all 8 entities
                    brief_md = f"**📑 Contract Brief: {uploaded_file.name}**\n\n"
                    brief_md += "| Field | Value |\n|:---|:---|\n"
                    
                    # Entity display config with icons
                    icons = {
                        "Document Name": "📄",
                        "Parties": "👥",
                        "Agreement Date": "📅",
                        "Effective Date": "🟢",
                        "Expiration Date": "🔴",
                        "Renewal Term": "🔄",
                        "Notice to Terminate Renewal": "⚠️",
                        "Governing Law": "⚖️"
                    }
                    
                    for k, v in brief_data.items():
                        icon = icons.get(k, "📌")
                        brief_md += f"| {icon} **{k}** | {v} |\n"
                    
                    # Append it as an official Assistant message so it shows in chat right away
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": brief_md,
                        "trace": None, "latency": None, "risk": "Low"
                    })
                    chat_history.save_message(st.session_state.session_id, "assistant", brief_md, risk_level="Low")
                    st.rerun() # Refresh chat UI instantly
                    
            st.success(f"Loaded & Analyzed: {uploaded_file.name}")
    else:
        if st.session_state.local_context is not None:
            st.session_state.local_context = None
            st.session_state.uploaded_filename = None

    # ── Theme Toggle ──
    st.markdown('<div class="sidebar-title">🎨 Theme</div>', unsafe_allow_html=True)
    theme_toggle = st.toggle(
        "🌙 Dark Mode" if is_dark else "☀️ Light Mode",
        value=is_dark,
        key="theme_toggle"
    )
    if theme_toggle != is_dark:
        st.session_state.theme = "dark" if theme_toggle else "light"
        st.rerun()

    # ── Pipeline Info ──
    pipeline_choice = "Baseline (Gemini API)"
    st.markdown('<div class="sidebar-title">⚙️ Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""<div class="glass-card">
        <span class="metric-badge badge-blue">GEMINI</span>
        Gemini 2.5 Flash + Full-Document Extraction + Hybrid Search
    </div>""", unsafe_allow_html=True)

    # ── System Status ──
    st.markdown('<div class="sidebar-title">📡 System Status</div>', unsafe_allow_html=True)

    gemini_status = "online" if os.getenv("GEMINI_API_KEY") else "offline"
    store_status = "online"

    st.markdown(f"""<div class="glass-card">
        <div><span class="status-dot status-{gemini_status}"></span> Gemini API</div>
        <div><span class="status-dot status-{store_status}"></span> Local Store (SQLite)</div>
    </div>""", unsafe_allow_html=True)

    # ── Chat History ──
    st.markdown('<div class="sidebar-title">💬 Chat History</div>', unsafe_allow_html=True)
    
    col_new, col_id = st.columns([1, 1])
    with col_new:
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.session_id = chat_history.new_session_id()
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am **LexGuard** ⚖️. What contract clauses would you like me to audit today?",
                 "trace": None, "latency": None, "risk": None}
            ]
            st.session_state.query_history = []
            st.rerun()
    with col_id:
        st.markdown(f'<small style="color:var(--text-muted)">ID: {st.session_state.session_id}</small>', unsafe_allow_html=True)
    
    # List saved sessions
    sessions = chat_history.list_sessions(limit=10)
    if sessions:
        for sess in sessions:
            is_current = sess["id"] == st.session_state.session_id
            col_sess, col_del = st.columns([5, 1])
            with col_sess:
                label = f"{'▶ ' if is_current else ''}{sess['title'][:40]}"
                if st.button(label, key=f"sess_{sess['id']}", use_container_width=True, disabled=is_current):
                    loaded = chat_history.load_session(sess["id"])
                    if loaded:
                        st.session_state.session_id = sess["id"]
                        st.session_state.messages = loaded
                        st.session_state.query_history = []
                        st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{sess['id']}", help="Delete this session"):
                    chat_history.delete_session(sess["id"])
                    if is_current:
                        st.session_state.session_id = chat_history.new_session_id()
                        st.session_state.messages = [
                            {"role": "assistant", "content": "Hello! I am **LexGuard** ⚖️. What contract clauses would you like me to audit today?",
                             "trace": None, "latency": None, "risk": None}
                        ]
                        st.session_state.query_history = []
                    st.rerun()
    
    # ── Analytics Dashboard ──
    collector = st.session_state.collector
    if collector.total_queries() > 0:
        st.markdown('<div class="sidebar-title">📊 Session Analytics</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Queries", collector.total_queries())
        col2.metric("Avg Latency", f"{collector.avg_latency()}s")

        col3, col4 = st.columns(2)
        col3.metric("Success Rate", f"{collector.success_rate()}%")

        # Pipeline breakdown
        breakdown = collector.pipeline_breakdown()
        if len(breakdown) > 1:
            st.markdown("**Pipeline Usage:**")
            for pipeline, count in breakdown.items():
                short_name = pipeline.split("(")[0].strip()
                st.progress(count / collector.total_queries(), text=f"{short_name}: {count}")

        # Average latency by pipeline
        avg_by_pipeline = collector.avg_latency_by_pipeline()
        if avg_by_pipeline:
            st.markdown("**Avg Latency by Pipeline:**")
            for pipeline, avg in avg_by_pipeline.items():
                short_name = pipeline.split("(")[0].strip()
                st.markdown(f"<span class='metric-badge badge-blue'>{short_name}</span> {avg}s", unsafe_allow_html=True)

        # Tool usage
        tool_usage = collector.tool_usage_breakdown()
        if tool_usage:
            st.markdown("**Tool Calls:**")
            for tool, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
                st.markdown(f"<span class='metric-badge badge-green'>{tool}</span> ×{count}", unsafe_allow_html=True)

    # ── Query History ──
    if st.session_state.query_history:
        st.markdown('<div class="sidebar-title">🕐 Query History</div>', unsafe_allow_html=True)
        for i, item in enumerate(reversed(st.session_state.query_history[-10:])):
            risk_badge = ""
            if item.get("risk") == "High":
                risk_badge = '<span class="risk-high">HIGH</span>'
            elif item.get("risk") == "Medium":
                risk_badge = '<span class="risk-medium">MED</span>'
            elif item.get("risk") == "Low":
                risk_badge = '<span class="risk-low">LOW</span>'

            st.markdown(f"""<div class="history-item">
                <div>{item['query'][:50]}{'...' if len(item['query']) > 50 else ''}</div>
                <div style="margin-top:4px">
                    <span class="latency-tag">{item.get('latency', '?')}s</span>
                    {risk_badge}
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 4. Main Chat Area
# ═══════════════════════════════════════════════
if st.session_state.local_context:
    main_col, doc_col = st.columns([1, 1], gap="large")
    with doc_col:
        st.markdown('<div class="sidebar-title">📄 Uploaded Contract View</div>', unsafe_allow_html=True)
        # Using a direct code injection for newlines since markdown often drops them
        st.markdown(f'<div class="glass-card" style="height: 600px; overflow-y: scroll; font-size: 0.85rem; padding: 15px; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(124, 58, 237, 0.2);">{st.session_state.local_context.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
else:
    main_col = st.container()

with main_col:
    st.markdown('<h1 class="main-header">⚖️ LexGuard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Neuro-Symbolic Compliance Auditor for Contract Risk Analysis</p>', unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show latency badge on assistant messages
        if msg["role"] == "assistant" and msg.get("latency") is not None:
            risk = msg.get("risk", "N/A")
            risk_class = "risk-high" if risk == "High" else "risk-medium" if risk == "Medium" else "risk-low" if risk == "Low" else "badge-blue"
            st.markdown(f"""
                <span class="latency-tag">⏱ {msg['latency']}s</span>
                <span class="{risk_class}" style="font-size:0.75rem">{risk} Risk</span>
            """, unsafe_allow_html=True)

        # Show expandable source annotations for risk clause audit
        if msg["role"] == "assistant" and msg.get("annotations"):
            for clause_name, info, ref_num in msg["annotations"]:
                risk_level = info.get("risk_level", "Unknown")
                excerpt = info.get("excerpt", "No excerpt available")
                section = info.get("section", "Unknown section")
                risk_color = "#EF4444" if risk_level == "High" else "#F59E0B" if risk_level == "Medium" else "#10B981"
                
                with st.expander(f"[{ref_num}] 📋 {clause_name} — {risk_level} Risk ({section})"):
                    st.markdown(f"""
<div style="border-left: 4px solid {risk_color}; padding: 12px 16px; background: rgba(15, 23, 42, 0.4); border-radius: 6px; margin: 8px 0;">
    <p style="font-size: 0.75rem; color: #94A3B8; margin-bottom: 6px;">📍 Source: {section} | Risk Level: <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span></p>
    <p style="font-size: 0.9rem; color: #E2E8F0; font-style: italic; line-height: 1.6;">"{excerpt}"</p>
</div>
                    """, unsafe_allow_html=True)

        # Show debug trace in expandable panel
        if msg["role"] == "assistant" and msg.get("trace"):
            with st.expander("🔍 Execution Trace & Debug Log"):
                for step in msg["trace"]:
                    step_type = step.get("step", "")
                    step_time = step.get("time", "")
                    time_str = f" — {step_time}s" if step_time else ""

                    if step_type == "start":
                        st.markdown(f'<div class="trace-step">📝 <b>Query Received:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "tool_call":
                        tool = step.get("tool", "unknown")
                        preview = step.get("result_preview", "")
                        st.markdown(f'<div class="trace-step">🛠️ <b>Tool Call:</b> <code>{tool}</code>{time_str}<br><small style="color:#94A3B8">{preview}</small></div>', unsafe_allow_html=True)
                    elif step_type == "model_inference":
                        st.markdown(f'<div class="trace-step">🤖 <b>Model Inference:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "response":
                        st.markdown(f'<div class="trace-step">✅ <b>Response Generated</b>{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "error":
                        st.markdown(f'<div class="trace-step" style="border-color:#EF4444">❌ <b>Error:</b> {step.get("detail", "")}</div>', unsafe_allow_html=True)
                    elif step_type == "greeting_filter":
                        st.markdown(f'<div class="trace-step">👋 <b>Greeting Detected</b> — skipped pipeline</div>', unsafe_allow_html=True)
                    elif step_type == "no_results":
                        st.markdown(f'<div class="trace-step" style="border-color:#F59E0B">⚠️ <b>{step.get("detail", "")}</b></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="trace-step">ℹ️ {step.get("detail", step_type)}{time_str}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# 5. User Chat Input & Agent Execution
# ═══════════════════════════════════════════════
if prompt := st.chat_input("e.g., Are there any high-risk indemnification clauses?"):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt, "trace": None, "latency": None, "risk": None})
    chat_history.save_message(st.session_state.session_id, "user", prompt)
    
    # Generate LLM title for fresh sessions (only user messages, only if title is 'New Chat' or first msg)
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if len(user_msgs) == 1:
        title = chat_history.generate_title(prompt)
        chat_history.update_title(st.session_state.session_id, title)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Start metrics collection
    collector = st.session_state.collector
    pipeline_label = pipeline_choice
    metrics = collector.start(prompt, pipeline_label)

    # Execute agent with error handling
    with st.chat_message("assistant"):
        with st.spinner(f"🔍 Analyzing with **{pipeline_choice}**..."):
            try:
                cached_clauses = st.session_state.get("local_extracted_clauses")
                result = run_baseline_agent(prompt, local_context=st.session_state.local_context, pre_extracted_clauses=cached_clauses)

                response_text = result["response"]
                trace = result["trace"]
                tool_calls = result["tool_calls"]
                retrieval_count = result["retrieval_count"]
                risk_level = result["risk_level"]
                success = result["success"]

            except Exception as e:
                response_text = f"⚠️ **An error occurred:** {str(e)}\n\nPlease check your API keys and network connection, then try again."
                trace = [{"step": "error", "detail": str(e), "time": 0}]
                tool_calls = []
                retrieval_count = 0
                risk_level = "N/A"
                success = False

        # Finalize metrics
        collector.finish(metrics, success=success, tool_calls=tool_calls,
                        retrieval_count=retrieval_count, risk_level=risk_level)

        # Display response
        st.markdown(response_text)

        # Inline latency + risk badge
        latency = metrics.latency_s
        risk_class = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low" if risk_level == "Low" else "badge-blue"
        st.markdown(f"""
            <span class="latency-tag">⏱ {latency}s</span>
            <span class="{risk_class}" style="font-size:0.75rem">{risk_level} Risk</span>
        """, unsafe_allow_html=True)

        # Debug trace expander
        if trace:
            with st.expander("🔍 Execution Trace & Debug Log"):
                for step in trace:
                    step_type = step.get("step", "")
                    step_time = step.get("time", "")
                    time_str = f" — {step_time}s" if step_time else ""

                    if step_type == "start":
                        st.markdown(f'<div class="trace-step">📝 <b>Query Received:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "tool_call":
                        tool = step.get("tool", "unknown")
                        preview = step.get("result_preview", "")
                        st.markdown(f'<div class="trace-step">🛠️ <b>Tool Call:</b> <code>{tool}</code>{time_str}<br><small style="color:#94A3B8">{preview}</small></div>', unsafe_allow_html=True)
                    elif step_type == "model_inference":
                        st.markdown(f'<div class="trace-step">🤖 <b>Model Inference:</b> {step.get("detail", "")}{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "response":
                        st.markdown(f'<div class="trace-step">✅ <b>Response Generated</b>{time_str}</div>', unsafe_allow_html=True)
                    elif step_type == "error":
                        st.markdown(f'<div class="trace-step" style="border-color:#EF4444">❌ <b>Error:</b> {step.get("detail", "")}</div>', unsafe_allow_html=True)
                    elif step_type == "greeting_filter":
                        st.markdown(f'<div class="trace-step">👋 <b>Greeting Detected</b> — skipped pipeline</div>', unsafe_allow_html=True)
                    elif step_type == "no_results":
                        st.markdown(f'<div class="trace-step" style="border-color:#F59E0B">⚠️ <b>{step.get("detail", "")}</b></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="trace-step">ℹ️ {step.get("detail", step_type)}{time_str}</div>', unsafe_allow_html=True)

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "trace": trace,
        "latency": latency,
        "risk": risk_level
    })
    chat_history.save_message(st.session_state.session_id, "assistant", response_text, risk_level=risk_level)

    # Add to query history
    st.session_state.query_history.append({
        "query": prompt,
        "latency": latency,
        "risk": risk_level,
        "pipeline": pipeline_choice
    })

    st.rerun()