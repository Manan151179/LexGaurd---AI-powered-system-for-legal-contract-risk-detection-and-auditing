# Team Contribution Report — Labs 1–9 (Through Phase 3)
**Project:** LexGuard — Neuro-Symbolic Compliance Auditor for Contract Risk Analysis
**Course:** CS 5542 — Big Data Analytics & Applications, UMKC, Spring 2026
**GitHub:** https://github.com/JoeDoan/Lab9_BigData

---

## Phase 3 — Team Contribution Table

| Team Member | Role | Phase 3 Contributions | % |
|---|---|---|---|
| **Joe Doan** | Data Pipeline & Adaptation Lead | BERT vs LLM evaluation (`evaluate_e2e.py`), full-doc LLM extraction pipeline (`extract_risk_clauses_llm`, `extract_contract_brief`), Snowflake chat persistence (`chat_history.py`), dark/light theme toggle, chat history UI with delete & LLM-generated titles, Phase 3 report | 30% |
| **Manan Koradiya** | Agent Architect & Integrator | `app.py` UI redesign (glassmorphism CSS, chat interface), RAG fallback enhancement (`tools.py`), end-to-end system integration, reasoning panels and query history sidebar | 25% |
| **Aditya Naredla** | Storage & Evaluation Engineer | PEFT training notebook (`LexGuard_PEFT_Training.ipynb`), `monitor.py` module, live analytics dashboard, HuggingFace Hub adapter upload | 25% |
| **Ruixuan Hou** | Reproducibility Lead | `requirements.txt`, `Dockerfile`, `.streamlit/config.toml`, `reproduce.sh`, `REPRO_AUDIT.md`, `RUN.md` setup instructions, system status panel | 20% |
| **Total** | | | **100%** |

---

## Phase 3 — Key Technical Decisions

### 1. BERT → Full-Document LLM Extraction
- Fine-tuned BERT QA model (`doandune/LexGuard-CUAD-BERT`) achieved only **53.8% accuracy with ~0% recall** on 12 risk clause types.
- Root cause: BERT's 512-token window misses clauses spanning multiple paragraphs.
- **Decision:** Replaced with Gemini 2.5 Flash full-document extraction (**86.3% accuracy**), passing up to 200K characters directly to the LLM.

### 2. Chunking + RAG → Direct Full-Document Input
- Evaluated hybrid retrieval (FAISS + BM25 + cross-encoder reranking) with document chunking.
- Chunking fragmented important clause context, lowering extraction accuracy.
- **Decision:** Production pipeline now feeds the entire document directly to Gemini, leveraging its 1M-token context window.

### 3. Snowflake Chat Persistence
- Added `CHAT_SESSIONS` and `CHAT_MESSAGES` tables with annotation metadata serialization (JSON).
- LLM-generated session titles, delete functionality, and full session restore including expandable source annotations.

---

## System Architecture (Phase 3 Production)

```
User (Streamlit UI — Dark/Light Theme)
        ↓
  [File Upload: PDF/TXT]
        ↓
  PyMuPDF Text Extraction (Full Document)
        ↓
  ┌─────────────────────────────────────┐
  │ PRIMARY PATH: Full-Doc LLM         │
  │   • Risk Audit (200K chars → Gemini)│
  │   • Metadata Brief (8 entities)     │
  │   • General Q&A (50K chars)         │
  └─────────────────────────────────────┘
        ↓
  Gemini 2.5 Flash Response
  + Expandable Source Annotations
        ↓
  Snowflake Persistence
  (CHAT_SESSIONS + CHAT_MESSAGES + METADATA)
```

---

## Lab 9 — Team Contribution Table

| Team Member | Role | Lab 9 Contributions | % |
|---|---|---|---|
| **Joe Doan** | Data Pipeline & Adaptation Lead | Structured execution traces in `agent.py` and `adapted_agent.py`, timed tool calls, trace-based debug logging, `LAB9_REPORT.md` | 30% |
| **Manan Koradiya** | Agent Architect & Integrator | Complete `app.py` UI redesign with premium dark theme, glassmorphism CSS, chat interface, reasoning panels, query history sidebar, error handling | 25% |
| **Aditya Naredla** | Storage & Evaluation Engineer | `monitor.py` module (`QueryMetrics` + `MetricsCollector`), live analytics dashboard in sidebar, per-pipeline latency comparison | 25% |
| **Ruixuan Hou** | Reproducibility Lead | `requirements.txt`, `.streamlit/config.toml`, `Dockerfile`, deployment configuration, system status panel | 20% |
| **Total** | | | **100%** |

---

## Lab 8 — Team Contribution Table

| Team Member | Role | Lab 8 Contributions | % |
|---|---|---|---|
| **Joe Doan** | Data Pipeline & Adaptation Lead | Instruction dataset generation (`generate_dataset.py`), `adapted_agent.py` full pipeline, Colab FastAPI server debugging, prompt format fix, response parsing, `EVALUATION.md` | 30% |
| **Manan Koradiya** | Agent Architect & Integrator | Streamlit baseline vs. adapted toggle (`app.py`), RAG fallback enhancement (`tools.py`), end-to-end system integration | 25% |
| **Aditya Naredla** | Storage & Evaluation Engineer | Domain task definition, model selection (Llama-3), PEFT training notebook (`LexGuard_PEFT_Training.ipynb`), HuggingFace Hub adapter upload, evaluation design | 25% |
| **Ruixuan Hou** | Reproducibility Lead | `reproduce.sh` Lab 8 updates, new smoke tests for adapted pipeline, `REPRO_AUDIT.md` non-determinism documentation, `RUN.md` setup instructions | 20% |
| **Total** | | | **100%** |

---

## Deliverables Summary

| Deliverable | File | Status |
|---|---|---|
| Phase 3 Report | `Phase_3_Report_LexGuard.docx` | ✅ Complete |
| Full-Doc LLM Extraction | `tools.py` (`extract_risk_clauses_llm`, `extract_contract_brief`) | ✅ Production |
| Chat Persistence | `chat_history.py` | ✅ Snowflake-backed |
| Dark/Light Theme | `app.py` (CSS variables + toggle) | ✅ Deployed |
| BERT Evaluation | `evaluate_e2e.py` | ✅ 53.8% → deprecated |
| Premium Streamlit UI | `app.py` | ✅ Dark theme + glassmorphism |
| Monitoring Module | `monitor.py` | ✅ QueryMetrics + Analytics |
| Structured Traces | `agent.py`, `adapted_agent.py` | ✅ Timed tool calls |
| Deployment Config | `Dockerfile`, `.streamlit/config.toml` | ✅ Docker + Theme |
| Dependencies | `requirements.txt` | ✅ Pinned versions |
| Development Report | `LAB9_REPORT.md` | ✅ Complete |
| Individual Reports | `CONTRIBUTION_*.md` | ✅ All 4 members |
