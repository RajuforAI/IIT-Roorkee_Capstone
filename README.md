# TeleGenie AI

> **Multi-Agent Retrieval-Augmented Generation for Telecom Operations**
> IIT Roorkee × Futurense Technologies — Capstone Project (Theme 5)
> Version 1.0.0 · June 2026

[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-009688)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![AWS App Runner](https://img.shields.io/badge/AWS-App_Runner-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/apprunner/)

[![CI](https://github.com/RajuforAI/IIT-Roorkee_Capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/RajuforAI/IIT-Roorkee_Capstone/actions/workflows/ci.yml)

---

## 1. Product Overview

**TeleGenie AI** is a multi-agent Retrieval-Augmented Generation (RAG) application purpose-built for **telecom operations teams**. It enables engineers, field technicians, NOC analysts, and operations managers to query a private corpus of telecom documents — 3GPP specs, vendor manuals, internal SOPs, calibration guides, troubleshooting runbooks — using natural language, and receive **cited, validated answers in real time**.

TeleGenie is **not** a general-purpose chatbot. It is deliberately scoped to the telecom domain: out-of-domain queries are detected and refused with a transparent explanation, and every answer carries source attribution (document name + page number) so engineers can verify before acting.

### What problem it solves

Telecom operations teams manage **thousands of pages** of documentation. During a live network incident — a 5G NR cell drop, a fiber cut, a calibration anomaly — finding the correct procedure inside a 400-page vendor manual takes too long. TeleGenie collapses that search into a single chat query, grounded in the actual documents, with the citations to back it up.

### Who it's for

| Persona | Use case | SLA expectation |
|---|---|---|
| **NOC Engineer** (primary) | Find correct procedure for a live incident in under 2 minutes | < 5 s response, cited |
| **Field Technician** | Look up antenna azimuth / fiber splice steps on a mobile browser | < 3 s load, formatted |
| **Network Architect** | Cross-vendor procedure comparison + multi-doc synthesis | < 10 s, comprehensive |
| **Ops Manager** (admin) | Monitor query quality, upload new PDFs, review feedback | Admin panel < 2 s |

---

## 2. Core Capabilities

1. **Answer factual telecom questions** using retrieved document context (Q&A mode)
2. **Summarize single or multiple telecom documents** on demand (Summarize mode)
3. **Validate** whether a technical statement is supported by documents (Validate mode)
4. **Refuse and explain** out-of-domain queries (Refuse mode)
5. **Ingest new PDFs** and make them immediately queryable
6. **Persist conversation history** per user thread across sessions
7. **Source-cite every answer** (document + page number)
8. **Multi-provider LLM** — OpenAI primary, Gemini fallback when quota/network fails
9. **Structured JSONL audit logs** for compliance and debugging
10. **Admin panel** for ingestion monitoring, user management, and cost telemetry

---

## 3. Architecture at a Glance

```
┌─────────────┐    ┌──────────────────────────────────────────┐    ┌─────────────┐
│   User UI   │───▶│       Streamlit Chat Page (app/main.py) │───▶│  LangGraph  │
│  (Browser)  │    │   + auth gate + admin pages              │    │  StateGraph │
└─────────────┘    └──────────────────────────────────────────┘    └──────┬──────┘
                                                                           │
                       ┌───────────────────────────────────────────────────┤
                       ▼                              ▼                    ▼
              ┌─────────────────┐          ┌──────────────────┐   ┌─────────────────┐
              │   ChromaDB      │          │   LLM Providers  │   │  LangGraph      │
              │   (vector store)│          │  OpenAI / Gemini │   │  Checkpoint DB  │
              └─────────────────┘          └──────────────────┘   │  (SQLite/PG)    │
                                                                  └─────────────────┘
                       ▲
                       │
              ┌────────┴────────┐
              │  PDF Ingestion  │
              │  (pypdf +       │
              │  LangChain      │
              │  splitters)     │
              └─────────────────┘
```

| Layer | Technology | Version |
|---|---|---|
| Chat UI | Streamlit | 1.39 |
| Agent orchestration | LangGraph | 1.2.6 |
| LLM framework | LangChain | 1.3.11 |
| Vector store | ChromaDB | 1.5.9 |
| LLM providers | OpenAI `gpt-4o-mini` + Google Gemini | latest API |
| PDF ingestion | pypdf + LangChain text splitters | 6.14.1 / 1.1.2 |
| Auth gate | streamlit-authenticator + bcrypt | 0.3.2 / 5.0.0 |
| Observability | LangSmith tracing + custom JSONL audit log | 0.9.0 |
| Evaluation harness | RAGAS | 0.2.6 |

The full multi-agent StateGraph contains **Planner → Retrieval → Grader → Generator → Validator** nodes, each routed through LangGraph's conditional edges. See `telecom_rag/graphs/` for the implementation.

---

## 4. Prerequisites

Install these on your local machine **before** cloning the repo.

### Required

| Tool | Version | Why | Install |
|---|---|---|---|
| **Python** | 3.11.x | Pinned — `langgraph 1.2.6` + `langchain 1.3.x` require it | https://www.python.org/downloads/ |
| **Git** | 2.30+ | Clone + push | https://git-scm.com/downloads |
| **Docker Desktop** | 4.x (with Compose v2) | Local containerized run + production parity | https://www.docker.com/products/docker-desktop/ |

### For cloud deployment only

| Tool | Version | Why | Install |
|---|---|---|---|
| **AWS CLI** | 2.x | Push to AWS, manage Secrets Manager | https://awscli.amazonaws.com/AWSCLIV2.msi |
| **AWS account** | — | App Runner + S3 + Secrets Manager + CloudWatch | https://aws.amazon.com/console/ |

### For LLM access

| Credential | Where to get it |
|---|---|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `GEMINI_API_KEY` (optional, fallback) | https://aistudio.google.com/apikey |
| `LANGCHAIN_API_KEY` (optional, for tracing) | https://smith.langchain.com → Settings → API Keys |

> ⚠️ **Never commit API keys.** Copy `.env.example` → `.env` and fill the values locally. The `.env` file is git-ignored.

---

## 5. Local Quickstart

### Option A — Docker Compose (recommended, mirrors production)

```powershell
# 1. Clone the repo
git clone git@github.com:RajuforAI/IIT-Roorkee_Capstone.git
cd IIT-Roorkee_Capstone

# 2. Create your local env file
Copy-Item .env.example .env
# Edit .env and fill in OPENAI_API_KEY (and any other keys you have)

# 3. Build + start all three services (app + Chroma + Postgres)
docker compose up -d --build

# 4. Open the Streamlit UI
start http://localhost:8501
```

The first build takes ~5–8 minutes (multi-stage pip install). Subsequent starts take ~10 seconds.

### Option B — Bare Python (faster iteration, no Chroma persistence)

```powershell
git clone git@github.com:RajuforAI/IIT-Roorkee_Capstone.git
cd IIT-Roorkee_Capstone

Copy-Item .env.example .env
# Fill in OPENAI_API_KEY in .env

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0
```

The Streamlit UI auto-opens at `http://localhost:8501`.

### Verify the install

```powershell
# Container health (Docker path)
docker compose ps            # all 3 services "healthy"

# Inside the container
docker compose exec app python -c "from telecom_rag.config import settings; print('model=', settings.llm_model)"
# → model= gpt-4o-mini
```

---

## 6. Project Structure

```
IIT-Roorkee_Capstone/
├── app/                          # Streamlit chat UI
│   ├── main.py                   # auth gate + chat page entrypoint
│   ├── components/               # reusable UI components
│   └── pages/                    # admin / upload / my_uploads pages
├── telecom_rag/                  # RAG library
│   ├── agents/                   # Planner, Retrieval, Generator, Validator
│   ├── auth/                     # bcrypt auth + SQLite user store
│   ├── eval/                     # RAGAS evaluation harness
│   ├── graphs/                   # LangGraph StateGraph definitions
│   ├── ingestion/                # PDF loader, chunker, embedder
│   ├── memory/                   # LangGraph checkpointers (SQLite / Postgres)
│   ├── observability/            # /healthz, JSONL audit log
│   ├── security/                 # input validators, PII redaction
│   ├── storage/                  # S3 upload helper
│   ├── tools/                    # retrieval tool wrappers
│   ├── config.py                 # Pydantic settings (env-prefixed)
│   ├── llm.py                    # multi-provider LLM layer + retry
│   └── schemas.py                # typed contracts
├── Dockerfile                    # multi-stage production image
├── docker-compose.yml            # local: app + Chroma + Postgres
├── requirements.txt              # pinned Python deps
├── .env.example                  # env template (copy → .env)
├── .dockerignore
├── .gitignore
├── README.md                     # ← you are here
└── AWS_DEPLOYMENT.md             # step-by-step AWS push guide
```

---

## 7. Post-Requisites (after first run / first deploy)

A short checklist of things to do **once the app is up**:

- [ ] **Create the first admin user.** On a fresh database, set `TELECOM_RAG_BOOTSTRAP_ADMIN_PASSWORD` in `.env`, restart the app, then log in at `/admin` with username `admin` and that password. **Rotate this immediately** from the User Management panel.
- [ ] **Verify the `/healthz` endpoint.** `curl http://localhost:8501/_healthz` (or your domain) should return `{"status":"ok"}`. This is the AWS / load-balancer liveness probe.
- [ ] **Upload at least one PDF.** Use the admin Upload page to ingest your first telecom SOP. Confirm it appears in the chat results within 30 seconds.
- [ ] **Check the LangSmith trace.** Open https://smith.langchain.com → project "TeleGenie AI" → confirm traces are landing.
- [ ] **Review the cost telemetry.** Visit `/admin` → Cost tab; confirm the daily USD cap is appropriate for your OpenAI plan.
- [ ] **Set up automated backups** for the Chroma persistence directory (`chroma_db/`) and the auth DB (`auth.db`). On AWS, use S3 lifecycle policies.
- [ ] **Rotate the `TELECOM_RAG_SECRET_KEY`** in `.env` to a 32-byte random hex value (`python -c "import secrets; print(secrets.token_hex(32))"`).
- [ ] **Configure CORS / allowed origins** if you're hosting the Streamlit app behind a custom domain.
- [ ] **Enable CloudWatch log shipping** from the container to your central observability stack (the JSONL audit log in `telecom_rag/observability/` is the source of truth).
- [ ] **Subscribe to AWS billing alerts** at 50% and 80% of your monthly cap.

---

## 8. Security Notes

- The default admin password bootstrap (`TELECOM_RAG_BOOTSTRAP_ADMIN_PASSWORD=replace-me`) is **rejected at startup** — you must set a real value.
- API keys are read from environment variables only; **never** from files committed to the repo.
- The Streamlit auth gate (`streamlit-authenticator` + bcrypt cost factor 12) hashes passwords; nothing is stored in plaintext.
- Out-of-domain query detection prevents prompt-injection style "ignore previous instructions" abuse.
- The container runs as a **non-root user** (UID 10001) in the production image.

---

## 9. License

MIT — see `LICENSE` for the full text.

---

## 10. About

This project was built as the capstone deliverable for the **IIT Roorkee × Futurense Technologies AI Engineering program (Theme 5: Applied RAG for Industry)** by **Raju Bera** in June 2026.

The production codebase lives at https://github.com/RajuforAI/IIT-Roorkee_Capstone.

For deployment instructions, see [`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md).