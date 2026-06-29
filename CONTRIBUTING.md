# Contributing to TeleGenie AI

Thank you for your interest in making TeleGenie AI better. TeleGenie is a
multi-agent Retrieval-Augmented Generation system for telecom operations, built
as the IIT Roorkee × Futurense Technologies capstone project. Bug reports,
feature ideas, documentation fixes, and pull requests are all welcome.

## Bug reports

Open a [GitHub Issue](../../issues) and include:

- **Repro steps** — the exact query, PDF, or admin action that triggers the bug
- **Expected behavior** — what should have happened
- **Observed behavior** — what actually happened (include stack traces, screenshot, or audit-log line)
- **Environment** — `python --version`, `docker --version`, OS, commit SHA, deployment mode (local Docker / bare Python / AWS App Runner)

## Feature requests

Open a thread in [GitHub Discussions](../../discussions) under the *Ideas*
category. Describe the telecom-ops workflow it unblocks, sketch the proposed
user-facing behavior, and link any relevant 3GPP or vendor specs that justify
the change.

## Pull requests

### Branch from `main`

Fork the repo and create a feature branch off the latest `main`.

### Branch naming

| Prefix | Use for |
|---|---|
| `feat/<short-name>` | New user-facing capability |
| `fix/<short-name>` | Bug fix |
| `docs/<short-name>` | Documentation-only change |

Examples: `feat/ingest-citations-table`, `fix/admin-create-user-username-normalization`.

### Commit messages — Conventional Commits

Use one of these prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `chore:` — tooling, deps, or non-functional cleanup
- `docs:` — documentation only
- `refactor:` — code change that neither fixes a bug nor adds a feature

Keep the subject under 72 characters and the body focused on *why*.

### Pre-commit checklist

Before opening the PR, run:

```powershell
# Compile-check the app and the RAG library
python -m compileall app telecom_rag

# If you touched CloudFormation:
cfn-lint cloudformation/template.yaml

# If your change is user-facing, update README.md
```

### PR description must include

1. **Summary** — one-paragraph description of the change
2. **Motivation** — why the change is needed (link the issue / discussion)
3. **Test plan** — exact steps you ran locally to verify
4. **Screenshots** — required for any UI / Streamlit page change

## Local dev setup

Three commands to a working local stack:

```powershell
# 1. Clone
git clone git@github.com:RajuforAI/IIT-Roorkee_Capstone.git
cd IIT-Roorkee_Capstone

# 2. Copy the env template and fill in OPENAI_API_KEY
Copy-Item .env.example .env

# 3. Build + start all services
docker compose up -d --build
```

The Streamlit UI auto-opens at `http://localhost:8501`.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).