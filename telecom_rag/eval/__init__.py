"""RAGAS-based offline evaluation harness (Issue #9).

Public surface:

- :mod:`telecom_rag.eval.dataset` — golden Q&A loader + schema validation.
- :mod:`telecom_rag.eval._pearson` — Pearson correlation helper.
- :mod:`telecom_rag.eval.evaluator` — :class:`TelecomRAGEvaluator`,
  :class:`EvaluationReport`, :class:`PerQueryResult`.
- :mod:`telecom_rag.eval.ragas_eval` — CLI entry point
  (``python -m telecom_rag.eval.ragas_eval``).

The package is offline / on-demand only; CI integration is deferred
to follow-up Issue #10.
"""
