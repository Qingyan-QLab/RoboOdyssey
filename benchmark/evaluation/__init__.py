"""Task success checks, metrics, judges, and report aggregation."""

from .. import _register_module_aliases

MODULE_ALIASES = {
    "accounting": "benchmark.evaluation.report",
    "request_report": "benchmark.evaluation.report",
    "request_audit": "benchmark.reporting.request_audit",
    "task_diagnostics_report": "benchmark.reporting.task_diagnostics_report",
}

_register_module_aliases(__name__, MODULE_ALIASES)
