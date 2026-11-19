"""Task success checks, metrics, judges, and report aggregation."""

from .. import _register_module_aliases

MODULE_ALIASES = {
    "accounting": "roboodyssey.evaluation.report",
    "request_report": "roboodyssey.evaluation.report",
    "request_audit": "roboodyssey.reporting.request_audit",
    "task_diagnostics_report": "roboodyssey.reporting.task_diagnostics_report",
}

_register_module_aliases(__name__, MODULE_ALIASES)
