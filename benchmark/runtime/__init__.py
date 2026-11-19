"""Session execution and result audit logging."""

from .. import _register_module_aliases

MODULE_ALIASES = {
    "simplenav_launch": "benchmark.launch.simplenav",
    "sequence_runs": "benchmark.launch.sequences",
    "task_repair": "benchmark.legacy.task_repair",
    "task_history": "benchmark.legacy.task_history",
    "repair_audit": "benchmark.reporting.legacy",
}

_register_module_aliases(__name__, MODULE_ALIASES)
