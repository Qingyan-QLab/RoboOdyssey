"""Session execution and result audit logging."""

from .. import _register_module_aliases

MODULE_ALIASES = {
    "simplenav_launch": "roboodyssey.launch.simplenav",
    "sequence_runs": "roboodyssey.launch.sequences",
    "task_repair": "roboodyssey.legacy.task_repair",
    "task_history": "roboodyssey.legacy.task_history",
    "repair_audit": "roboodyssey.reporting.legacy",
}

_register_module_aliases(__name__, MODULE_ALIASES)
