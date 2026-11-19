"""Reporting interfaces."""

from .. import _register_module_aliases

MODULE_ALIASES = {
    "legacy_repair": "roboodyssey.reporting.legacy",
}

_register_module_aliases(__name__, MODULE_ALIASES)
