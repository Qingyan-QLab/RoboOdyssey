"""Benchmark-owned frozen-data runtime contracts; no generator dependency."""

from .. import _register_module_aliases

MODULE_ALIASES = {
    "history": "roboodyssey.contracts.language",
    "calendar": "roboodyssey.contracts.language",
    "wording": "roboodyssey.contracts.language",
    "naming": "roboodyssey.contracts.language",
    "catalog": "roboodyssey.contracts.language",
    "object_references": "roboodyssey.contracts.references",
    "request_references": "roboodyssey.contracts.references",
    "labels": "roboodyssey.contracts.navigation",
    "routes": "roboodyssey.contracts.navigation",
    "coverage": "roboodyssey.contracts.navigation",
    "mutations": "roboodyssey.contracts.navigation",
    "legacy_audit": "roboodyssey.contracts.legacy_templates",
}

_register_module_aliases(__name__, MODULE_ALIASES)
