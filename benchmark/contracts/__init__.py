"""Benchmark-owned frozen-data runtime contracts; no generator dependency."""

from .. import _register_module_aliases

MODULE_ALIASES = {
    "history": "benchmark.contracts.language",
    "calendar": "benchmark.contracts.language",
    "wording": "benchmark.contracts.language",
    "naming": "benchmark.contracts.language",
    "catalog": "benchmark.contracts.language",
    "object_references": "benchmark.contracts.references",
    "request_references": "benchmark.contracts.references",
    "labels": "benchmark.contracts.navigation",
    "routes": "benchmark.contracts.navigation",
    "coverage": "benchmark.contracts.navigation",
    "mutations": "benchmark.contracts.navigation",
    "legacy_audit": "benchmark.contracts.legacy_templates",
}

_register_module_aliases(__name__, MODULE_ALIASES)
