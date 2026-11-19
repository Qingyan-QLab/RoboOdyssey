"""RoboOdyssey simulation and evaluation package."""

from importlib import import_module
import sys


def _register_module_aliases(package, aliases):
    """Keep historical imports bound to the single canonical implementation."""
    parent = sys.modules[package]
    for alias, target in aliases.items():
        module = import_module(target)
        sys.modules[package + "." + alias] = module
        setattr(parent, alias, module)
