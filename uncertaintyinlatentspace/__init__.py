# coding: utf-8
"""Main module of uncertaintyinlatentspace."""

try:
    from .version import __version__, __version_date__  # pylint: disable=import-error
except ImportError:
    __version__ = None
    __version_date__ = ""
