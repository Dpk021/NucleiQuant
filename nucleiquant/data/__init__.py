"""Data utilities.

``NucleiDataset`` / ``make_targets`` need torch, so they are imported lazily:
``prepare_monuseg`` and the XML parser work in a plain numpy/OpenCV environment.
"""
from .monuseg import parse_aperio_xml, xml_to_instance_mask  # noqa: F401

__all__ = ["parse_aperio_xml", "xml_to_instance_mask", "NucleiDataset", "make_targets"]


def __getattr__(name):
    if name in ("NucleiDataset", "make_targets"):
        from . import dataset
        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
