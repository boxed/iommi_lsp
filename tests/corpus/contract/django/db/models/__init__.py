"""Skeleton stand-in for django.db.models. Just enough surface for the
contract fixture to type-check without django-stubs installed.

This is intentionally not a faithful Django reproduction — the point is
to exercise the false-positive shape that ``ty`` produces against a
realistic-looking import graph.
"""


class Model:
    pass


class CharField:
    def __init__(self, *args, **kwargs): ...


class ForeignKey:
    def __init__(self, *args, **kwargs): ...


class CASCADE: ...
