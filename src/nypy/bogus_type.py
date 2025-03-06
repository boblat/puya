"""A Bogus[T] type alias for marking when we subvert the type system

We need this for compiling with mypyc, which inserts runtime
typechecks that cause problems when we subvert the type system. So
when compiling with mypyc, we turn those places into Any, while
keeping the types around for normal typechecks.

Since this causes the runtime types to be Any, this is best used
in places where efficient access to properties is not important.
For those cases some other technique should be used.
"""

from __future__ import annotations

from typing import TypeVar

from mypy_extensions import FlexibleAlias

T = TypeVar("T")

Bogus = FlexibleAlias[T, T]
