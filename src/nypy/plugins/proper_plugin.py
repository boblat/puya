"""
This plugin is helpful for mypy development itself.
By default, it is not enabled for mypy users.

It also can be used by plugin developers as a part of their CI checks.

It finds missing ``get_proper_type()`` call, which can lead to multiple errors.
"""

from __future__ import annotations

from typing import Callable

from nypy.checker import TypeChecker
from nypy.nodes import TypeInfo
from nypy.plugin import FunctionContext, Plugin
from nypy.subtypes import is_proper_subtype
from nypy.types import (
    AnyType,
    FunctionLike,
    Instance,
    NoneTyp,
    ProperType,
    TupleType,
    Type,
    UnionType,
    get_proper_type,
    get_proper_types,
)


class ProperTypePlugin(Plugin):
    """
    A plugin to ensure that every type is expanded before doing any special-casing.

    This solves the problem that we have hundreds of call sites like:

        if isinstance(typ, UnionType):
            ...  # special-case union

    But after introducing a new type TypeAliasType (and removing immediate expansion)
    all these became dangerous because typ may be e.g. an alias to union.
    """

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if fullname == "builtins.isinstance":
            return isinstance_proper_hook
        if fullname == "nypy.types.get_proper_type":
            return proper_type_hook
        if fullname == "nypy.types.get_proper_types":
            return proper_types_hook
        return None


def isinstance_proper_hook(ctx: FunctionContext) -> Type:
    if len(ctx.arg_types) != 2 or not ctx.arg_types[1]:
        return ctx.default_return_type

    right = get_proper_type(ctx.arg_types[1][0])
    for arg in ctx.arg_types[0]:
        if (
            is_improper_type(arg) or isinstance(get_proper_type(arg), AnyType)
        ) and is_dangerous_target(right):
            if is_special_target(right):
                return ctx.default_return_type
            ctx.api.fail(
                "Never apply isinstance() to unexpanded types;"
                " use nypy.types.get_proper_type() first",
                ctx.context,
            )
            ctx.api.note(  # type: ignore[attr-defined]
                "If you pass on the original type"
                " after the check, always use its unexpanded version",
                ctx.context,
            )
    return ctx.default_return_type


def is_special_target(right: ProperType) -> bool:
    """Whitelist some special cases for use in isinstance() with improper types."""
    if isinstance(right, FunctionLike) and right.is_type_obj():
        if right.type_object().fullname == "builtins.tuple":
            # Used with Union[Type, Tuple[Type, ...]].
            return True
        if right.type_object().fullname in (
            "nypy.types.Type",
            "nypy.types.ProperType",
            "nypy.types.TypeAliasType",
        ):
            # Special case: things like assert isinstance(typ, ProperType) are always OK.
            return True
        if right.type_object().fullname in (
            "nypy.types.UnboundType",
            "nypy.types.TypeVarLikeType",
            "nypy.types.TypeVarType",
            "nypy.types.UnpackType",
            "nypy.types.TypeVarTupleType",
            "nypy.types.ParamSpecType",
            "nypy.types.Parameters",
            "nypy.types.RawExpressionType",
            "nypy.types.EllipsisType",
            "nypy.types.StarType",
            "nypy.types.TypeList",
            "nypy.types.CallableArgument",
            "nypy.types.PartialType",
            "nypy.types.ErasedType",
            "nypy.types.DeletedType",
            "nypy.types.RequiredType",
            "nypy.types.ReadOnlyType",
        ):
            # Special case: these are not valid targets for a type alias and thus safe.
            # TODO: introduce a SyntheticType base to simplify this?
            return True
    elif isinstance(right, TupleType):
        return all(is_special_target(t) for t in get_proper_types(right.items))
    return False


def is_improper_type(typ: Type) -> bool:
    """Is this a type that is not a subtype of ProperType?"""
    typ = get_proper_type(typ)
    if isinstance(typ, Instance):
        info = typ.type
        return info.has_base("nypy.types.Type") and not info.has_base("nypy.types.ProperType")
    if isinstance(typ, UnionType):
        return any(is_improper_type(t) for t in typ.items)
    return False


def is_dangerous_target(typ: ProperType) -> bool:
    """Is this a dangerous target (right argument) for an isinstance() check?"""
    if isinstance(typ, TupleType):
        return any(is_dangerous_target(get_proper_type(t)) for t in typ.items)
    if isinstance(typ, FunctionLike) and typ.is_type_obj():
        return typ.type_object().has_base("nypy.types.Type")
    return False


def proper_type_hook(ctx: FunctionContext) -> Type:
    """Check if this get_proper_type() call is not redundant."""
    arg_types = ctx.arg_types[0]
    if arg_types:
        arg_type = get_proper_type(arg_types[0])
        proper_type = get_proper_type_instance(ctx)
        if is_proper_subtype(arg_type, UnionType.make_union([NoneTyp(), proper_type])):
            # Minimize amount of spurious errors from overload machinery.
            # TODO: call the hook on the overload as a whole?
            if isinstance(arg_type, (UnionType, Instance)):
                ctx.api.fail("Redundant call to get_proper_type()", ctx.context)
    return ctx.default_return_type


def proper_types_hook(ctx: FunctionContext) -> Type:
    """Check if this get_proper_types() call is not redundant."""
    arg_types = ctx.arg_types[0]
    if arg_types:
        arg_type = arg_types[0]
        proper_type = get_proper_type_instance(ctx)
        item_type = UnionType.make_union([NoneTyp(), proper_type])
        ok_type = ctx.api.named_generic_type("typing.Iterable", [item_type])
        if is_proper_subtype(arg_type, ok_type):
            ctx.api.fail("Redundant call to get_proper_types()", ctx.context)
    return ctx.default_return_type


def get_proper_type_instance(ctx: FunctionContext) -> Instance:
    checker = ctx.api
    assert isinstance(checker, TypeChecker)
    types = checker.modules["nypy.types"]
    proper_type_info = types.names["ProperType"]
    assert isinstance(proper_type_info.node, TypeInfo)
    return Instance(proper_type_info.node, [])


def plugin(version: str) -> type[ProperTypePlugin]:
    return ProperTypePlugin
