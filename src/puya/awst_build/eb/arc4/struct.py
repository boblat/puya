from __future__ import annotations

import typing

import structlog

from puya.awst import wtypes
from puya.awst.nodes import (
    ARC4Encode,
    Expression,
    IndexExpression,
    Literal,
    TupleExpression,
    UInt64Constant,
)
from puya.awst_build.eb._utils import bool_eval_to_constant
from puya.awst_build.eb.arc4._utils import expect_arc4_operand_wtype
from puya.awst_build.eb.arc4.base import CopyBuilder, arc4_compare_bytes, get_bytes_expr_builder
from puya.awst_build.eb.base import BuilderComparisonOp, ValueExpressionBuilder
from puya.awst_build.eb.bytes_backed import BytesBackedClassExpressionBuilder
from puya.awst_build.eb.var_factory import var_expression
from puya.awst_build.utils import get_arg_mapping
from puya.errors import CodeError

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    import mypy.nodes

    from puya.awst_build.eb.base import (
        ExpressionBuilder,
    )
    from puya.parse import SourceLocation

logger: structlog.types.FilteringBoundLogger = structlog.get_logger(__name__)


class ARC4StructClassExpressionBuilder(BytesBackedClassExpressionBuilder):
    def produces(self) -> wtypes.WType:
        return self.wtype

    def __init__(
        self,
        wtype: wtypes.ARC4Struct,
        location: SourceLocation,
    ):
        super().__init__(location)
        self.wtype = wtype

    def call(
        self,
        args: Sequence[ExpressionBuilder | Literal],
        arg_kinds: list[mypy.nodes.ArgKind],
        arg_names: list[str | None],
        location: SourceLocation,
    ) -> ExpressionBuilder:
        ordered_field_names = self.wtype.names
        field_mapping = get_arg_mapping(
            positional_arg_names=ordered_field_names,
            args=zip(arg_names, args, strict=True),
            location=location,
        )

        args_positioned = list[Expression]()
        for field_name, field_type in self.wtype.fields.items():
            field_value = field_mapping.pop(field_name, None)
            if field_value is None:
                raise CodeError(f"Missing required argument {field_name}", location)
            args_positioned.append(expect_arc4_operand_wtype(field_value, field_type))
        if field_mapping:
            raise CodeError(f"Unexpected keyword arguments: {' '.join(field_mapping)}", location)

        tuple_expr = TupleExpression(
            items=args_positioned,
            source_location=location,
            wtype=wtypes.WTuple.from_types(types=self.wtype.types),
        )

        return var_expression(
            ARC4Encode(wtype=self.wtype, value=tuple_expr, source_location=location)
        )


class ARC4StructExpressionBuilder(ValueExpressionBuilder):
    def __init__(self, expr: Expression):
        assert isinstance(expr.wtype, wtypes.ARC4Struct)
        self.wtype: wtypes.ARC4Struct = expr.wtype
        super().__init__(expr)

    def member_access(self, name: str, location: SourceLocation) -> ExpressionBuilder | Literal:
        match name:
            case tuple_field if tuple_field in self.wtype.fields:
                index = self.wtype.names.index(tuple_field)
                return var_expression(
                    IndexExpression(
                        source_location=location,
                        base=self.expr,
                        index=UInt64Constant(value=index, source_location=location),
                        wtype=self.wtype.fields[tuple_field],
                    )
                )
            case "bytes":
                return get_bytes_expr_builder(self.expr)
            case "copy":
                return CopyBuilder(self.expr, location)
            case _:
                return super().member_access(name, location)

    def compare(
        self, other: ExpressionBuilder | Literal, op: BuilderComparisonOp, location: SourceLocation
    ) -> ExpressionBuilder:
        return arc4_compare_bytes(self, op, other, location)

    def bool_eval(self, location: SourceLocation, *, negate: bool = False) -> ExpressionBuilder:
        return bool_eval_to_constant(value=True, location=location, negate=negate)
