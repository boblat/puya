"""
Used to map algopy/_gen.pyi stubs to AWST.
Referenced by both scripts/generate_stubs.py and src/puya/awst_build/eb/intrinsics.py
"""

from collections.abc import Sequence, Set
from functools import cached_property

import attrs

from puya.awst_build import pytypes
from puya.errors import InternalError


@attrs.frozen
class PropertyOpMapping:
    op_code: str
    immediate: str
    typ: pytypes.PyType = attrs.field(
        validator=attrs.validators.not_(attrs.validators.in_([pytypes.NoneType]))
    )


@attrs.frozen
class FunctionOpMapping:
    op_code: str
    immediates: Sequence[str | int | type[str | int]] = attrs.field(
        default=(), converter=tuple[str | int | type[str | int], ...]
    )  # TODO: field validation
    args: Sequence[Sequence[pytypes.PyType] | int] = attrs.field(
        default=(), converter=tuple[Sequence[pytypes.PyType] | int, ...]
    )
    """Mapping of stack argument names to valid types for the argument,
     in descending priority for literal conversions
     OR
        TODO
    """

    @cached_property
    def literal_arg_positions(self) -> Set[int]:
        return {idx for idx, arg in enumerate(self.args) if isinstance(arg, int)}

    def __attrs_post_init__(self) -> None:
        for idx, arg in enumerate(self.args):
            if isinstance(arg, int):
                try:
                    should_be_type = self.immediates[arg]
                except IndexError:
                    raise InternalError(
                        f"intrinsic {self.op_code!r} argument {idx}: immediates index out of range"
                    ) from None
                else:
                    if should_be_type not in (str, int):
                        raise InternalError(
                            f"intrinsic {self.op_code!r}: immediate from args should have type"
                        )
            else:
                if not arg:
                    raise InternalError(
                        f"intrinsic {self.op_code!r}: no stack input types provided"
                    )
                if pytypes.BigUIntType in arg and pytypes.UInt64Type in arg:
                    raise InternalError(f"intrinsic {self.op_code!r}: overlap in integer types")


@attrs.frozen(kw_only=True)
class OpMappingWithOverloads:
    # TODO: validate arity matches up with all overloads
    arity: int = attrs.field(validator=attrs.validators.ge(0))
    result: pytypes.PyType = pytypes.NoneType
    """Types output by TEAL op"""
    overloads: Sequence[FunctionOpMapping] = attrs.field(
        validator=attrs.validators.min_len(1), converter=tuple[FunctionOpMapping, ...]
    )
