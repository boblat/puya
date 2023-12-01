from collections.abc import Sequence

import attrs

from wyvern.codegen.utils import format_bytes
from wyvern.ir.types_ import AVMBytesEncoding


@attrs.frozen(str=False, kw_only=True)
class TealOp:
    op_code: str
    comment: str | None = attrs.field(default=None)
    """A comment that is always emitted, should only be used for user comments related to an
    op such as assert or err"""

    @property
    def immediates(self) -> Sequence[int | str]:
        return ()

    def __str__(self) -> str:
        teal_args = [self.op_code, *map(str, self.immediates)]
        if self.comment:
            teal_args.append(f"// {self.comment}")
        return " ".join(teal_args)


@attrs.frozen
class Dup(TealOp):
    op_code: str = "dup"


@attrs.frozen
class Pop(TealOp):
    op_code: str = "pop"


@attrs.frozen
class RetSub(TealOp):
    op_code: str = "retsub"


@attrs.frozen
class TealOpN(TealOp):
    n: int

    @property
    def immediates(self) -> Sequence[int | str]:
        return (self.n,)


@attrs.frozen
class Cover(TealOpN):
    op_code: str = "cover"


@attrs.frozen
class Uncover(TealOpN):
    op_code: str = "uncover"


@attrs.frozen
class Dig(TealOpN):
    op_code: str = "dig"


@attrs.frozen
class Bury(TealOpN):
    op_code: str = "bury"


@attrs.frozen
class FrameDig(TealOpN):
    op_code: str = "frame_dig"


@attrs.frozen
class FrameBury(TealOpN):
    op_code: str = "frame_bury"


@attrs.frozen
class Store(TealOpN):
    op_code: str = "store"


@attrs.frozen
class Load(TealOpN):
    op_code: str = "load"


@attrs.frozen
class PushInt(TealOp):
    n: int | str
    op_code: str = "int"

    @property
    def immediates(self) -> Sequence[int | str]:
        return (self.n,)


@attrs.frozen
class PopN(TealOpN):
    op_code: str = "popn"


@attrs.frozen
class DupN(TealOpN):
    op_code: str = "dupn"


@attrs.frozen
class Proto(TealOp):
    parameters: int
    returns: int
    op_code: str = "proto"

    @property
    def immediates(self) -> Sequence[int | str]:
        return self.parameters, self.returns


@attrs.frozen
class PushBytes(TealOp):
    n: bytes
    encoding: AVMBytesEncoding
    op_code: str = "byte"

    @property
    def immediates(self) -> Sequence[int | str]:
        bytes_str = format_bytes(self.n, self.encoding)
        if self.encoding in (
            AVMBytesEncoding.utf8,
            AVMBytesEncoding.base16,
        ):
            return (bytes_str,)
        hint = self.encoding.name
        return hint, bytes_str


@attrs.frozen
class PushAddress(TealOp):
    a: str
    op_code: str = "addr"

    @property
    def immediates(self) -> Sequence[int | str]:
        return (self.a,)


@attrs.frozen
class Intrinsic(TealOp):
    immediates: Sequence[int | str]


@attrs.frozen
class CallSub(TealOp):
    target: str
    op_code: str = "callsub"

    @property
    def immediates(self) -> Sequence[int | str]:
        return (self.target,)
