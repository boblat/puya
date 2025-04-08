import typing

from algopy import ARC4Contract, Bytes, arc4


class MyContract(ARC4Contract):
    @arc4.abimethod()
    def test_receive(
        self, b_32: Bytes[typing.Literal[32]], b_64: Bytes[typing.Literal[64]]
    ) -> None:
        assert b_32 + b_32 == b_64
