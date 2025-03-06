"""Tests for mypy incremental error output."""

from __future__ import annotations

from nypy import build
from nypy.errors import CompileError
from nypy.modulefinder import BuildSource
from nypy.options import Options
from nypy.test.data import DataDrivenTestCase, DataSuite
from nypy.test.helpers import assert_string_arrays_equal


class ErrorStreamSuite(DataSuite):
    required_out_section = True
    base_path = "."
    files = ["errorstream.test"]

    def run_case(self, testcase: DataDrivenTestCase) -> None:
        test_error_stream(testcase)


def test_error_stream(testcase: DataDrivenTestCase) -> None:
    """Perform a single error streaming test case.

    The argument contains the description of the test case.
    """
    options = Options()
    options.show_traceback = True
    options.hide_error_codes = True

    logged_messages: list[str] = []

    def flush_errors(filename: str | None, msgs: list[str], serious: bool) -> None:
        if msgs:
            logged_messages.append("==== Errors flushed ====")
            logged_messages.extend(msgs)

    sources = [BuildSource("main", "__main__", "\n".join(testcase.input))]
    try:
        build.build(sources=sources, options=options, flush_errors=flush_errors)
    except CompileError as e:
        assert e.messages == []

    assert_string_arrays_equal(
        testcase.output, logged_messages, f"Invalid output ({testcase.file}, line {testcase.line})"
    )
