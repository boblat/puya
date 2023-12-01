from collections.abc import Sequence
from pathlib import Path

import attrs

from wyvern.logging_config import LogLevel


@attrs.define(kw_only=True)
class WyvernOptions:
    paths: Sequence[Path] = ()
    output_teal: bool = True
    output_awst: bool = False
    output_ssa_ir: bool = False
    output_optimization_ir: bool = False
    output_cssa_ir: bool = False
    output_post_ssa_ir: bool = False
    output_parallel_copies_ir: bool = False
    output_final_ir: bool = False
    out_dir: Path | None = None
    debug_level: int = 0
    optimization_level: int = 0
    log_level: LogLevel = LogLevel.notset
