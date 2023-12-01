import os
import sys
from pathlib import Path

import mypy.build
import mypy.errors
import mypy.find_sources
import mypy.fscache
import mypy.modulefinder
import mypy.nodes
import mypy.options
import structlog

from wyvern.awst import nodes as awst_nodes
from wyvern.awst_build.main import transform_ast
from wyvern.codegen.builder import compile_ir_to_teal
from wyvern.codegen.emitprogram import CompiledContract
from wyvern.context import CompileContext
from wyvern.errors import Errors, InternalError, ParseError
from wyvern.ir.destructure.main import destructure_ssa
from wyvern.ir.main import build_ir
from wyvern.ir.optimize.main import optimize_contract_ir
from wyvern.ir.to_text_visitor import output_contract_ir_to_path
from wyvern.options import WyvernOptions
from wyvern.parse import TYPESHED_PATH, ParseSource, parse_and_typecheck
from wyvern.utils import determine_out_dir

logger = structlog.get_logger(__name__)


def parse_with_mypy(wyvern_options: WyvernOptions) -> CompileContext:
    mypy_options = get_mypy_options()
    # this generates the ASTs from the build sources, and all imported modules (recursively)
    try:
        parse_result = parse_and_typecheck(wyvern_options.paths, mypy_options)
    except mypy.errors.CompileError as ex:
        parse_errors = list[str]()
        parse_errors.extend(ex.messages)
        if not parse_errors:
            for a in ex.args:
                lines = a.splitlines()
                parse_errors.extend(lines)
        raise ParseError(parse_errors) from ex

    # Sometimes when we call back into mypy, there might be errors.
    # We don't want to crash when that happens.
    parse_result.manager.errors.set_file("<wyvern>", module=None, scope=None, options=mypy_options)

    # extract the source reader
    read_source = parse_result.manager.errors.read_source
    if read_source is None:
        raise InternalError("parse_results.manager.errors.read_source is None")

    errors = Errors()
    context = CompileContext(
        options=wyvern_options,
        parse_result=parse_result,
        errors=errors,
        read_source=read_source,
    )

    return context


def awst_to_teal(
    context: CompileContext, module_asts: dict[str, awst_nodes.Module]
) -> dict[ParseSource, list[CompiledContract]] | None:
    errors = context.errors
    parse_result = context.parse_result

    if errors.num_errors:
        return None
    module_irs = {
        module_name: [
            build_ir(context, node, module_asts)
            for node in module_ast.body
            if isinstance(node, awst_nodes.ContractFragment) and not node.is_abstract
        ]
        for module_name, module_ast in module_asts.items()
    }
    if errors.num_errors:
        return None

    result = dict[ParseSource, list[CompiledContract]]()
    for src in parse_result.sources:
        module_ir = module_irs.get(src.module_name)
        assert module_ir is not None

        if not module_ir:
            if src.is_explicit:
                logger.warning(f"No contracts found in explicitly named source file: {src.path}")
        else:
            for contract_ir in module_ir:
                out_dir = determine_out_dir(src.path.parent, context.options)
                contract_ir_base_path = out_dir / "_".join((src.path.stem, contract_ir.class_name))
                if context.options.output_ssa_ir:
                    output_contract_ir_to_path(
                        contract_ir, contract_ir_base_path.with_suffix(".ssa.ir")
                    )

                if context.options.optimization_level > 0:
                    logger.info(
                        f"Optimizing {contract_ir.full_name}"
                        f" at level {context.options.optimization_level}"
                    )
                    contract_ir = optimize_contract_ir(
                        context,
                        contract_ir,
                        contract_ir_base_path if context.options.output_optimization_ir else None,
                    )

                contract_ir = destructure_ssa(context, contract_ir, contract_ir_base_path)

                compiled_contract = compile_ir_to_teal(context, contract_ir)
                result.setdefault(src, []).append(compiled_contract)

    if errors.num_errors:
        return None

    return result


def write_compiled_contracts(
    context: CompileContext,
    compiled_contracts_by_source_path: dict[ParseSource, list[CompiledContract]],
) -> None:
    for src, compiled_contracts in compiled_contracts_by_source_path.items():
        if len(compiled_contracts) == 1:
            base_path = determine_out_dir(src.path.parent, context.options) / src.path.stem
            write_contract_files(base_path=base_path, compiled_contract=compiled_contracts[0])
        else:
            for contract in compiled_contracts:
                base_path = determine_out_dir(src.path.parent, context.options)

                qualified_path = base_path / f"{src.path.stem}_{contract.name}"
                write_contract_files(base_path=qualified_path, compiled_contract=contract)


def compile_to_teal(wyvern_options: WyvernOptions) -> None:
    """Drive the actual core compilation step."""
    context = parse_with_mypy(wyvern_options)
    awst = transform_ast(context)
    compiled_contracts_by_source_path = awst_to_teal(context, awst)
    if compiled_contracts_by_source_path is None:
        logger.error("Build failed")
        sys.exit(1)
    elif not compiled_contracts_by_source_path:
        logger.error("No contracts discovered in any source files")
    elif wyvern_options.output_teal:
        write_compiled_contracts(context, compiled_contracts_by_source_path)


def get_mypy_options() -> mypy.options.Options:
    # TODO: build configuration interface to these options
    mypy_opts = mypy.options.Options()
    # improve mypy parsing performance by using a cut-down typeshed
    mypy_opts.custom_typeshed_dir = str(TYPESHED_PATH)
    mypy_opts.abs_custom_typeshed_dir = str(TYPESHED_PATH.resolve())

    mypy_opts.export_types = True
    mypy_opts.preserve_asts = True
    # next two options disable caching entirely.
    # slows things down but prevents intermittent failures.
    mypy_opts.incremental = False
    mypy_opts.cache_dir = os.devnull

    # strict mode flags, need to review these and all others too
    mypy_opts.disallow_any_generics = True
    mypy_opts.disallow_subclassing_any = True
    mypy_opts.disallow_untyped_calls = True
    mypy_opts.disallow_untyped_defs = True
    mypy_opts.disallow_incomplete_defs = True
    mypy_opts.check_untyped_defs = True
    mypy_opts.disallow_untyped_decorators = True
    mypy_opts.warn_redundant_casts = True
    mypy_opts.warn_unused_ignores = True
    mypy_opts.warn_return_any = True
    mypy_opts.strict_equality = True
    mypy_opts.strict_concatenate = True

    # disallow use of any
    mypy_opts.disallow_any_unimported = True
    mypy_opts.disallow_any_expr = True
    mypy_opts.disallow_any_decorated = True
    mypy_opts.disallow_any_explicit = True

    # mypy_opts.dump_graph = True
    # mypy_opts.dump_deps = True

    mypy_opts.pretty = True  # show source in output

    return mypy_opts


def write_contract_files(base_path: Path, compiled_contract: CompiledContract) -> None:
    output_paths = {
        ".approval.teal": compiled_contract.approval_program.src,
        ".approval.debug.teal": compiled_contract.approval_program.debug_src,
        ".clear.teal": compiled_contract.clear_program.src,
        ".clear.debug.teal": compiled_contract.clear_program.debug_src,
    }
    for suffix, src in output_paths.items():
        if src is None:
            continue
        output_path = base_path.with_suffix(suffix)
        output_text = "\n".join(src)
        logger.info(f"Writing {output_path}")
        output_path.write_text(output_text, encoding="utf-8")
