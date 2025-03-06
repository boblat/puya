import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import attrs
import mypy.build
import mypy.find_sources
import mypy.fscache
import mypy.modulefinder
import mypy.nodes
import mypy.types
import mypy.util
from mypy.options import NEW_GENERIC_SYNTAX

from puya import log
from puyapy.compile import _get_python_executable
from puyapy.parse import _parse_log_message

logger = log.get_logger(__name__)


@attrs.define
class TsType:
    name: str
    base_types: list[str] = attrs.field(factory=list)
    fields: dict[str, str] = attrs.field(factory=dict)


@attrs.frozen
class SourceModule:
    name: str
    node: mypy.nodes.MypyFile
    path: Path
    lines: Sequence[str] | None


def safe_name(name: str) -> str:
    match name:
        case "Function":
            return "_Function"
        case _:
            return name


def base_type_name(x: mypy.nodes.Expression) -> str:
    match x:
        case mypy.nodes.NameExpr(name=name):
            return safe_name(name)
        case mypy.nodes.MemberExpr(
            expr=mypy.nodes.NameExpr(fullname="puya.awst.nodes.enum"), name=name
        ):
            return safe_name(name)
        case mypy.nodes.MemberExpr(
            expr=mypy.nodes.NameExpr(fullname="puya.awst.txn_fields.enum"), name=name
        ):
            return safe_name(name)
        case mypy.nodes.MemberExpr(expr=mypy.nodes.NameExpr(name="enum"), name=name):
            return name
        case _:
            raise ValueError("Unexpected value")


def capitalize_first(x: str) -> str:
    if len(x) == 0:
        return x
    return x[0].upper() + x[1:]


def to_camel_case(snake_str: str) -> str:
    return "".join(capitalize_first(x) for x in snake_str.split("_"))


def to_lower_camel_case(snake_str: str) -> str:
    # We capitalize the first letter of each component except the first one
    # with the 'capitalize' method and join them together.
    camel_string = to_camel_case(snake_str)
    return camel_string[0].lower() + camel_string[1:]


def extract_type_name(_t: mypy.types.Type) -> str:
    match _t:
        case mypy.types.TypeAliasType(alias=mypy.nodes.TypeAlias(target=target)):
            return extract_type_name(target)
        case mypy.types.UnionType(items=items):
            return " | ".join(
                extract_type_name(t) for t in items if extract_type_name(t) != "Range"
            )
        case mypy.types.NoneType():
            return "null"

        case mypy.types.TypeType(item=item):
            return extract_type_name(item)
        case mypy.types.Instance(type=type, args=args):
            if type.fullname == "typing.Sequence":
                return f"Array<{extract_type_name(args[0])}>"
            if type.fullname == "typing.Mapping":
                return f"Map<{extract_type_name(args[0])}, {extract_type_name(args[1])}>"
            if type.fullname == "immutabledict.immutabledict":
                return f"Map<{extract_type_name(args[0])}, {extract_type_name(args[1])}>"
            if type.fullname == "builtins.tuple":
                return "[" + ", ".join(extract_type_name(t) for t in args) + "]"
            if type.fullname == "puya.utils.StableSet":
                return f"Set<{extract_type_name(args[0])}>"
            if type.fullname == "typing.AbstractSet":
                return f"Set<{extract_type_name(args[0])}>"
            if type.fullname == "builtins.str":
                return "string"
            if type.fullname == "builtins.int":
                return "bigint"
            if type.fullname == "builtins.bool":
                return "boolean"
            if type.fullname == "builtins.bytes":
                return "Uint8Array"
            if type.fullname == "decimal.Decimal":
                return "string"
            if type.fullname == "puya.awst.nodes.Label":
                return "string"
            if type.fullname.startswith("puya.awst.wtypes"):
                return f"wtypes.{type.fullname[17:]}"
            if type.fullname.startswith("puya."):
                module_name = ".".join(type.fullname.split(".")[:-1])
                return type.fullname[len(module_name) + 1 :]

            return type.fullname
        case _:
            raise ValueError("AAARGH")


def visit_class(c: mypy.nodes.ClassDef) -> TsType:
    type_instance = TsType(safe_name(c.name))
    type_instance.base_types.extend(base_type_name(n) for n in c.base_type_exprs)
    # Special case: Treat this enum as a string enum
    if c.fullname == "puya.awst.nodes.PuyaLibFunction":
        type_instance.base_types.append("StrEnum")
    is_str_enum = "StrEnum" in type_instance.base_types
    enum_auto = 1
    for x in c.defs.body:
        match x:
            case mypy.nodes.AssignmentStmt(
                lvalues=[
                    mypy.nodes.NameExpr(
                        name=member_name,
                    )
                ],
                rvalue=mypy.nodes.StrExpr(value=enum_value),
            ) if enum_value and is_str_enum:
                member_name_camel = to_lower_camel_case(member_name)
                type_instance.fields[member_name_camel] = enum_value
            case mypy.nodes.AssignmentStmt(
                lvalues=[
                    mypy.nodes.NameExpr(
                        name=member_name,
                        node=mypy.nodes.Var(
                            type=mypy.types.Instance(
                                type=mypy.nodes.TypeInfo(fullname="enum.auto")
                            ),
                        ),
                    )
                ]
            ):
                member_name_camel = to_lower_camel_case(member_name)
                type_instance.fields[member_name_camel] = (
                    member_name if is_str_enum else f"{enum_auto}"
                )
                enum_auto += 1
            case mypy.nodes.AssignmentStmt(
                lvalues=[
                    mypy.nodes.NameExpr(
                        name=member_name,
                    )
                ]
            ) if is_str_enum:
                member_name_camel = to_lower_camel_case(member_name)
                type_instance.fields[member_name_camel] = member_name
            case mypy.nodes.AssignmentStmt(
                lvalues=[
                    mypy.nodes.NameExpr(
                        name=member_name,
                        node=mypy.nodes.Var(
                            type=member_type,
                        ),
                    )
                ]
            ) if member_type:
                # if member_name == "methods" and c.name == "Contract":
                #     continue
                member_name_camel = to_lower_camel_case(member_name)
                type_instance.fields[member_name_camel] = extract_type_name(member_type)
            case mypy.nodes.FuncDef():
                pass
            case _:
                pass

    return type_instance


def print_str_enum(ts_type: TsType) -> Iterable[str]:
    yield f"export enum {ts_type.name} {{"
    for field, value in ts_type.fields.items():
        enum_value = field if value == "enum" else value
        yield f'  {field} = "{enum_value}",'
    yield "}"


def print_num_enum(ts_type: TsType) -> Iterable[str]:
    yield f"export enum {ts_type.name} {{"
    for field in ts_type.fields:
        yield f"  {field} = {ts_type.fields[field]},"
    yield "}"


def get_visitor_type(ts_type: TsType, ts_types: list[TsType]) -> str | None:
    match ts_type.name:
        case "Expression" | "Statement" | "RootNode" | "ContractMemberNode":
            return ts_type.name

    base_types = (t for t in ts_types if t.name in ts_type.base_types)
    base_visitor_types = (get_visitor_type(t, ts_types) for t in base_types)

    return next((t for t in base_visitor_types if t), None)


def print_visitor(ts_types: list[TsType], name: str) -> Iterable[str]:
    yield f"export interface {name}Visitor<T> {{"
    for ts_type in ts_types:
        if "ABC" in ts_type.base_types or ts_type.name == "Node":
            continue

        if get_visitor_type(ts_type, ts_types) == name:
            yield f"  visit{ts_type.name}({name[0].lower()}{name[1:]}: {ts_type.name}): T"

    yield "}"


def print_concrete_type_map(ts_types: list[TsType]) -> Iterable[str]:
    yield "export const concreteNodes = {"
    for ts_type in ts_types:
        if "StrEnum" in ts_type.base_types:
            continue
        if "Enum" in ts_type.base_types:
            continue
        if "ABC" in ts_type.base_types or ts_type.name == "Node":
            continue
        yield f"  {ts_type.name[0].lower()}{ts_type.name[1:]}: {ts_type.name},"

    # Special cases
    yield "  uInt64Constant: IntegerConstant,"
    yield "  bigUIntConstant: IntegerConstant,"
    yield "} as const"


def print_type(ts_type: TsType, ts_types: list[TsType]) -> Iterable[str]:
    if "StrEnum" in ts_type.base_types:
        yield from print_str_enum(ts_type)
        return
    if "Enum" in ts_type.base_types:
        yield from print_num_enum(ts_type)
        return
    is_abstract = "ABC" in ts_type.base_types or ts_type.name == "Node"
    visitor_type = get_visitor_type(ts_type, ts_types)

    abstract_kw = " abstract " if is_abstract else " "
    bases = [b for b in ts_type.base_types if b != "ABC"]
    if len(bases) == 1:
        extends = f" extends {bases[0]} "
    elif len(bases) > 1:
        extends = f" extends classes({",".join(bases)}) "
    else:
        extends = ""
    yield f"export{abstract_kw}class {ts_type.name}{extends}{{"

    yield f"  constructor(props: Props<{ts_type.name}>) {{"
    if len(bases) == 1:
        yield "    super(props)"
    elif len(bases) > 1:
        yield "    super("
        yield ", ".join("[props]" for _ in bases)
        yield ")"
    for field_name in ts_type.fields:
        yield f"    this.{field_name} = props.{field_name}"
    yield "  }"

    for field_name, field_type in ts_type.fields.items():
        if field_name == "id" and ts_type.name == "SingleEvaluation":
            yield f"  readonly {field_name}: symbol"
        elif "undefined" in field_type:
            yield f"  readonly {field_name}?: {field_type}"
        else:
            yield f"  readonly {field_name}: {field_type}"
    if not is_abstract and visitor_type:
        yield f"  accept<T>(visitor: {visitor_type}Visitor<T>): T {{"
        yield f"     return visitor.visit{ts_type.name}(this)"
        yield "   }"
    elif is_abstract and visitor_type:
        yield f"  abstract accept<T>(visitor: {visitor_type}Visitor<T>): T"

    yield "}"


def print_types(ts_types: list[TsType]) -> Iterable[str]:
    yield "/* AUTOGENERATED FILE - DO NOT EDIT (see puya/scripts/generate_ts_nodes.py) */"
    yield "import { classes } from 'polytype'"
    yield "import type { Props } from '../typescript-helpers'"
    yield (
        "import type { ContractReference, LogicSigReference, OnCompletionAction } "
        "from './models'"
    )
    yield "import type { SourceLocation } from './source-location'"
    yield "import type { TxnField } from './txn-fields'"
    yield "import type { wtypes } from './wtypes'"

    for t in ts_types:
        if t.name == "TxnField":
            continue
        yield from print_type(t, ts_types)

    yield "export type LValue = "
    yield "    | VarExpression"
    yield "    | FieldExpression"
    yield "    | IndexExpression"
    yield "    | TupleExpression"
    yield "    | AppStateExpression"
    yield "    | AppAccountStateExpression"

    yield "export type Constant = "
    yield "  | IntegerConstant"
    yield "  | BoolConstant"
    yield "  | BytesConstant"
    yield "  | StringConstant"

    yield "export type AWST = Contract | LogicSignature | Subroutine"
    yield "export type ARC4MethodConfig = ARC4BareMethodConfig | ARC4ABIMethodConfig"

    yield from print_concrete_type_map(ts_types)

    yield from print_visitor(ts_types, "Expression")
    yield from print_visitor(ts_types, "Statement")
    yield from print_visitor(ts_types, "ContractMemberNode")
    yield from print_visitor(ts_types, "RootNode")


def write_file(ts_types: list[TsType], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("\n".join(print_types(ts_types)), encoding="utf-8")


def get_mypy_options() -> mypy.options.Options:
    mypy_opts = mypy.options.Options()

    # set python_executable so third-party packages can be found
    mypy_opts.python_executable = _get_python_executable()

    mypy_opts.preserve_asts = True
    mypy_opts.include_docstrings = True
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
    mypy_opts.disallow_any_unimported = False
    mypy_opts.disallow_any_expr = False
    mypy_opts.disallow_any_decorated = False
    mypy_opts.disallow_any_explicit = False

    mypy_opts.pretty = True  # show source in output

    return mypy_opts


_MYPY_FSCACHE = mypy.fscache.FileSystemCache()


def parse_and_typecheck(
    paths: Sequence[Path], mypy_options: mypy.options.Options
) -> tuple[mypy.build.BuildManager, dict[str, SourceModule]]:
    """Generate the ASTs from the build sources, and all imported modules (recursively)"""

    # ensure we have the absolute, canonical paths to the files
    resolved_input_paths = {p.resolve() for p in paths}
    # creates a list of BuildSource objects from the contract Paths
    mypy_build_sources = mypy.find_sources.create_source_list(
        paths=[str(p) for p in resolved_input_paths],
        options=mypy_options,
        fscache=_MYPY_FSCACHE,
    )
    result = _mypy_build(mypy_build_sources, mypy_options, _MYPY_FSCACHE)
    # Sometimes when we call back into mypy, there might be errors.
    # We don't want to crash when that happens.
    result.manager.errors.set_file("<puyapy>", module=None, scope=None, options=mypy_options)
    missing_module_names = {s.module for s in mypy_build_sources} - result.manager.modules.keys()
    # Note: this shouldn't happen, provided we've successfully disabled the mypy cache
    assert (
        not missing_module_names
    ), f"mypy parse failed, missing modules: {', '.join(missing_module_names)}"

    # order modules by dependency, and also sanity check the contents
    ordered_modules = {}
    for scc_module_names in mypy.build.sorted_components(result.graph):
        for module_name in scc_module_names:
            module = result.manager.modules[module_name]
            assert (
                module_name == module.fullname
            ), f"mypy module mismatch, expected {module_name}, got {module.fullname}"
            assert module.path, f"no path for mypy module: {module_name}"
            module_path = Path(module.path).resolve()
            if module_path.is_dir():
                # this module is a module directory with no __init__.py, ie it contains
                # nothing and is only in the graph as a reference
                pass
            else:
                lines = mypy.util.read_py_file(str(module_path), _MYPY_FSCACHE.read)
                ordered_modules[module_name] = SourceModule(
                    name=module_name, node=module, path=module_path, lines=lines
                )

    return result.manager, ordered_modules


def _mypy_build(
    sources: list[mypy.modulefinder.BuildSource],
    options: mypy.options.Options,
    fscache: mypy.fscache.FileSystemCache | None,
) -> mypy.build.BuildResult:
    """Simple wrapper around mypy.build.build

    Makes it so that check errors and parse errors are handled the same (ie with an exception)
    """

    all_messages = list[str]()

    def flush_errors(
        _filename: str | None,
        new_messages: list[str],
        _is_serious: bool,  # noqa: FBT001
    ) -> None:
        all_messages.extend(msg for msg in new_messages if os.devnull not in msg)

    try:
        result = mypy.build.build(
            sources=sources,
            options=options,
            flush_errors=flush_errors,
            fscache=fscache,
        )
    finally:
        _log_mypy_messages(all_messages)
    return result


def _log_mypy_message(message: log.Log | None, related_lines: list[str]) -> None:
    if not message:
        return
    logger.log(
        message.level, message.message, location=message.location, related_lines=related_lines
    )


def _log_mypy_messages(messages: list[str]) -> None:
    first_message: log.Log | None = None
    related_messages = list[str]()
    for message_str in messages:
        message = _parse_log_message(message_str)
        if not first_message:
            first_message = message
        elif not message.location:
            # collate related error messages and log together
            related_messages.append(message.message)
        else:
            _log_mypy_message(first_message, related_messages)
            related_messages = []
            first_message = message
    _log_mypy_message(first_message, related_messages)


def parse_with_mypy(paths: Sequence[Path]) -> dict[str, SourceModule]:
    mypy_options = get_mypy_options()

    # Enable new generic syntax
    mypy_options.enable_incomplete_feature += [NEW_GENERIC_SYNTAX]
    # this generates the ASTs from the build sources, and all imported modules (recursively)
    (manager, ordered_modules) = parse_and_typecheck(paths, mypy_options)
    # Sometimes when we call back into mypy, there might be errors.
    # We don't want to crash when that happens.
    manager.errors.set_file("<puyapy>", module=None, scope=None, options=mypy_options)

    return ordered_modules


def generate_file(*, out_path: Path, puya_path: Path) -> None:
    nodes_path = puya_path / "awst/nodes.py"
    txn_fields_path = puya_path / "awst/txn_fields.py"
    paths = [nodes_path, txn_fields_path]

    ignored_types = ("ContinueStatement", "BreakStatement")

    ts_types = list[TsType]()

    ordered_modules = parse_with_mypy(paths)
    for module in ordered_modules.values():
        if module.path not in (nodes_path, txn_fields_path):
            continue
        for statement in module.node.defs:
            match statement:
                case mypy.nodes.ClassDef() as cdef:
                    if cdef.name.startswith("_"):
                        continue
                    ts_type = visit_class(cdef)
                    if ts_type.name in ignored_types:
                        continue
                    ts_types.append(ts_type)

    write_file(ts_types, out_path)
