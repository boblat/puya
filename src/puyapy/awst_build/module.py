import typing
from collections.abc import Callable, Set

import attrs

import nypy.nodes
import nypy.types
import nypy.visitor
from puya import log
from puya.algo_constants import MAX_SCRATCH_SLOT_NUMBER
from puya.awst.nodes import AWST, LogicSignature, RootNode, StateTotals
from puya.errors import CodeError, InternalError
from puya.program_refs import LogicSigReference
from puya.utils import coalesce
from puyapy.awst_build import constants, pytypes
from puyapy.awst_build.arc4_client import ARC4ClientASTVisitor
from puyapy.awst_build.base_mypy_visitor import (
    BaseMyPyExpressionVisitor,
    BaseMyPyStatementVisitor,
)
from puyapy.awst_build.context import ASTConversionModuleContext
from puyapy.awst_build.contract import ContractASTConverter
from puyapy.awst_build.exceptions import UnsupportedASTError
from puyapy.awst_build.subroutine import FunctionASTConverter
from puyapy.awst_build.utils import (
    extract_bytes_literal_from_mypy,
    fold_binary_expr,
    fold_unary_expr,
    get_decorators_by_fullname,
    get_subroutine_decorator_inline_arg,
    get_unaliased_fullname,
)
from puyapy.models import ConstantValue, ContractClassOptions

logger = log.get_logger(__name__)


DeferredRootNode: typing.TypeAlias = Callable[[ASTConversionModuleContext], RootNode | None]

StatementResult: typing.TypeAlias = list[DeferredRootNode]


@attrs.frozen(kw_only=True)
class _LogicSigDecoratorInfo:
    name_override: str | None
    avm_version: int | None
    scratch_slots: Set[int]


_BUILTIN_INHERITABLE: typing.Final = frozenset(
    ("builtins.object", "abc.ABC", *nypy.types.PROTOCOL_NAMES)
)


class ModuleASTConverter(
    BaseMyPyStatementVisitor[StatementResult], BaseMyPyExpressionVisitor[ConstantValue]
):
    """This does basic validation, and traversal of valid module scope elements, collecting
    and folding constants."""

    def __init__(self, context: ASTConversionModuleContext, module: nypy.nodes.MypyFile):
        super().__init__(context)
        self.module_name: typing.Final = module.fullname
        self._pre_parse_result = list[tuple[nypy.nodes.Context, StatementResult]]()
        for node in module.defs:
            with self.context.log_exceptions(fallback_location=node):
                items = node.accept(self)
                self._pre_parse_result.append((node, items))

    def convert(self) -> AWST:
        awst = []
        for location, deferrals in self._pre_parse_result:
            with self.context.log_exceptions(fallback_location=location):
                for deferred in deferrals:
                    awst_node = deferred(self.context)
                    if awst_node is not None:
                        awst.append(awst_node)
        return awst

    # Supported Statements

    def empty_statement(self, _stmt: nypy.nodes.Statement) -> StatementResult:
        return []

    def visit_function(
        self, func_def: nypy.nodes.FuncDef, decorator: nypy.nodes.Decorator | None
    ) -> StatementResult:
        self._precondition(
            func_def.abstract_status == nypy.nodes.NOT_ABSTRACT,
            "module level functions should not be classified as abstract by mypy",
            decorator or func_def,
        )
        dec_by_fullname = get_decorators_by_fullname(self.context, decorator) if decorator else {}
        source_location = self._location(decorator or func_def)
        logicsig_dec = dec_by_fullname.pop(constants.LOGICSIG_DECORATOR, None)
        if logicsig_dec:
            for dec_fullname, dec in dec_by_fullname.items():
                self._error(f'Unsupported logicsig decorator "{dec_fullname}"', dec)
            info = self._process_logic_sig_decorator(logicsig_dec)

            def deferred(ctx: ASTConversionModuleContext) -> RootNode:
                program = FunctionASTConverter.convert(
                    ctx, func_def, source_location, inline=False
                )
                ctx.register_pytype(pytypes.LogicSigType, alias=func_def.fullname)
                return LogicSignature(
                    id=LogicSigReference(func_def.fullname),
                    program=program,
                    short_name=coalesce(info.name_override, program.name),
                    docstring=func_def.docstring,
                    source_location=self._location(logicsig_dec),
                    avm_version=info.avm_version,
                    reserved_scratch_space=info.scratch_slots,
                )

            return [deferred]
        subroutine_dec = dec_by_fullname.pop(constants.SUBROUTINE_HINT, None)
        if subroutine_dec is None:
            self._error(
                f"free functions must be annotated with @{constants.SUBROUTINE_HINT_ALIAS}",
                func_def,
            )
            inline = None
        else:
            inline = get_subroutine_decorator_inline_arg(self.context, subroutine_dec)
        abimethod_dec = dec_by_fullname.pop(constants.ABIMETHOD_DECORATOR, None)
        if abimethod_dec is not None:
            self._error("free functions cannot be ARC4 ABI methods", abimethod_dec)
        # any further decorators are unsupported
        for dec_fullname, dec in dec_by_fullname.items():
            self._error(f'unsupported function decorator "{dec_fullname}"', dec)

        return [
            lambda ctx: FunctionASTConverter.convert(ctx, func_def, source_location, inline=inline)
        ]

    def _process_logic_sig_decorator(
        self, decorator: nypy.nodes.Expression
    ) -> _LogicSigDecoratorInfo:
        name_override = None
        avm_version = None
        scratch_slot_reservations = set[int]()
        match decorator:
            case nypy.nodes.NameExpr():
                pass
            case nypy.nodes.CallExpr(arg_names=arg_names, args=args):
                for arg_name, arg in zip(arg_names, args, strict=True):
                    match arg_name:
                        case "name":
                            name_const = arg.accept(self)
                            if isinstance(name_const, str):
                                name_override = name_const
                            else:
                                self.context.error("expected a str", arg)
                        case "avm_version":
                            version_const = arg.accept(self)
                            if isinstance(version_const, int):
                                avm_version = version_const
                            else:
                                self.context.error("expected an int", arg)
                        case "scratch_slots":
                            if isinstance(arg, nypy.nodes.TupleExpr | nypy.nodes.ListExpr):
                                slot_items = arg.items
                            else:
                                slot_items = [arg]
                            for item_expr in slot_items:
                                slots = _map_scratch_space_reservation(
                                    self.context, self, item_expr
                                )
                                if not slots:
                                    self.context.error("range is empty", item_expr)
                                elif (min(slots) < 0) or (max(slots) > MAX_SCRATCH_SLOT_NUMBER):
                                    self.context.error(
                                        "invalid scratch slot reservation - range must fall"
                                        f" entirely between 0 and {MAX_SCRATCH_SLOT_NUMBER}",
                                        item_expr,
                                    )
                                else:
                                    scratch_slot_reservations.update(slots)
            case _:
                self.context.error(
                    f"invalid {constants.LOGICSIG_DECORATOR_ALIAS} usage", decorator
                )
        return _LogicSigDecoratorInfo(
            name_override=name_override,
            avm_version=avm_version,
            scratch_slots=scratch_slot_reservations,
        )

    def visit_class_def(self, cdef: nypy.nodes.ClassDef) -> StatementResult:
        self.check_fatal_decorators(cdef.decorators)
        cdef_loc = self._location(cdef)
        match cdef.analyzed:
            case None:
                pass
            case nypy.nodes.TypedDictExpr():
                self._unsupported(cdef, "TypedDict classes are not supported")
            case nypy.nodes.NamedTupleExpr():
                return _process_named_tuple(self.context, cdef)
            case unrecognised_analysis_expression:
                self.context.warning(
                    "Analyzed class expression of type"
                    f" {type(unrecognised_analysis_expression).__name__},"
                    " please report this issue and check the compilation results carefully",
                    cdef_loc,
                )
        for decorator in cdef.decorators:
            self._error(
                (
                    "Unsupported decorator "
                    "(note: *all* class decorators are currently unsupported)"
                ),
                location=decorator,
            )
        info: nypy.nodes.TypeInfo = cdef.info
        if info.bad_mro:
            self._error("Bad MRO", location=cdef_loc)
            return []
        if info.metaclass_type and info.metaclass_type.type.fullname not in (
            "abc.ABCMeta",
            "typing._ProtocolMeta",
            "typing_extensions._ProtocolMeta",
            constants.CLS_ARC4_STRUCT_META,
            constants.STRUCT_META,
        ):
            self._error(
                f"Unsupported metaclass: {info.metaclass_type.type.fullname}",
                location=cdef_loc,
            )
            return []

        direct_base_types = [
            self.context.require_ptype(ti.fullname, cdef_loc)
            for ti in info.direct_base_classes()
            if ti.fullname not in _BUILTIN_INHERITABLE
        ]
        mro_types = [
            self.context.require_ptype(ti.fullname, cdef_loc)
            for ti in info.mro[1:]
            if ti.fullname not in _BUILTIN_INHERITABLE
        ]
        # create a static type, but don't register it yet,
        # it might end up being a struct instead
        static_type = pytypes.StaticType(
            name=cdef.fullname, bases=direct_base_types, mro=mro_types
        )
        for struct_base in (pytypes.StructBaseType, pytypes.ARC4StructBaseType):
            # note that since these struct bases aren't protocols, any subclasses
            # cannot be protocols
            if struct_base < static_type:
                if direct_base_types != [struct_base]:
                    self._error(
                        f"{struct_base} classes must only inherit directly from {struct_base}",
                        cdef_loc,
                    )
                return _process_struct(self.context, struct_base, cdef)

        if pytypes.ContractBaseType < static_type:
            module_name = cdef.info.module_name
            class_name = cdef.name
            assert "." not in class_name
            assert cdef.fullname == f"{module_name}.{class_name}"
            contract_type = pytypes.ContractType(
                module_name=module_name,
                class_name=class_name,
                bases=direct_base_types,
                mro=mro_types,
                source_location=cdef_loc,
            )
            self.context.register_pytype(contract_type)

            class_options = _process_contract_class_options(self.context, self, cdef)
            converter = ContractASTConverter(self.context, cdef, class_options, contract_type)
            return [converter.build]

        if info.is_protocol:
            self.context.register_pytype(static_type)
            if pytypes.ARC4ClientBaseType in direct_base_types:
                ARC4ClientASTVisitor.visit(self.context, cdef)
            else:
                logger.debug(
                    f"Skipping further processing of protocol class {cdef.fullname}",
                    location=cdef_loc,
                )
            return []

        logger.error(
            f"Unsupported class declaration."
            f" Contract classes must inherit either directly"
            f" or indirectly from {pytypes.ContractBaseType}.",
            location=cdef_loc,
        )
        return []

    def visit_operator_assignment_stmt(
        self, stmt: nypy.nodes.OperatorAssignmentStmt
    ) -> StatementResult:
        match stmt.lvalue:
            case nypy.nodes.NameExpr(name="__all__"):
                return self.empty_statement(stmt)
            case _:
                self._unsupported(stmt)

    def visit_if_stmt(self, stmt: nypy.nodes.IfStmt) -> StatementResult:
        for expr, block in zip(stmt.expr, stmt.body, strict=True):
            if self._evaluate_compile_time_constant_condition(expr):
                return block.accept(self)
        # if we didn't return, we end up here, which means the user code
        # should evaluate to the else body (if present)
        if stmt.else_body:
            return stmt.else_body.accept(self)
        else:
            return []

    def visit_assignment_stmt(self, stmt: nypy.nodes.AssignmentStmt) -> StatementResult:
        stmt_loc = self._location(stmt)
        self._precondition(
            bool(stmt.lvalues), "assignment statements should have at least one lvalue", stmt_loc
        )
        self._precondition(
            not stmt.invalid_recursive_alias,
            "assignment statement with invalid_recursive_alias",
            stmt_loc,
        )
        lvalues = stmt.lvalues
        if not self._check_assignment_lvalues(lvalues):
            return []
        if stmt.is_alias_def:
            match stmt.rvalue:
                case nypy.nodes.RefExpr(
                    is_alias_rvalue=True, node=nypy.nodes.TypeInfo(fullname=alias_fullname)
                ):
                    maybe_aliased_pytype = self.context.lookup_pytype(alias_fullname)
                    if maybe_aliased_pytype is None:
                        self.context.error(
                            f"Unknown type for type alias: {alias_fullname}", stmt_loc
                        )
                        return []
                    aliased_pytype = maybe_aliased_pytype
                case nypy.nodes.IndexExpr(
                    analyzed=nypy.nodes.TypeAliasExpr(
                        node=nypy.nodes.TypeAlias(alias_tvars=[], target=alias_type)
                    )
                ):
                    aliased_pytype = self.context.type_to_pytype(
                        alias_type, source_location=stmt_loc
                    )
                case _:
                    self._error("Unsupported type-alias format", stmt_loc)
                    return []
            for lvalue in lvalues:
                self.context.register_pytype(aliased_pytype, alias=lvalue.fullname)
            # We don't include type aliases in AWST since they're Python specific
            return []
        if any(lvalue.is_special_form for lvalue in lvalues):
            self._error("Unsupported type-form", stmt_loc)

        constant_value = stmt.rvalue.accept(self)
        for lvalue in lvalues:
            self.context.constants[lvalue.fullname] = constant_value

        return []

    def _check_assignment_lvalues(
        self, lvalues: list[nypy.nodes.Lvalue]
    ) -> typing.TypeGuard[list[nypy.nodes.NameExpr]]:
        """Does some pre-condition checks, including that all lvalues are simple (ie name-exprs),
        hence the TypeGuard return type. If it returns True, then we should try and handle the
        assignment."""
        result = True
        for lvalue in lvalues:
            if not isinstance(lvalue, nypy.nodes.NameExpr):
                self._error(
                    "Only straight-forward assignment targets supported at module level", lvalue
                )
                result = False
            else:
                if len(lvalues) > 1:
                    self._precondition(
                        not lvalue.is_special_form, "special form with multiple lvalues", lvalue
                    )
                if lvalue.name == "__all__":
                    # Special notation to denote the public members of a file, we don't need to
                    # store this as we don't validate star-imports, mypy does, hence the False.
                    # We check inside functions if this is attempted to be referenced and produce
                    # a specific error message.
                    result = False
                    if len(lvalues) > 1:
                        self._error("Multi-assignment with __all__ not supported", lvalue)
                    break
                fullname = ".".join((self.module_name, lvalue.name))
                # fullname might be unset if this is in
                # a conditional branch that's !TYPE_CHECKING
                if lvalue.fullname:
                    self._precondition(
                        lvalue.fullname == fullname,
                        f"assignment to module const - expected fullname of {fullname},"
                        f" but mypy had {lvalue.fullname}",
                        lvalue,
                    )
                else:
                    # fix it up
                    lvalue._fullname = fullname  # noqa: SLF001
        return result

    def visit_block(self, block: nypy.nodes.Block) -> StatementResult:
        result = StatementResult()
        for stmt in block.body:
            items = stmt.accept(self)
            result.extend(items)
        return result

    # Expressions

    def visit_name_expr(self, expr: nypy.nodes.NameExpr) -> ConstantValue:
        match expr.name:
            case "True":
                return True
            case "False":
                return False
        # TODO: is the GDEF check always valid?
        if not isinstance(expr.node, nypy.nodes.Var) or expr.kind != nypy.nodes.GDEF:
            self._unsupported(
                expr,
                "references to anything other than module-level constants "
                "are not supported at module level",
            )
        value = self.context.constants.get(expr.fullname)
        if value is None:
            raise CodeError(
                f"not a known constant value: {expr.name}"
                f" (qualified source name: {expr.fullname})",
                self._location(expr),
            )
        return value

    def visit_unary_expr(self, expr: nypy.nodes.UnaryExpr) -> ConstantValue:
        value = expr.expr.accept(self)
        return fold_unary_expr(self._location(expr), expr.op, value)

    def visit_op_expr(self, expr: nypy.nodes.OpExpr) -> ConstantValue:
        left_value = expr.left.accept(self)
        right_value = expr.right.accept(self)
        return fold_binary_expr(self._location(expr), expr.op, left_value, right_value)

    def visit_comparison_expr(self, expr: nypy.nodes.ComparisonExpr) -> ConstantValue:
        match (expr.operators, expr.operands):
            case ([op], [expr_left, expr_right]):
                lhs = expr_left.accept(self)
                rhs = expr_right.accept(self)
                return fold_binary_expr(self._location(expr), op, lhs, rhs)
            case _:
                self._unsupported(
                    expr, details="chained comparisons not supported at module level"
                )

    def visit_int_expr(self, expr: nypy.nodes.IntExpr) -> ConstantValue:
        return expr.value

    def visit_str_expr(self, expr: nypy.nodes.StrExpr) -> ConstantValue:
        return expr.value

    def visit_bytes_expr(self, expr: nypy.nodes.BytesExpr) -> ConstantValue:
        return extract_bytes_literal_from_mypy(expr)

    def visit_member_expr(self, expr: nypy.nodes.MemberExpr) -> ConstantValue:
        if (
            isinstance(expr.node, nypy.nodes.Var)
            and expr.kind == nypy.nodes.GDEF
            and expr.fullname
            and isinstance(expr.expr, nypy.nodes.RefExpr)
            and isinstance(expr.expr.node, nypy.nodes.MypyFile)
        ):
            try:
                return self.context.constants[expr.fullname]
            except KeyError as ex:
                self._unsupported(
                    expr, details="could not resolve external module constant", ex=ex
                )
        else:
            self._unsupported(expr)

    def visit_call_expr(self, expr: nypy.nodes.CallExpr) -> ConstantValue:
        # unfortunately, mypy doesn't preserve f-string identification info,
        # they get translated into either a str.join or str.format call at the top level
        # References:
        # https://github.com/python/mypy/blob/cb813259c3b9dff6aaa8686793cf6a0634cf1f69/mypy/fastparse.py#L1528
        # https://github.com/python/mypy/blob/cb813259c3b9dff6aaa8686793cf6a0634cf1f69/mypy/fastparse.py#L1550
        match expr:
            case nypy.nodes.CallExpr(
                callee=nypy.nodes.MemberExpr(expr=nypy.nodes.StrExpr(value=joiner), name="join"),
                args=[nypy.nodes.ListExpr() as args_list],
            ):
                args = [arg.accept(self) for arg in args_list.items]
                assert all(isinstance(x, str) for x in args)
                result = joiner.join(map(str, args))
                return result
            case nypy.nodes.CallExpr(
                callee=nypy.nodes.MemberExpr(
                    expr=nypy.nodes.StrExpr(value=format_str), name="format"
                )
            ):
                args = [arg.accept(self) for arg in expr.args]
                return format_str.format(*args)
            case nypy.nodes.CallExpr(
                callee=nypy.nodes.MemberExpr(
                    expr=nypy.nodes.StrExpr(value=str_value), name="encode"
                ),
                args=[nypy.nodes.StrExpr(value=encoding)],
                arg_names=[("encoding" | None)],
            ):
                return str_value.encode(encoding=encoding)
            case nypy.nodes.CallExpr(
                callee=nypy.nodes.MemberExpr(
                    expr=nypy.nodes.StrExpr(value=str_value), name="encode"
                ),
                args=[],
            ):
                return str_value.encode()
        return self._unsupported(expr)

    def visit_conditional_expr(self, expr: nypy.nodes.ConditionalExpr) -> ConstantValue:
        if self._evaluate_compile_time_constant_condition(expr.cond):
            return expr.if_expr.accept(self)
        else:
            return expr.else_expr.accept(self)

    def _evaluate_compile_time_constant_condition(self, expr: nypy.nodes.Expression) -> bool:
        kind = _infer_condition_value(expr)
        if kind == _TRUTH_VALUE_UNKNOWN:
            try:
                result = expr.accept(self)
            except UnsupportedASTError as ex:
                self._unsupported(
                    expr,
                    details="only constant valued conditions supported at module level",
                    ex=ex,
                )
            kind = _ALWAYS_TRUE if result else _ALWAYS_FALSE
        if kind in (_ALWAYS_TRUE, _TYPECHECKING_FALSE):
            return True
        elif kind in (_ALWAYS_FALSE, _TYPECHECKING_TRUE):
            return False
        else:
            raise InternalError(
                f"Unexpected reachability value: {kind}", location=self._location(expr)
            )

    def _unsupported(
        self,
        node: nypy.nodes.Node,
        details: str = "not supported at module level",
        ex: Exception | None = None,
    ) -> typing.Never:
        raise UnsupportedASTError(node, self._location(node), details=details) from ex

    # Unsupported Statements

    def visit_expression_stmt(self, stmt: nypy.nodes.ExpressionStmt) -> StatementResult:
        if isinstance(stmt.expr, nypy.nodes.StrExpr):
            # ignore any doc-strings at module level
            return []
        else:
            return self._unsupported(stmt)

    def visit_while_stmt(self, stmt: nypy.nodes.WhileStmt) -> StatementResult:
        return self._unsupported(stmt)

    def visit_for_stmt(self, stmt: nypy.nodes.ForStmt) -> StatementResult:
        return self._unsupported(stmt)

    def visit_break_stmt(self, stmt: nypy.nodes.BreakStmt) -> StatementResult:
        return self._unsupported(stmt)

    def visit_continue_stmt(self, stmt: nypy.nodes.ContinueStmt) -> StatementResult:
        return self._unsupported(stmt)

    def visit_assert_stmt(self, stmt: nypy.nodes.AssertStmt) -> StatementResult:
        return self._unsupported(stmt)

    def visit_del_stmt(self, stmt: nypy.nodes.DelStmt) -> StatementResult:
        return self._unsupported(stmt)

    def visit_match_stmt(self, stmt: nypy.nodes.MatchStmt) -> StatementResult:
        return self._unsupported(stmt)

    def visit_type_alias_stmt(self, stmt: nypy.nodes.TypeAliasStmt) -> StatementResult:
        return self._unsupported(stmt)

    # the remaining statements (below) are invalid at the module lexical scope,
    # mypy should have caught these errors already
    def visit_return_stmt(self, stmt: nypy.nodes.ReturnStmt) -> StatementResult:
        raise InternalError("encountered return statement at module level", self._location(stmt))

    # Unsupported Expressions

    def visit_super_expr(self, expr: nypy.nodes.SuperExpr) -> ConstantValue:
        return self._unsupported(expr)

    def visit_index_expr(self, expr: nypy.nodes.IndexExpr) -> ConstantValue:
        return self._unsupported(expr)

    def visit_ellipsis(self, expr: nypy.nodes.EllipsisExpr) -> ConstantValue:
        return self._unsupported(expr)

    def visit_dict_expr(self, expr: nypy.nodes.DictExpr) -> ConstantValue:
        return self._unsupported(expr)

    def visit_list_expr(self, expr: nypy.nodes.ListExpr) -> ConstantValue:
        return self._unsupported(expr)

    def visit_tuple_expr(self, expr: nypy.nodes.TupleExpr) -> ConstantValue:
        return self._unsupported(expr)

    def visit_list_comprehension(self, expr: nypy.nodes.ListComprehension) -> ConstantValue:
        return self._unsupported(expr)

    def visit_slice_expr(self, expr: nypy.nodes.SliceExpr) -> ConstantValue:
        return self._unsupported(expr)

    def visit_assignment_expr(self, o: nypy.nodes.AssignmentExpr) -> ConstantValue:
        return self._unsupported(o)

    def visit_lambda_expr(self, expr: nypy.nodes.LambdaExpr) -> ConstantValue:
        return self._unsupported(expr)


def _process_contract_class_options(
    context: ASTConversionModuleContext,
    expr_visitor: nypy.visitor.ExpressionVisitor[ConstantValue],
    cdef: nypy.nodes.ClassDef,
) -> ContractClassOptions:
    name_override: str | None = None
    scratch_slot_reservations = set[int]()
    state_totals = None
    avm_version = None
    for kw_name, kw_expr in cdef.keywords.items():
        with context.log_exceptions(kw_expr):
            match kw_name:
                case "name":
                    name_value = kw_expr.accept(expr_visitor)
                    if isinstance(name_value, str):
                        name_override = name_value
                    else:
                        context.error("unexpected argument type", kw_expr)
                case "scratch_slots":
                    if isinstance(kw_expr, nypy.nodes.TupleExpr | nypy.nodes.ListExpr):
                        slot_items = kw_expr.items
                    else:
                        slot_items = [kw_expr]
                    for item_expr in slot_items:
                        slots = _map_scratch_space_reservation(context, expr_visitor, item_expr)
                        if not slots:
                            context.error("range is empty", item_expr)
                        elif (min(slots) < 0) or (max(slots) > MAX_SCRATCH_SLOT_NUMBER):
                            context.error(
                                "invalid scratch slot reservation - range must fall"
                                f" entirely between 0 and {MAX_SCRATCH_SLOT_NUMBER}",
                                item_expr,
                            )
                        else:
                            scratch_slot_reservations.update(slots)
                case "state_totals":
                    if not isinstance(kw_expr, nypy.nodes.CallExpr):
                        context.error("unexpected argument type", kw_expr)
                    else:
                        arg_map = dict[str, int]()
                        for arg_name, arg in zip(kw_expr.arg_names, kw_expr.args, strict=True):
                            if arg_name is None:
                                context.error("unexpected positional argument", arg)
                            else:
                                arg_value = arg.accept(expr_visitor)
                                if not isinstance(arg_value, int):
                                    context.error("unexpected argument type", arg)
                                else:
                                    arg_map[arg_name] = arg_value
                        state_totals = StateTotals(**arg_map)
                case "avm_version":
                    version_value = kw_expr.accept(expr_visitor)
                    if isinstance(version_value, int):
                        avm_version = version_value
                    else:
                        context.error("unexpected argument type", kw_expr)
                case "metaclass":
                    context.error("metaclass option is unsupported", kw_expr)
                case _:
                    context.error("unrecognised class option", kw_expr)
    return ContractClassOptions(
        name_override=name_override,
        scratch_slot_reservations=scratch_slot_reservations,
        state_totals=state_totals,
        avm_version=avm_version,
    )


def _process_dataclass_like_fields(
    context: ASTConversionModuleContext, cdef: nypy.nodes.ClassDef, base_type: pytypes.PyType
) -> dict[str, pytypes.PyType] | None:
    fields = dict[str, pytypes.PyType]()
    has_error = False
    for stmt in cdef.defs.body:
        stmt_loc = context.node_location(stmt)
        match stmt:
            case nypy.nodes.ExpressionStmt(expr=nypy.nodes.StrExpr()):
                # ignore class docstring, already extracted
                # TODO: should we capture field "docstrings"?
                pass
            case nypy.nodes.AssignmentStmt(
                lvalues=[nypy.nodes.NameExpr(name=field_name)],
                rvalue=nypy.nodes.TempNode(),
                type=nypy.types.Type() as mypy_type,
            ):
                pytype = context.type_to_pytype(mypy_type, source_location=stmt_loc)
                fields[field_name] = pytype
                if isinstance((maybe_err := pytype.wtype), str):
                    logger.error(maybe_err, location=stmt_loc)
                    has_error = True
            case nypy.nodes.SymbolNode(name=symbol_name) if (
                cdef.info.names[symbol_name].plugin_generated
            ):
                pass
            case nypy.nodes.PassStmt():
                pass
            case _:
                logger.error(
                    f"unsupported syntax for {base_type} member declaration", location=stmt_loc
                )
                has_error = True
    return fields if not has_error else None


def _process_struct(
    context: ASTConversionModuleContext, base: pytypes.PyType, cdef: nypy.nodes.ClassDef
) -> StatementResult:
    fields = _process_dataclass_like_fields(context, cdef, base)
    if fields is None:
        return []
    cls_loc = context.node_location(cdef)
    frozen = cdef.info.metadata["dataclass"]["frozen"]
    assert isinstance(frozen, bool)
    struct_typ = pytypes.StructType(
        base=base,
        name=cdef.fullname,
        desc=cdef.docstring,
        fields=fields,
        frozen=frozen,
        source_location=cls_loc,
    )
    context.register_pytype(struct_typ)
    return []


def _process_named_tuple(
    context: ASTConversionModuleContext, cdef: nypy.nodes.ClassDef
) -> StatementResult:
    fields = _process_dataclass_like_fields(context, cdef, pytypes.NamedTupleBaseType)
    if fields is None:
        return []
    cls_loc = context.node_location(cdef)
    named_tuple_type = pytypes.NamedTupleType(
        name=cdef.fullname,
        desc=cdef.docstring,
        fields=fields,
        source_location=cls_loc,
    )
    context.register_pytype(named_tuple_type)
    return []


def _map_scratch_space_reservation(
    context: ASTConversionModuleContext,
    expr_visitor: nypy.visitor.ExpressionVisitor[ConstantValue],
    expr: nypy.nodes.Expression,
) -> list[int]:
    def get_int_arg(arg_expr: nypy.nodes.Expression, *, error_msg: str) -> int:
        const_value = arg_expr.accept(expr_visitor)
        if isinstance(const_value, int):
            return const_value
        raise CodeError(error_msg, context.node_location(arg_expr))

    expr_loc = context.node_location(expr)
    match expr:
        case nypy.nodes.CallExpr(
            callee=nypy.nodes.RefExpr() as callee, args=args
        ) if get_unaliased_fullname(callee) == constants.URANGE:
            if not args:
                raise CodeError("Expected at least one argument to urange", expr_loc)
            if len(args) > 3:
                raise CodeError("Expected at most three arguments to urange", expr_loc)
            int_args = [
                get_int_arg(
                    arg_expr,
                    error_msg=(
                        "Unexpected argument for urange:"
                        " expected an in integer literal or module constant reference"
                    ),
                )
                for arg_expr in args
            ]
            if len(int_args) == 3 and int_args[-1] == 0:
                raise CodeError("urange arg 3 must not be zero", context.node_location(args[-1]))
            return list(range(*int_args))
        case _:
            return [
                get_int_arg(
                    expr,
                    error_msg=(
                        "Unexpected value:"
                        " Expected int literal, module constant reference, or urange expression"
                    ),
                )
            ]


def _infer_condition_value(expr: nypy.nodes.Expression) -> int:
    """Infer whether the given condition is always true/false.

    Return ALWAYS_TRUE if always true, ALWAYS_FALSE if always false,
    MYPY_TRUE if true under mypy and false at runtime, MYPY_FALSE if
    false under mypy and true at runtime, else TRUTH_VALUE_UNKNOWN.
    """
    negated = False
    alias = expr
    if isinstance(alias, nypy.nodes.UnaryExpr) and alias.op == "not":
        expr = alias.expr
        negated = True
    result = _TRUTH_VALUE_UNKNOWN
    if isinstance(expr, nypy.nodes.NameExpr | nypy.nodes.MemberExpr):
        if expr.name == "TYPE_CHECKING":
            result = _TYPECHECKING_TRUE
    elif isinstance(expr, nypy.nodes.OpExpr) and expr.op in ("and", "or"):
        left = _infer_condition_value(expr.left)
        if (left in (_ALWAYS_TRUE, _TYPECHECKING_TRUE) and expr.op == "and") or (
            left in (_ALWAYS_FALSE, _TYPECHECKING_FALSE) and expr.op == "or"
        ):
            # Either `True and <other>` or `False or <other>`: the result will
            # always be the right-hand-side.
            return _infer_condition_value(expr.right)
        else:
            # The result will always be the left-hand-side (e.g. ALWAYS_* or
            # TRUTH_VALUE_UNKNOWN).
            return left
    if negated:
        result = inverted_truth_mapping[result]
    return result


# Inferred truth value of an expression.
_ALWAYS_TRUE: typing.Final = 1
_TYPECHECKING_TRUE: typing.Final = 2  # True when type-checking, False at runtime
_ALWAYS_FALSE: typing.Final = 3
_TYPECHECKING_FALSE: typing.Final = 4  # False when type-checking, True at runtime
_TRUTH_VALUE_UNKNOWN: typing.Final = 5

inverted_truth_mapping: typing.Final = {
    _ALWAYS_TRUE: _ALWAYS_FALSE,
    _ALWAYS_FALSE: _ALWAYS_TRUE,
    _TRUTH_VALUE_UNKNOWN: _TRUTH_VALUE_UNKNOWN,
    _TYPECHECKING_TRUE: _TYPECHECKING_FALSE,
    _TYPECHECKING_FALSE: _TYPECHECKING_TRUE,
}
