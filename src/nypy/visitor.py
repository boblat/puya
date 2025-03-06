"""Generic abstract syntax tree node visitor"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from mypy_extensions import mypyc_attr, trait

if TYPE_CHECKING:
    # break import cycle only needed for mypy
    import nypy.nodes
    import nypy.patterns


T = TypeVar("T")


@trait
@mypyc_attr(allow_interpreted_subclasses=True)
class ExpressionVisitor(Generic[T]):
    @abstractmethod
    def visit_int_expr(self, o: nypy.nodes.IntExpr, /) -> T:
        pass

    @abstractmethod
    def visit_str_expr(self, o: nypy.nodes.StrExpr, /) -> T:
        pass

    @abstractmethod
    def visit_bytes_expr(self, o: nypy.nodes.BytesExpr, /) -> T:
        pass

    @abstractmethod
    def visit_float_expr(self, o: nypy.nodes.FloatExpr, /) -> T:
        pass

    @abstractmethod
    def visit_complex_expr(self, o: nypy.nodes.ComplexExpr, /) -> T:
        pass

    @abstractmethod
    def visit_ellipsis(self, o: nypy.nodes.EllipsisExpr, /) -> T:
        pass

    @abstractmethod
    def visit_star_expr(self, o: nypy.nodes.StarExpr, /) -> T:
        pass

    @abstractmethod
    def visit_name_expr(self, o: nypy.nodes.NameExpr, /) -> T:
        pass

    @abstractmethod
    def visit_member_expr(self, o: nypy.nodes.MemberExpr, /) -> T:
        pass

    @abstractmethod
    def visit_yield_from_expr(self, o: nypy.nodes.YieldFromExpr, /) -> T:
        pass

    @abstractmethod
    def visit_yield_expr(self, o: nypy.nodes.YieldExpr, /) -> T:
        pass

    @abstractmethod
    def visit_call_expr(self, o: nypy.nodes.CallExpr, /) -> T:
        pass

    @abstractmethod
    def visit_op_expr(self, o: nypy.nodes.OpExpr, /) -> T:
        pass

    @abstractmethod
    def visit_comparison_expr(self, o: nypy.nodes.ComparisonExpr, /) -> T:
        pass

    @abstractmethod
    def visit_cast_expr(self, o: nypy.nodes.CastExpr, /) -> T:
        pass

    @abstractmethod
    def visit_assert_type_expr(self, o: nypy.nodes.AssertTypeExpr, /) -> T:
        pass

    @abstractmethod
    def visit_reveal_expr(self, o: nypy.nodes.RevealExpr, /) -> T:
        pass

    @abstractmethod
    def visit_super_expr(self, o: nypy.nodes.SuperExpr, /) -> T:
        pass

    @abstractmethod
    def visit_unary_expr(self, o: nypy.nodes.UnaryExpr, /) -> T:
        pass

    @abstractmethod
    def visit_assignment_expr(self, o: nypy.nodes.AssignmentExpr, /) -> T:
        pass

    @abstractmethod
    def visit_list_expr(self, o: nypy.nodes.ListExpr, /) -> T:
        pass

    @abstractmethod
    def visit_dict_expr(self, o: nypy.nodes.DictExpr, /) -> T:
        pass

    @abstractmethod
    def visit_tuple_expr(self, o: nypy.nodes.TupleExpr, /) -> T:
        pass

    @abstractmethod
    def visit_set_expr(self, o: nypy.nodes.SetExpr, /) -> T:
        pass

    @abstractmethod
    def visit_index_expr(self, o: nypy.nodes.IndexExpr, /) -> T:
        pass

    @abstractmethod
    def visit_type_application(self, o: nypy.nodes.TypeApplication, /) -> T:
        pass

    @abstractmethod
    def visit_lambda_expr(self, o: nypy.nodes.LambdaExpr, /) -> T:
        pass

    @abstractmethod
    def visit_list_comprehension(self, o: nypy.nodes.ListComprehension, /) -> T:
        pass

    @abstractmethod
    def visit_set_comprehension(self, o: nypy.nodes.SetComprehension, /) -> T:
        pass

    @abstractmethod
    def visit_dictionary_comprehension(self, o: nypy.nodes.DictionaryComprehension, /) -> T:
        pass

    @abstractmethod
    def visit_generator_expr(self, o: nypy.nodes.GeneratorExpr, /) -> T:
        pass

    @abstractmethod
    def visit_slice_expr(self, o: nypy.nodes.SliceExpr, /) -> T:
        pass

    @abstractmethod
    def visit_conditional_expr(self, o: nypy.nodes.ConditionalExpr, /) -> T:
        pass

    @abstractmethod
    def visit_type_var_expr(self, o: nypy.nodes.TypeVarExpr, /) -> T:
        pass

    @abstractmethod
    def visit_paramspec_expr(self, o: nypy.nodes.ParamSpecExpr, /) -> T:
        pass

    @abstractmethod
    def visit_type_var_tuple_expr(self, o: nypy.nodes.TypeVarTupleExpr, /) -> T:
        pass

    @abstractmethod
    def visit_type_alias_expr(self, o: nypy.nodes.TypeAliasExpr, /) -> T:
        pass

    @abstractmethod
    def visit_namedtuple_expr(self, o: nypy.nodes.NamedTupleExpr, /) -> T:
        pass

    @abstractmethod
    def visit_enum_call_expr(self, o: nypy.nodes.EnumCallExpr, /) -> T:
        pass

    @abstractmethod
    def visit_typeddict_expr(self, o: nypy.nodes.TypedDictExpr, /) -> T:
        pass

    @abstractmethod
    def visit_newtype_expr(self, o: nypy.nodes.NewTypeExpr, /) -> T:
        pass

    @abstractmethod
    def visit__promote_expr(self, o: nypy.nodes.PromoteExpr, /) -> T:
        pass

    @abstractmethod
    def visit_await_expr(self, o: nypy.nodes.AwaitExpr, /) -> T:
        pass

    @abstractmethod
    def visit_temp_node(self, o: nypy.nodes.TempNode, /) -> T:
        pass


@trait
@mypyc_attr(allow_interpreted_subclasses=True)
class StatementVisitor(Generic[T]):
    # Definitions

    @abstractmethod
    def visit_assignment_stmt(self, o: nypy.nodes.AssignmentStmt, /) -> T:
        pass

    @abstractmethod
    def visit_for_stmt(self, o: nypy.nodes.ForStmt, /) -> T:
        pass

    @abstractmethod
    def visit_with_stmt(self, o: nypy.nodes.WithStmt, /) -> T:
        pass

    @abstractmethod
    def visit_del_stmt(self, o: nypy.nodes.DelStmt, /) -> T:
        pass

    @abstractmethod
    def visit_func_def(self, o: nypy.nodes.FuncDef, /) -> T:
        pass

    @abstractmethod
    def visit_overloaded_func_def(self, o: nypy.nodes.OverloadedFuncDef, /) -> T:
        pass

    @abstractmethod
    def visit_class_def(self, o: nypy.nodes.ClassDef, /) -> T:
        pass

    @abstractmethod
    def visit_global_decl(self, o: nypy.nodes.GlobalDecl, /) -> T:
        pass

    @abstractmethod
    def visit_nonlocal_decl(self, o: nypy.nodes.NonlocalDecl, /) -> T:
        pass

    @abstractmethod
    def visit_decorator(self, o: nypy.nodes.Decorator, /) -> T:
        pass

    # Module structure

    @abstractmethod
    def visit_import(self, o: nypy.nodes.Import, /) -> T:
        pass

    @abstractmethod
    def visit_import_from(self, o: nypy.nodes.ImportFrom, /) -> T:
        pass

    @abstractmethod
    def visit_import_all(self, o: nypy.nodes.ImportAll, /) -> T:
        pass

    # Statements

    @abstractmethod
    def visit_block(self, o: nypy.nodes.Block, /) -> T:
        pass

    @abstractmethod
    def visit_expression_stmt(self, o: nypy.nodes.ExpressionStmt, /) -> T:
        pass

    @abstractmethod
    def visit_operator_assignment_stmt(self, o: nypy.nodes.OperatorAssignmentStmt, /) -> T:
        pass

    @abstractmethod
    def visit_while_stmt(self, o: nypy.nodes.WhileStmt, /) -> T:
        pass

    @abstractmethod
    def visit_return_stmt(self, o: nypy.nodes.ReturnStmt, /) -> T:
        pass

    @abstractmethod
    def visit_assert_stmt(self, o: nypy.nodes.AssertStmt, /) -> T:
        pass

    @abstractmethod
    def visit_if_stmt(self, o: nypy.nodes.IfStmt, /) -> T:
        pass

    @abstractmethod
    def visit_break_stmt(self, o: nypy.nodes.BreakStmt, /) -> T:
        pass

    @abstractmethod
    def visit_continue_stmt(self, o: nypy.nodes.ContinueStmt, /) -> T:
        pass

    @abstractmethod
    def visit_pass_stmt(self, o: nypy.nodes.PassStmt, /) -> T:
        pass

    @abstractmethod
    def visit_raise_stmt(self, o: nypy.nodes.RaiseStmt, /) -> T:
        pass

    @abstractmethod
    def visit_try_stmt(self, o: nypy.nodes.TryStmt, /) -> T:
        pass

    @abstractmethod
    def visit_match_stmt(self, o: nypy.nodes.MatchStmt, /) -> T:
        pass

    @abstractmethod
    def visit_type_alias_stmt(self, o: nypy.nodes.TypeAliasStmt, /) -> T:
        pass


@trait
@mypyc_attr(allow_interpreted_subclasses=True)
class PatternVisitor(Generic[T]):
    @abstractmethod
    def visit_as_pattern(self, o: nypy.patterns.AsPattern, /) -> T:
        pass

    @abstractmethod
    def visit_or_pattern(self, o: nypy.patterns.OrPattern, /) -> T:
        pass

    @abstractmethod
    def visit_value_pattern(self, o: nypy.patterns.ValuePattern, /) -> T:
        pass

    @abstractmethod
    def visit_singleton_pattern(self, o: nypy.patterns.SingletonPattern, /) -> T:
        pass

    @abstractmethod
    def visit_sequence_pattern(self, o: nypy.patterns.SequencePattern, /) -> T:
        pass

    @abstractmethod
    def visit_starred_pattern(self, o: nypy.patterns.StarredPattern, /) -> T:
        pass

    @abstractmethod
    def visit_mapping_pattern(self, o: nypy.patterns.MappingPattern, /) -> T:
        pass

    @abstractmethod
    def visit_class_pattern(self, o: nypy.patterns.ClassPattern, /) -> T:
        pass


@trait
@mypyc_attr(allow_interpreted_subclasses=True)
class NodeVisitor(Generic[T], ExpressionVisitor[T], StatementVisitor[T], PatternVisitor[T]):
    """Empty base class for parse tree node visitors.

    The T type argument specifies the return type of the visit
    methods. As all methods defined here return None by default,
    subclasses do not always need to override all the methods.

    TODO: make the default return value explicit, then turn on
          empty body checking in mypy_self_check.ini.
    """

    # Not in superclasses:

    def visit_mypy_file(self, o: nypy.nodes.MypyFile, /) -> T:  # type: ignore[empty-body]
        pass

    # TODO: We have a visit_var method, but no visit_typeinfo or any
    # other non-Statement SymbolNode (accepting those will raise a
    # runtime error). Maybe this should be resolved in some direction.
    def visit_var(self, o: nypy.nodes.Var, /) -> T:  # type: ignore[empty-body]
        pass

    # Module structure

    def visit_import(self, o: nypy.nodes.Import, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_import_from(self, o: nypy.nodes.ImportFrom, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_import_all(self, o: nypy.nodes.ImportAll, /) -> T:  # type: ignore[empty-body]
        pass

    # Definitions

    def visit_func_def(self, o: nypy.nodes.FuncDef, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_overloaded_func_def(self, o: nypy.nodes.OverloadedFuncDef, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_class_def(self, o: nypy.nodes.ClassDef, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_global_decl(self, o: nypy.nodes.GlobalDecl, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_nonlocal_decl(self, o: nypy.nodes.NonlocalDecl, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_decorator(self, o: nypy.nodes.Decorator, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_type_alias(self, o: nypy.nodes.TypeAlias, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_placeholder_node(self, o: nypy.nodes.PlaceholderNode, /) -> T:  # type: ignore[empty-body]
        pass

    # Statements

    def visit_block(self, o: nypy.nodes.Block, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_expression_stmt(self, o: nypy.nodes.ExpressionStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_assignment_stmt(self, o: nypy.nodes.AssignmentStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_operator_assignment_stmt(self, o: nypy.nodes.OperatorAssignmentStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_while_stmt(self, o: nypy.nodes.WhileStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_for_stmt(self, o: nypy.nodes.ForStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_return_stmt(self, o: nypy.nodes.ReturnStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_assert_stmt(self, o: nypy.nodes.AssertStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_del_stmt(self, o: nypy.nodes.DelStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_if_stmt(self, o: nypy.nodes.IfStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_break_stmt(self, o: nypy.nodes.BreakStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_continue_stmt(self, o: nypy.nodes.ContinueStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_pass_stmt(self, o: nypy.nodes.PassStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_raise_stmt(self, o: nypy.nodes.RaiseStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_try_stmt(self, o: nypy.nodes.TryStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_with_stmt(self, o: nypy.nodes.WithStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_match_stmt(self, o: nypy.nodes.MatchStmt, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_type_alias_stmt(self, o: nypy.nodes.TypeAliasStmt, /) -> T:  # type: ignore[empty-body]
        pass

    # Expressions (default no-op implementation)

    def visit_int_expr(self, o: nypy.nodes.IntExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_str_expr(self, o: nypy.nodes.StrExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_bytes_expr(self, o: nypy.nodes.BytesExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_float_expr(self, o: nypy.nodes.FloatExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_complex_expr(self, o: nypy.nodes.ComplexExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_ellipsis(self, o: nypy.nodes.EllipsisExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_star_expr(self, o: nypy.nodes.StarExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_name_expr(self, o: nypy.nodes.NameExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_member_expr(self, o: nypy.nodes.MemberExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_yield_from_expr(self, o: nypy.nodes.YieldFromExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_yield_expr(self, o: nypy.nodes.YieldExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_call_expr(self, o: nypy.nodes.CallExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_op_expr(self, o: nypy.nodes.OpExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_comparison_expr(self, o: nypy.nodes.ComparisonExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_cast_expr(self, o: nypy.nodes.CastExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_assert_type_expr(self, o: nypy.nodes.AssertTypeExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_reveal_expr(self, o: nypy.nodes.RevealExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_super_expr(self, o: nypy.nodes.SuperExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_assignment_expr(self, o: nypy.nodes.AssignmentExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_unary_expr(self, o: nypy.nodes.UnaryExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_list_expr(self, o: nypy.nodes.ListExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_dict_expr(self, o: nypy.nodes.DictExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_tuple_expr(self, o: nypy.nodes.TupleExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_set_expr(self, o: nypy.nodes.SetExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_index_expr(self, o: nypy.nodes.IndexExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_type_application(self, o: nypy.nodes.TypeApplication, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_lambda_expr(self, o: nypy.nodes.LambdaExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_list_comprehension(self, o: nypy.nodes.ListComprehension, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_set_comprehension(self, o: nypy.nodes.SetComprehension, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_dictionary_comprehension(self, o: nypy.nodes.DictionaryComprehension, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_generator_expr(self, o: nypy.nodes.GeneratorExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_slice_expr(self, o: nypy.nodes.SliceExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_conditional_expr(self, o: nypy.nodes.ConditionalExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_type_var_expr(self, o: nypy.nodes.TypeVarExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_paramspec_expr(self, o: nypy.nodes.ParamSpecExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_type_var_tuple_expr(self, o: nypy.nodes.TypeVarTupleExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_type_alias_expr(self, o: nypy.nodes.TypeAliasExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_namedtuple_expr(self, o: nypy.nodes.NamedTupleExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_enum_call_expr(self, o: nypy.nodes.EnumCallExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_typeddict_expr(self, o: nypy.nodes.TypedDictExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_newtype_expr(self, o: nypy.nodes.NewTypeExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit__promote_expr(self, o: nypy.nodes.PromoteExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_await_expr(self, o: nypy.nodes.AwaitExpr, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_temp_node(self, o: nypy.nodes.TempNode, /) -> T:  # type: ignore[empty-body]
        pass

    # Patterns

    def visit_as_pattern(self, o: nypy.patterns.AsPattern, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_or_pattern(self, o: nypy.patterns.OrPattern, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_value_pattern(self, o: nypy.patterns.ValuePattern, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_singleton_pattern(self, o: nypy.patterns.SingletonPattern, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_sequence_pattern(self, o: nypy.patterns.SequencePattern, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_starred_pattern(self, o: nypy.patterns.StarredPattern, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_mapping_pattern(self, o: nypy.patterns.MappingPattern, /) -> T:  # type: ignore[empty-body]
        pass

    def visit_class_pattern(self, o: nypy.patterns.ClassPattern, /) -> T:  # type: ignore[empty-body]
        pass
