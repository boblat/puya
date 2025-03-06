"""Test cases for graph processing code in build.py."""

from __future__ import annotations

import sys
from collections.abc import Set as AbstractSet

from nypy.build import BuildManager, BuildSourceSet, State, order_ascc, sorted_components
from nypy.errors import Errors
from nypy.fscache import FileSystemCache
from nypy.graph_utils import strongly_connected_components, topsort
from nypy.modulefinder import SearchPaths
from nypy.options import Options
from nypy.plugin import Plugin
from nypy.report import Reports
from nypy.test.helpers import Suite, assert_equal
from nypy.version import __version__


class GraphSuite(Suite):
    def test_topsort(self) -> None:
        a = frozenset({"A"})
        b = frozenset({"B"})
        c = frozenset({"C"})
        d = frozenset({"D"})
        data: dict[AbstractSet[str], set[AbstractSet[str]]] = {a: {b, c}, b: {d}, c: {d}}
        res = list(topsort(data))
        assert_equal(res, [{d}, {b, c}, {a}])

    def test_scc(self) -> None:
        vertices = {"A", "B", "C", "D"}
        edges: dict[str, list[str]] = {"A": ["B", "C"], "B": ["C"], "C": ["B", "D"], "D": []}
        sccs = {frozenset(x) for x in strongly_connected_components(vertices, edges)}
        assert_equal(sccs, {frozenset({"A"}), frozenset({"B", "C"}), frozenset({"D"})})

    def _make_manager(self) -> BuildManager:
        options = Options()
        options.use_builtins_fixtures = True
        errors = Errors(options)
        fscache = FileSystemCache()
        search_paths = SearchPaths((), (), (), ())
        manager = BuildManager(
            data_dir="",
            search_paths=search_paths,
            ignore_prefix="",
            source_set=BuildSourceSet([]),
            reports=Reports("", {}),
            options=options,
            version_id=__version__,
            plugin=Plugin(options),
            plugins_snapshot={},
            errors=errors,
            flush_errors=lambda filename, msgs, serious: None,
            fscache=fscache,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        return manager

    def test_sorted_components(self) -> None:
        manager = self._make_manager()
        graph = {
            "a": State("a", None, "import b, c", manager),
            "d": State("d", None, "pass", manager),
            "b": State("b", None, "import c", manager),
            "c": State("c", None, "import b, d", manager),
        }
        res = sorted_components(graph)
        assert_equal(res, [frozenset({"d"}), frozenset({"c", "b"}), frozenset({"a"})])

    def test_order_ascc(self) -> None:
        manager = self._make_manager()
        graph = {
            "a": State("a", None, "import b, c", manager),
            "d": State("d", None, "def f(): import a", manager),
            "b": State("b", None, "import c", manager),
            "c": State("c", None, "import b, d", manager),
        }
        res = sorted_components(graph)
        assert_equal(res, [frozenset({"a", "d", "c", "b"})])
        ascc = res[0]
        scc = order_ascc(graph, ascc)
        assert_equal(scc, ["d", "c", "b", "a"])
