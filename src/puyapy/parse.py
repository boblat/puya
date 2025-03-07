from __future__ import annotations

import codecs
import enum
import os
import re
import shutil
import subprocess
import sysconfig
import typing
from collections.abc import Sequence
from functools import cached_property
from importlib import metadata
from pathlib import Path

import attrs
import docstring_parser
from packaging import version

import nypy.build
import nypy.errors
import nypy.find_sources
import nypy.fscache
import nypy.modulefinder
import nypy.nodes
import nypy.options
import nypy.util
from puya import log
from puya.awst.nodes import MethodDocumentation
from puya.parse import SourceLocation
from puya.utils import make_path_relative_to_cwd

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set


_PUYAPY_SRC_ROOT = Path(__file__).parent
_PUYA_SRC_ROOT = _PUYAPY_SRC_ROOT.parent / "puya"
TYPESHED_PATH = _PUYAPY_SRC_ROOT / "_typeshed"
_MYPY_FSCACHE = nypy.fscache.FileSystemCache()
_MYPY_SEVERITY_TO_LOG_LEVEL = {
    "error": log.LogLevel.error,
    "warning": log.LogLevel.warning,
    "note": log.LogLevel.info,
}

_MAX_SUPPORTED_ALGOPY_VERSION_EX = version.parse("2.8.0")
_MIN_SUPPORTED_ALGOPY_VERSION = version.parse(f"{_MAX_SUPPORTED_ALGOPY_VERSION_EX.major}.0.0")

logger = log.get_logger(__name__)


@attrs.frozen
class EmbeddedSource:
    path: Path
    mypy_module_name: str
    puya_module_name: str

    @classmethod
    def from_path(cls, filename: str, *, module_override: str | None = None) -> typing.Self:
        path = _PUYA_SRC_ROOT / "lib_embedded" / filename
        return cls(
            path=path,
            mypy_module_name=path.stem,
            puya_module_name=module_override or path.stem,
        )


class SourceDiscoveryMechanism(enum.Enum):
    explicit_file = enum.auto()
    explicit_directory_walk = enum.auto()
    dependency = enum.auto()


@attrs.frozen
class SourceModule:
    name: str
    node: nypy.nodes.MypyFile
    path: Path
    lines: Sequence[str] | None
    discovery_mechanism: SourceDiscoveryMechanism


@attrs.frozen
class ParseResult:
    mypy_options: nypy.options.Options
    ordered_modules: Mapping[str, SourceModule]
    """All discovered modules, topologically sorted by dependencies.
    The sort order is from leaves (nodes without dependencies) to
    roots (nodes on which no other nodes depend)."""

    @cached_property
    def sources_by_path(self) -> Mapping[Path, Sequence[str] | None]:
        return {s.path: s.lines for s in self.ordered_modules.values()}

    @cached_property
    def explicit_source_paths(self) -> Set[Path]:
        return {
            sm.path
            for sm in self.ordered_modules.values()
            if sm.discovery_mechanism != SourceDiscoveryMechanism.dependency
        }


def parse_and_typecheck(paths: Sequence[Path]) -> ParseResult:
    """Generate the ASTs from the build sources, and all imported modules (recursively)"""

    mypy_options = get_mypy_options()

    # ensure we have the absolute, canonical paths to the files
    resolved_input_paths = {p.resolve() for p in paths}
    # creates a list of BuildSource objects from the contract Paths
    mypy_build_sources = nypy.find_sources.create_source_list(
        paths=[str(p) for p in resolved_input_paths],
        options=mypy_options,
        fscache=_MYPY_FSCACHE,
    )
    build_source_paths = {
        Path(m.path).resolve() for m in mypy_build_sources if m.path and not m.followed
    }
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
    for scc_module_names in nypy.build.sorted_components(result.graph):
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
                _check_encoding(_MYPY_FSCACHE, module_path)
                lines = nypy.util.read_py_file(str(module_path), _MYPY_FSCACHE.read)
                if module_path in resolved_input_paths:
                    discovery_mechanism = SourceDiscoveryMechanism.explicit_file
                elif module_path in build_source_paths:
                    discovery_mechanism = SourceDiscoveryMechanism.explicit_directory_walk
                else:
                    discovery_mechanism = SourceDiscoveryMechanism.dependency
                ordered_modules[module_name] = SourceModule(
                    name=module_name,
                    node=module,
                    path=module_path,
                    lines=lines,
                    discovery_mechanism=discovery_mechanism,
                )

    return ParseResult(
        mypy_options=mypy_options,
        ordered_modules=ordered_modules,
    )


def _check_encoding(mypy_fscache: nypy.fscache.FileSystemCache, module_path: Path) -> None:
    module_rel_path = make_path_relative_to_cwd(module_path)
    module_loc = SourceLocation(file=module_path, line=1)
    try:
        source = mypy_fscache.read(str(module_path))
    except OSError:
        logger.warning(
            f"Couldn't read source for {module_rel_path}",
            location=module_loc,
        )
        return
    # below is based on mypy/util.py:decode_python_encoding
    # check for BOM UTF-8 encoding
    if source.startswith(b"\xef\xbb\xbf"):
        return
    # otherwise look at first two lines and check if PEP-263 coding is present
    encoding, _ = nypy.util.find_python_encoding(source)
    # find the codec for this encoding and check it is utf-8
    codec = codecs.lookup(encoding)
    if codec.name != "utf-8":
        logger.warning(
            "UH OH SPAGHETTI-O's,"
            " darn tootin' non-utf8(?!) encoded file encountered:"
            f" {module_rel_path} encoded as {encoding}",
            location=module_loc,
        )


def _mypy_build(
    sources: list[nypy.modulefinder.BuildSource],
    options: nypy.options.Options,
    fscache: nypy.fscache.FileSystemCache | None,
) -> nypy.build.BuildResult:
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
        result = nypy.build.build(
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


_MYPY_LOG_MESSAGE = re.compile(
    r"""^
    (?P<filename>([A-Z]:\\)?[^:]*)(:(?P<line>\d+))?
    :\s(?P<severity>error|warning|note)
    :\s(?P<msg>.*)$""",
    re.VERBOSE,
)


def _parse_log_message(log_message: str) -> log.Log:
    match = _MYPY_LOG_MESSAGE.match(log_message)
    if match:
        matches = match.groupdict()
        return _try_parse_log_parts(
            matches.get("filename"),
            matches.get("line") or "",
            matches.get("severity") or "error",
            matches["msg"],
        )
    return log.Log(
        level=log.LogLevel.error,
        message=log_message,
        location=None,
    )


def _try_parse_log_parts(
    path_str: str | None, line_str: str, severity_str: str, msg: str
) -> log.Log:
    if not path_str:
        location = None
    else:
        try:
            line = int(line_str)
        except ValueError:
            line = 1
        location = SourceLocation(file=Path(path_str).resolve(), line=line)
    level = _MYPY_SEVERITY_TO_LOG_LEVEL[severity_str]
    return log.Log(message=msg, level=level, location=location)


def _join_single_new_line(doc: str) -> str:
    return doc.strip().replace("\n", " ")


def parse_docstring(docstring_raw: str | None) -> MethodDocumentation:
    if docstring_raw is None:
        return MethodDocumentation()
    docstring = docstring_parser.parse(docstring_raw)
    method_desc = "\n".join(
        _join_single_new_line(line)
        for lines in filter(None, (docstring.short_description, docstring.long_description))
        for line in lines.split("\n\n")
    )
    return MethodDocumentation(
        description=method_desc if method_desc else None,
        args={
            p.arg_name: _join_single_new_line(p.description)
            for p in docstring.params
            if p.description
        },
        returns=(
            _join_single_new_line(docstring.returns.description)
            if docstring.returns and docstring.returns.description
            else None
        ),
    )


def source_location_from_mypy(file: Path, node: nypy.nodes.Context) -> SourceLocation:
    assert node.line is not None
    assert node.line >= 1

    match node:
        case (
            nypy.nodes.FuncDef(body=body)
            | nypy.nodes.Decorator(func=nypy.nodes.FuncDef(body=body))
        ):
            # end_line of a function node includes the entire body
            # try to get just the signature
            end_line = node.line
            # no body means the end_line is ok to use
            if body is None:
                end_line = max(end_line, node.end_line or node.line)
            # if there is a body, attempt to use the first line before the body as the end
            else:
                end_line = max(end_line, body.line - 1)
            return SourceLocation(
                file=file,
                line=node.line,
                end_line=end_line,
            )
        case nypy.nodes.ClassDef(decorators=class_decorators, defs=class_body):
            line = node.line
            for dec in class_decorators:
                line = min(dec.line, line)
            end_line = max(line, class_body.line - 1)
            return SourceLocation(
                file=file,
                line=line,
                end_line=end_line,
            )
        case nypy.nodes.WhileStmt(body=compound_body) | nypy.nodes.ForStmt(body=compound_body):
            return SourceLocation(
                file=file,
                line=node.line,
                end_line=compound_body.line - 1,
            )
        case nypy.nodes.IfStmt(body=[*bodies], else_body=else_body):
            body_start: int | None = None
            if else_body is not None:
                bodies.append(else_body)
            for body in bodies:
                if body_start is None:
                    body_start = body.line
                else:
                    body_start = min(body_start, body.line)
            if body_start is None:
                # this shouldn't happen, there should be at least one body in one branch,
                # but this serves okay as a fallback
                end_line = node.end_line or node.line
            else:
                end_line = body_start - 1
            return SourceLocation(
                file=file,
                line=node.line,
                end_line=end_line,
            )
    return SourceLocation(
        file=file,
        line=node.line,
        end_line=(
            node.end_line if (node.end_line is not None and node.end_line >= 1) else node.line
        ),
        column=node.column if (node.column is not None and node.column >= 0) else 0,
        end_column=(
            node.end_column if (node.end_column is not None and node.end_column >= 0) else None
        ),
    )


def get_mypy_options() -> nypy.options.Options:
    mypy_opts = nypy.options.Options()

    # improve mypy parsing performance by using a cut-down typeshed
    mypy_opts.custom_typeshed_dir = str(TYPESHED_PATH)
    mypy_opts.abs_custom_typeshed_dir = str(TYPESHED_PATH.resolve())

    # set python_executable so third-party packages can be found
    mypy_opts.python_executable = _get_python_executable()

    mypy_opts.include_docstrings = True

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
    mypy_opts.disallow_any_expr = False  # this is broken for tuples
    mypy_opts.disallow_any_decorated = True
    mypy_opts.disallow_any_explicit = True

    mypy_opts.pretty = True  # show source in output

    return mypy_opts


def _get_python_executable() -> str | None:
    prefix = _get_prefix()
    if not prefix:
        logger.warning("could not determine python prefix or algopy version")
        return None
    logger.info(f"found python prefix: {prefix}")
    install_paths = sysconfig.get_paths(vars={"base": prefix})

    python_exe = None
    for python in ("python3", "python"):
        python_exe = shutil.which(python, path=install_paths["scripts"])
        if python_exe:
            logger.debug(f"using python executable: {python_exe}")
            break
    else:
        logger.warning("found a python prefix, but could not find the expected python interpreter")
    # use glob here, as we don't want to assume the python version
    discovered_site_packages = list(
        Path(prefix).glob(str(Path("[Ll]ib") / "**" / "site-packages"))
    )
    try:
        (site_packages,) = discovered_site_packages
    except ValueError:
        logger.warning(
            "found a prefix, but could not find the expected"
            f" site-packages: {prefix=}, {discovered_site_packages=}"
        )
    else:
        logger.debug(f"using python site-packages: {site_packages}")
        _check_algopy_version(site_packages)

    return python_exe


def _get_prefix() -> str | None:
    # look for VIRTUAL_ENV as we want the venv puyapy is being run against (i.e. the project),
    # if no venv is active, then fallback to the ambient python prefix
    venv = os.getenv("VIRTUAL_ENV")
    if venv:
        return venv
    for python in ("python3", "python"):
        prefix_result = subprocess.run(  # noqa: S602
            f"{python} -c 'import sys; print(sys.prefix or sys.base_prefix)'",
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        if prefix_result.returncode == 0 and (maybe_prefix := prefix_result.stdout.strip()):
            return maybe_prefix
    return None


_STUBS_PACKAGE_NAME = "algorand-python"


def _check_algopy_version(site_packages: Path) -> None:
    pkgs = metadata.Distribution.discover(name=_STUBS_PACKAGE_NAME, path=[str(site_packages)])
    try:
        (algopy,) = pkgs
    except ValueError:
        logger.warning("Could not determine algopy version")
        return
    algopy_version = version.parse(algopy.version)
    logger.debug(f"found algopy: {algopy_version}")

    if not (_MIN_SUPPORTED_ALGOPY_VERSION <= algopy_version < _MAX_SUPPORTED_ALGOPY_VERSION_EX):
        logger.warning(
            f"{_STUBS_PACKAGE_NAME} version {algopy_version} is outside the supported range:"
            f" >={_MIN_SUPPORTED_ALGOPY_VERSION}, <{_MAX_SUPPORTED_ALGOPY_VERSION_EX}",
            important=True,
            related_lines=[
                "This will cause typing errors if there are incompatibilities in the API used.",
                "Please update your algorand-python package to be in the supported range.",
            ],
        )
