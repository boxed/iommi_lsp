"""Multi-root: one analyzer bundle per workspace folder, routed by URI,
plus each folder's virtualenv handed to ty through workspace/configuration."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from iommi_lsp.cli import ANALYZER_CLASSES, _build_analyzers
from iommi_lsp.interceptor import (
    DiagnosticInterceptor,
    DocumentStore,
    EditorRequestSniffer,
    WorkspaceConfigAugmenter,
)
from iommi_lsp.parse_cache import ParseCache
from iommi_lsp.workspaces import WorkspaceSet, workspace_venv

from tests.test_filter_e2e import _diag_at, _frame, _read_frame


HERE = Path(__file__).parent
BASIC = HERE / "corpus" / "basic_django"


class _Recorder:
    """A bundle-slot stand-in that remembers which root answered."""

    name = "recorder"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.indexed = False

    async def index(self, workspace_root: Path) -> None:
        self.indexed = True

    async def on_file_changed(self, uri: str) -> None:
        pass

    def is_false_positive(self, uri: str, diagnostic: dict) -> bool:
        return False

    def completions(self, uri: str, position: dict):
        return self.root


class _NoCompletions(_Recorder):
    completions = None  # type: ignore[assignment]


def _recorder_set() -> tuple[WorkspaceSet, list]:
    ws = WorkspaceSet(lambda root: [_Recorder(root)])
    return ws, ws.routed([_Recorder])


@pytest.mark.asyncio
async def test_calls_route_to_the_folder_holding_the_file(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ws, (routed,) = _recorder_set()
    await ws.add(a)
    await ws.add(b)
    assert routed.completions((a / "x.py").as_uri(), {}) == a
    assert routed.completions((b / "pkg" / "y.py").as_uri(), {}) == b


@pytest.mark.asyncio
async def test_a_file_outside_every_folder_goes_to_the_primary(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ws, (routed,) = _recorder_set()
    await ws.add(a)
    await ws.add(b)
    assert routed.completions((tmp_path / "site-packages" / "z.py").as_uri(), {}) == a
    assert routed.completions("untitled:Untitled-1", {}) == a


@pytest.mark.asyncio
async def test_a_nested_folder_wins_over_its_parent(tmp_path):
    outer, inner = tmp_path / "outer", tmp_path / "outer" / "inner"
    ws, (routed,) = _recorder_set()
    await ws.add(outer)
    await ws.add(inner)
    assert routed.completions((inner / "x.py").as_uri(), {}) == inner
    assert routed.completions((outer / "x.py").as_uri(), {}) == outer


@pytest.mark.asyncio
async def test_removing_the_primary_promotes_the_next_folder(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ws, (routed,) = _recorder_set()
    await ws.add(a)
    await ws.add(b)
    outside = (tmp_path / "elsewhere.py").as_uri()
    assert routed.completions(outside, {}) == a
    ws.remove(a)
    assert ws.roots == [b]
    assert routed.completions((a / "x.py").as_uri(), {}) == b
    assert routed.completions(outside, {}) == b


@pytest.mark.asyncio
async def test_adding_an_open_folder_again_keeps_its_bundle(tmp_path):
    built: list[Path] = []

    def factory(root):
        built.append(root)
        return [_Recorder(root)]

    ws = WorkspaceSet(factory)
    await ws.add(tmp_path)
    await ws.add(tmp_path)
    assert built == [tmp_path]


def test_with_no_folder_open_routed_calls_are_neutral(tmp_path):
    ws, (routed,) = _recorder_set()
    uri = (tmp_path / "x.py").as_uri()
    assert routed.completions(uri, {}) is None
    assert routed.is_false_positive(uri, {}) is False


def test_routed_slot_exposes_only_the_methods_its_class_has():
    ws = WorkspaceSet(lambda root: [_Recorder(root)])
    (routed,) = ws.routed([_Recorder])
    assert getattr(routed, "completions", None) is not None
    # The routers probe optional hooks with getattr(..., None).
    assert getattr(routed, "resolve_definition", None) is None
    assert getattr(routed, "code_actions", None) is None


def test_the_factory_builds_slots_in_the_routed_order(tmp_path):
    store = DocumentStore()
    bundle = _build_analyzers(tmp_path, store.get, ParseCache().get)
    assert tuple(type(a) for a in bundle) == ANALYZER_CLASSES


@pytest.mark.asyncio
async def test_each_folder_filters_against_its_own_models(tmp_path):
    """``Profile.objects`` is a known false positive only where ``Profile``
    is a model: in ``basic_django``, not in a second project without one.
    A single shared index would filter both."""
    probe = "from myapp.models import Profile\n\n\n\nqs = Profile.objects.all()\n"
    basic = tmp_path / "basic"
    shutil.copytree(BASIC, basic)
    other = tmp_path / "other"
    other.mkdir()
    for root in (basic, other):
        (root / "probe.py").write_text(probe)

    store = DocumentStore()
    cache = ParseCache()
    ws = WorkspaceSet(lambda r: _build_analyzers(r, store.get, cache.get))
    await ws.add(basic)
    await ws.add(other)
    interceptor = DiagnosticInterceptor(analyzers=ws.routed(ANALYZER_CLASSES))

    async def kept(path: Path) -> list[str]:
        diag = _diag_at(path.read_text(), line=4, needle="objects")
        out = await interceptor(json.dumps({
            "jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
            "params": {"uri": path.as_uri(), "diagnostics": [diag]},
        }).encode())
        return [d["message"] for d in json.loads(out)["params"]["diagnostics"]]

    assert await kept(basic / "probe.py") == []
    assert await kept(other / "probe.py") == ["unresolved attribute 'objects'"]


# --- sniffer -----------------------------------------------------------------


def _sniffer() -> tuple[EditorRequestSniffer, list, list]:
    added: list[Path] = []
    removed: list[Path] = []

    async def on_add(root: Path) -> None:
        added.append(root)

    sniffer = EditorRequestSniffer(on_workspace=on_add, on_workspace_removed=removed.append)
    return sniffer, added, removed


@pytest.mark.asyncio
async def test_initialize_opens_every_workspace_folder():
    sniffer, added, _ = _sniffer()
    await sniffer(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"workspaceFolders": [
            {"uri": "file:///tmp/a", "name": "a"},
            {"uri": "file:///tmp/b", "name": "b"},
        ]},
    }).encode())
    await asyncio.sleep(0)
    assert added == [Path("/tmp/a"), Path("/tmp/b")]


@pytest.mark.asyncio
async def test_workspace_folder_changes_open_and_close_folders():
    sniffer, added, removed = _sniffer()
    body = json.dumps({
        "jsonrpc": "2.0", "method": "workspace/didChangeWorkspaceFolders",
        "params": {"event": {
            "added": [{"uri": "file:///tmp/c", "name": "c"}],
            "removed": [{"uri": "file:///tmp/a", "name": "a"}],
        }},
    }).encode()
    # Forwarded untouched: ty needs the same notification.
    assert await sniffer(body) is body
    await asyncio.sleep(0)
    assert added == [Path("/tmp/c")]
    assert removed == [Path("/tmp/a")]


# --- per-folder virtualenv ---------------------------------------------------


def _venv(root: Path, name: str = "venv") -> Path:
    (root / name / "bin").mkdir(parents=True)
    (root / name / "bin" / "python").write_text("")
    return root / name


def test_workspace_venv_finds_dot_venv_or_venv(tmp_path):
    assert workspace_venv(tmp_path) is None
    assert workspace_venv(tmp_path) is None
    venv = _venv(tmp_path, ".venv")
    assert workspace_venv(tmp_path) == venv


async def _round_trip(augmenter, items: list[dict], result: list) -> list:
    await augmenter.on_server_message(json.dumps({
        "jsonrpc": "2.0", "id": 7, "method": "workspace/configuration",
        "params": {"items": items},
    }).encode())
    out = await augmenter.on_editor_message(json.dumps({
        "jsonrpc": "2.0", "id": 7, "result": result,
    }).encode())
    return json.loads(out)["result"]


@pytest.mark.asyncio
async def test_configuration_answers_carry_each_folders_venv(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    venv_a = _venv(a)
    b.mkdir()
    augmenter = WorkspaceConfigAugmenter(venv_for=workspace_venv)
    result = await _round_trip(
        augmenter,
        [{"scopeUri": a.as_uri(), "section": "ty"},
         {"scopeUri": b.as_uri(), "section": "ty"}],
        [{"diagnosticMode": "openFilesOnly"}, None],
    )
    assert result[0] == {
        "diagnosticMode": "openFilesOnly",
        "pythonExtension": {"activeEnvironment": {"executable": {"sysPrefix": str(venv_a)}}},
    }
    assert result[1] is None


@pytest.mark.asyncio
async def test_an_editor_chosen_interpreter_is_kept(tmp_path):
    _venv(tmp_path)
    augmenter = WorkspaceConfigAugmenter(venv_for=workspace_venv)
    chosen = {"pythonExtension": {"activeEnvironment": {"executable": {"sysPrefix": "/x"}}}}
    result = await _round_trip(
        augmenter, [{"scopeUri": tmp_path.as_uri(), "section": "ty"}], [chosen],
    )
    assert result == [chosen]


@pytest.mark.asyncio
async def test_unrelated_traffic_passes_through_untouched(tmp_path):
    _venv(tmp_path)
    augmenter = WorkspaceConfigAugmenter(venv_for=workspace_venv)
    request = json.dumps({
        "jsonrpc": "2.0", "id": 3, "method": "workspace/configuration",
        "params": {"items": [{"scopeUri": tmp_path.as_uri(), "section": "other"}]},
    }).encode()
    assert await augmenter.on_server_message(request) is request
    reply = json.dumps({"jsonrpc": "2.0", "id": 3, "result": [None]}).encode()
    # Another server's section: not ours to fill in.
    assert await augmenter.on_editor_message(reply) is reply
    hover = json.dumps({"jsonrpc": "2.0", "id": 9, "result": {"contents": "x"}}).encode()
    assert await augmenter.on_editor_message(hover) is hover


@pytest.mark.asyncio
async def test_the_proxy_hands_each_folder_its_venv_end_to_end(tmp_path):
    """Through the real ``iommi_lsp`` process: the backend asks for both
    folders' configuration, the editor answers ``null``, and the backend
    receives the venv for the folder that has one."""
    a, b = tmp_path / "a", tmp_path / "b"
    venv_a = _venv(a)
    b.mkdir()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "iommi_lsp",
        "--ty-command", f"{sys.executable} {HERE / 'fake_ty_config.py'}",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"capabilities": {}, "workspaceFolders": [
                {"uri": a.as_uri(), "name": "a"},
                {"uri": b.as_uri(), "name": "b"},
            ]},
        }))
        await proc.stdin.drain()
        assert (await asyncio.wait_for(_read_frame(proc.stdout), 10))["id"] == 1
        request = await asyncio.wait_for(_read_frame(proc.stdout), 10)
        assert request["method"] == "workspace/configuration"
        proc.stdin.write(_frame({
            "jsonrpc": "2.0", "id": request["id"], "result": [None, None],
        }))
        await proc.stdin.drain()
        report = await asyncio.wait_for(_read_frame(proc.stdout), 10)
        received = json.loads(report["params"]["diagnostics"][0]["message"])
        assert received == [
            {"pythonExtension": {"activeEnvironment": {
                "executable": {"sysPrefix": str(venv_a)}}}},
            None,
        ]
        proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "exit"}))
        await proc.stdin.drain()
        assert await asyncio.wait_for(proc.wait(), 5) == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
