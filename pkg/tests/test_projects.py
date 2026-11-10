"""Offline tests for 3MF project tools (inspect_project, slice_project) and write_profile.

OrcaSlicer is never executed: the slicer call is replaced by a fake that writes the
same files the real CLI writes (checked against OrcaSlicer 2.5.0 on macOS).
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from orca_mcp import profiles, server
from orca_mcp.slicer import SlicerResult

GCODE = """\
; estimated printing time (normal mode) = 41m 34s
; total layer number: 3
; total filament used [g] = 4.98
; filament used [mm] = 1669.94
M104 S210
"""

PARENT = "0.20mm Standard @Creality Ender3V3SE 0.4"


def _make_project(path: Path, objects: bool = True) -> Path:
    settings = {
        "printer_settings_id": "Creality Ender-3 V3 SE 0.4 nozzle",
        "print_settings_id": PARENT + " - Copy",
        "filament_settings_id": ["Creality Generic PLA High Speed @Ender-3V3-all - Copy"],
        "layer_height": "0.2",
        "wall_loops": "2",
        "nozzle_temperature": ["210"],
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Metadata/project_settings.config", json.dumps(settings))
        if objects:
            z.writestr("Metadata/model_settings.config",
                       '<config><object id="2"><metadata key="name" value="tower.stl"/>'
                       '<part id="1"/></object></config>')
        z.writestr("3D/3dmodel.model", "<model/>")
    return path


def _fake_run_slicer(args, *, timeout=600, cwd=None):
    """Mimic the OrcaSlicer CLI output for `--slice N --export-3mf sliced.3mf`."""
    outdir = Path(args[args.index("--outputdir") + 1])
    (outdir / "plate_1.gcode").write_text(GCODE)
    (outdir / "plate_1.statistics.json").write_text(json.dumps(
        {"summary": {"time": "2494.5", "layer_count": "3"}, "objects": []}))
    with zipfile.ZipFile(outdir / "sliced.3mf", "w") as z:
        z.writestr("Metadata/plate_1.png", b"\x89PNG-fake-preview")
        z.writestr("Metadata/plate_1.gcode", GCODE)
    return SlicerResult(returncode=0, stdout="", stderr="got warnings: test warning\n")


@pytest.fixture()
def cfg(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "OrcaSlicer"
    (root / "user" / "default" / "process").mkdir(parents=True)
    monkeypatch.setenv("ORCASLICER_CONFIG_DIR", str(root))
    return root


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    """Fake system bundle: a binary .opc file that contains the parent preset name."""
    d = tmp_path / "bundle"
    d.mkdir()
    (d / "Creality.opc").write_bytes(b"ZCRO\x00\x01" + PARENT.encode() + b"\x00\xff")
    return d


# ---------------------------------------------------------------- write_profile

class TestWriteProfile:
    def test_writes_inheriting_preset_with_overrides(self, cfg, bundle):
        r = profiles.write_user_profile(
            "process", "0.16 Test", PARENT,
            overrides={"layer_height": 0.16, "wall_loops": "3"}, bundle_dir=bundle)
        target = Path(r["written"])
        assert target == cfg / "user" / "default" / "process" / "0.16 Test.json"
        data = json.loads(target.read_text())
        assert data["name"] == "0.16 Test"
        assert data["inherits"] == PARENT
        assert data["from"] == "User"
        assert data["print_settings_id"] == "0.16 Test"
        assert data["layer_height"] == "0.16"  # coerced to Orca string
        assert data["wall_loops"] == "3"
        assert r["overrides"] == ["layer_height", "wall_loops"]

    def test_filament_id_key_is_a_list(self, cfg, bundle):
        profiles.write_user_profile("filament", "PLA Test", PARENT, bundle_dir=bundle)
        (cfg / "user" / "default" / "filament").mkdir(exist_ok=True)
        # write again into the filament dir to check the id key shape
        r = profiles.write_user_profile("filament", "PLA Test 2", PARENT,
                                        overrides={"nozzle_temperature": [215]},
                                        bundle_dir=bundle)
        data = json.loads(Path(r["written"]).read_text())
        assert data["filament_settings_id"] == ["PLA Test 2"]
        assert data["nozzle_temperature"] == ["215"]

    def test_refuses_existing_name_and_leaves_file_untouched(self, cfg, bundle):
        target = cfg / "user" / "default" / "process" / "Keep Me.json"
        target.write_text('{"name": "Keep Me", "inherits": "x"}')
        before = hashlib.sha256(target.read_bytes()).hexdigest()
        with pytest.raises(FileExistsError):
            profiles.write_user_profile("process", "Keep Me", PARENT, bundle_dir=bundle)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == before

    def test_overwrite_replaces_only_the_named_file(self, cfg, bundle):
        target = cfg / "user" / "default" / "process" / "Mine.json"
        target.write_text('{"name": "Mine", "inherits": "x"}')
        profiles.write_user_profile("process", "Mine", PARENT, overwrite=True,
                                    bundle_dir=bundle)
        assert json.loads(target.read_text())["inherits"] == PARENT

    def test_refuses_unknown_parent(self, cfg, bundle):
        with pytest.raises(FileNotFoundError):
            profiles.write_user_profile("process", "X", "No Such Preset", bundle_dir=bundle)
        assert not any((cfg / "user" / "default" / "process").iterdir())

    def test_refuses_reserved_and_unsafe_input(self, cfg, bundle):
        with pytest.raises(ValueError):
            profiles.write_user_profile("process", "A", PARENT,
                                        overrides={"inherits": "other"}, bundle_dir=bundle)
        with pytest.raises(ValueError):
            profiles.write_user_profile("process", "a/b", PARENT, bundle_dir=bundle)
        with pytest.raises(ValueError):
            profiles.write_user_profile("bogus", "A", PARENT, bundle_dir=bundle)
        with pytest.raises(ValueError):
            profiles.write_user_profile("process", "  ", PARENT, bundle_dir=bundle)

    def test_parent_found_in_user_presets_without_bundle(self, cfg):
        (cfg / "user" / "default" / "process" / "Mine.json").write_text(
            json.dumps({"name": "Mine"}))
        r = profiles.write_user_profile("process", "Child", "Mine", bundle_dir=None)
        assert json.loads(Path(r["written"]).read_text())["inherits"] == "Mine"

    def test_mcp_tool_returns_error_dict_instead_of_raising(self, cfg, monkeypatch):
        monkeypatch.setattr(server, "_bundle_dir", lambda: None)
        out = server.write_profile("process", "X", "Missing parent")
        assert out["error"] == "FileNotFoundError"


# ---------------------------------------------------------------- inspect / slice

class TestInspectProject:
    def test_summarises_settings_and_objects(self, tmp_path):
        proj = _make_project(tmp_path / "p.3mf")
        out = server.inspect_project(str(proj))
        assert out["settings"]["layer_height"] == "0.2"
        assert out["settings"]["nozzle_temperature"] == ["210"]
        assert out["objects"] == [{"id": "2", "name": "tower.stl", "parts": 1}]

    def test_rejects_non_project_zip(self, tmp_path):
        bad = tmp_path / "bad.3mf"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("x.txt", "nothing")
        assert "error" in server.inspect_project(str(bad))

    def test_missing_file(self, tmp_path):
        assert server.inspect_project(str(tmp_path / "nope.3mf"))["error"] == "FileNotFoundError"


class TestSliceProject:
    def test_produces_sliced_3mf_and_previews(self, tmp_path, cfg, monkeypatch):
        monkeypatch.setattr(server, "run_slicer", _fake_run_slicer)
        proj = _make_project(tmp_path / "tower.3mf")
        before = hashlib.sha256(proj.read_bytes()).hexdigest()

        out = server.slice_project(str(proj), output_dir=str(tmp_path / "runs"))

        assert out["returncode"] == 0
        assert Path(out["sliced_3mf"]).is_file()
        assert Path(out["previews"][0]).read_bytes() == b"\x89PNG-fake-preview"
        assert out["estimates"]["estimated_time_pretty"] == "41m 34s"
        assert out["statistics"]["summary"]["layer_count"] == "3"
        assert out["warnings"] == ["got warnings: test warning"]
        # the source project is never modified
        assert hashlib.sha256(proj.read_bytes()).hexdigest() == before

    def test_passes_export_and_datadir_arguments(self, tmp_path, cfg, monkeypatch):
        seen = {}

        def spy(args, *, timeout=600, cwd=None):
            seen["args"] = args
            return _fake_run_slicer(args)

        monkeypatch.setattr(server, "run_slicer", spy)
        server.slice_project(str(_make_project(tmp_path / "t.3mf")),
                             output_dir=str(tmp_path / "runs"), plate=2)
        args = seen["args"]
        assert args[args.index("--export-3mf") + 1] == "sliced.3mf"  # relative name
        assert args[args.index("--slice") + 1] == "2"
        assert args[args.index("--datadir") + 1] == str(cfg)

    def test_non_3mf_input_is_rejected(self, tmp_path):
        stl = tmp_path / "model.stl"
        stl.write_bytes(b"solid x\nendsolid x\n")
        assert server.slice_project(str(stl))["error"] == "ValueError"

    def test_missing_file_is_reported(self, tmp_path):
        assert server.slice_project(str(tmp_path / "gone.3mf"))["error"] == "FileNotFoundError"

    def test_failed_slice_without_gcode_returns_error(self, tmp_path, cfg, monkeypatch):
        def failing(args, *, timeout=600, cwd=None):
            return SlicerResult(returncode=251, stdout="", stderr="can not resolve preset")

        monkeypatch.setattr(server, "run_slicer", failing)
        out = server.slice_project(str(_make_project(tmp_path / "t.3mf")),
                                   output_dir=str(tmp_path / "runs"))
        assert out["error"] == "SlicerError"
        assert out["returncode"] == 251
