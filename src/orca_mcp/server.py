"""MCP server exposing the OrcaSlicer CLI.

Tools:
  check_installation   - verify OrcaSlicer is reachable, report how
  list_profiles        - enumerate user machine/process/filament profiles
  get_profile          - dump a profile's JSON
  get_model_info       - run `orca-slicer --info` on a model file
  slice_model          - slice with profiles + optional parameter overrides
  parameter_sweep      - slice N variants varying one parameter, compare
  get_slice_estimates  - parse estimates out of an existing G-code file
  analyze_gcode_layers - per-layer fan/speed/feature stats, Z-windowed
  inspect_project      - presets, objects and key settings of a saved 3MF project
  slice_project        - slice a saved 3MF; returns sliced_3mf (open in OrcaSlicer) + previews
  write_profile        - create a new user preset that inherits from a system preset

Run with:  orca-mcp   (stdio transport, the MCP default)
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import gcode as gcode_mod
from . import profiles as prof_mod
from .slicer import (
    SlicerError,
    SlicerNotFoundError,
    find_slicer,
    run_slicer,
    slicer_info,
)

mcp = FastMCP(
    "orca-mcp",
    instructions=(
        "Wraps the OrcaSlicer CLI. Typical flow: check_installation once, "
        "list_profiles to find machine/process/filament presets, then "
        "slice_model (optionally with process_overrides) and inspect the "
        "result with get_slice_estimates / analyze_gcode_layers. Use "
        "parameter_sweep to compare several values of one setting."
    ),
)

_MODEL_EXTS = {".stl", ".3mf", ".obj", ".step", ".stp", ".amf"}


def _default_output_dir() -> Path:
    """Pick a default output directory the slicer can actually write to.

    A sandboxed slicer (Flatpak) gets a *private* /tmp, so anything the CLI
    writes there is invisible to us and vice versa. Flatpak apps do share
    $HOME (per their filesystem permissions), so when the resolved slicer is
    a Flatpak we default to ~/.cache/orca-mcp instead of the system
    temp dir. Users can always pass output_dir explicitly.
    """
    sandboxed = False
    try:
        sandboxed = find_slicer().kind == "flatpak"
    except SlicerNotFoundError:
        pass

    if sandboxed:
        d = Path.home() / ".cache" / "orca-mcp"
    else:
        d = Path(tempfile.gettempdir()) / "orca-mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate_model_path(model_path: str) -> Path:
    p = Path(model_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {p}")
    if p.suffix.lower() not in _MODEL_EXTS:
        raise ValueError(
            f"Unsupported model extension '{p.suffix}'. "
            f"Supported: {sorted(_MODEL_EXTS)}"
        )
    return p


def _err(e: Exception) -> dict:
    payload = {"error": type(e).__name__, "message": str(e)}
    if isinstance(e, SlicerError):
        payload["returncode"] = e.returncode
        payload["stderr_tail"] = (e.stderr or "")[-2000:]
    return payload


@mcp.tool()
def check_installation() -> dict:
    """Check whether OrcaSlicer is installed and how it will be invoked.

    Returns the resolved executable/invocation, platform, and version banner.
    Call this first if any other tool fails unexpectedly.
    """
    return slicer_info()


@mcp.tool()
def list_profiles(profile_type: str | None = None) -> dict:
    """List user profiles from the OrcaSlicer config directory.

    Args:
        profile_type: Optional filter: 'machine', 'process', or 'filament'.
                      Omit to list all three types.
    """
    try:
        profs = prof_mod.discover_profiles(profile_type)
    except ValueError as e:
        return _err(e)
    return {
        "count": len(profs),
        "profiles": [p.to_dict() for p in profs],
        "note": (
            "Only user profiles are listed. System/vendor presets can still be "
            "referenced by passing an exported JSON file path directly."
        ),
    }


@mcp.tool()
def get_profile(name_or_path: str, profile_type: str) -> dict:
    """Return the full JSON contents of a profile.

    Args:
        name_or_path: Profile name (user profile) or path to a profile JSON.
        profile_type: 'machine', 'process', or 'filament'.
    """
    try:
        prof = prof_mod.resolve_profile(name_or_path, profile_type)
        data = prof_mod.read_profile_json(prof)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        return _err(e)
    return {"profile": prof.to_dict(), "settings": data}


@mcp.tool()
def get_model_info(model_path: str) -> dict:
    """Run OrcaSlicer's --info on a model file (bounding box, volume, facets).

    Args:
        model_path: Path to an STL/3MF/OBJ/STEP model.
    """
    try:
        p = _validate_model_path(model_path)
        res = run_slicer(["--info", str(p)], timeout=120)
    except (FileNotFoundError, ValueError, SlicerNotFoundError, SlicerError) as e:
        return _err(e)
    return {
        "model": str(p),
        "returncode": res.returncode,
        "info": res.stdout.strip(),
        "stderr_tail": res.stderr[-1000:] if res.returncode != 0 else "",
    }


def _do_slice(
    model: Path,
    machine: str,
    process: str,
    filament: str,
    output_dir: Path,
    process_overrides: dict | None,
    filament_overrides: dict | None,
    plate: int,
    timeout: int,
) -> dict:
    machine_p = prof_mod.resolve_profile(machine, "machine")
    process_p = prof_mod.resolve_profile(process, "process")
    filament_p = prof_mod.resolve_profile(filament, "filament")

    workdir = output_dir / "derived-profiles"
    process_path = Path(process_p.path)
    filament_path = Path(filament_p.path)
    if process_overrides:
        process_path = prof_mod.apply_overrides(process_p, process_overrides, workdir)
    if filament_overrides:
        filament_path = prof_mod.apply_overrides(filament_p, filament_overrides, workdir)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slice_dir = output_dir / f"{model.stem}-{stamp}"
    slice_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--load-settings", f"{machine_p.path};{process_path}",
        "--load-filaments", str(filament_path),
        "--slice", str(plate),
        "--export-slicedata", str(slice_dir / "slicedata"),
        "--outputdir", str(slice_dir),
        str(model),
    ]
    res = run_slicer(args, timeout=timeout)

    gcode_files = sorted(slice_dir.rglob("*.gcode"))
    if res.returncode != 0 and not gcode_files:
        raise SlicerError(
            "Slicing failed", returncode=res.returncode,
            stdout=res.stdout, stderr=res.stderr,
        )

    result: dict = {
        "returncode": res.returncode,
        "used_xvfb": res.used_xvfb,
        "output_dir": str(slice_dir),
        "gcode_files": [str(g) for g in gcode_files],
        "profiles_used": {
            "machine": machine_p.name,
            "process": str(process_path),
            "filament": str(filament_path),
        },
    }
    if gcode_files:
        result["estimates"] = gcode_mod.parse_estimates(gcode_files[0])
    else:
        result["stdout_tail"] = res.stdout[-1500:]
        result["stderr_tail"] = res.stderr[-1500:]
    return result


@mcp.tool()
def slice_model(
    model_path: str,
    machine_profile: str,
    process_profile: str,
    filament_profile: str,
    process_overrides: dict | None = None,
    filament_overrides: dict | None = None,
    output_dir: str | None = None,
    plate: int = 1,
    timeout_seconds: int = 600,
) -> dict:
    """Slice a model and return G-code path(s) plus parsed estimates.

    Args:
        model_path: Path to STL/3MF/OBJ/STEP file.
        machine_profile: Machine profile name or JSON path.
        process_profile: Process (print settings) profile name or JSON path.
        filament_profile: Filament profile name or JSON path.
        process_overrides: Optional dict of process settings to override,
            e.g. {"wall_loops": 4, "fan_max_speed": 30}. Keys use OrcaSlicer's
            internal names (see get_profile output for valid keys).
        filament_overrides: Same idea for filament settings,
            e.g. {"fan_cooling_layer_time": 12, "nozzle_temperature": [235]}.
        output_dir: Where to put G-code (default: system temp / orca-mcp).
        plate: Plate index to slice (default 1; 0 slices all plates).
        timeout_seconds: Kill the slice if it exceeds this.
    """
    try:
        model = _validate_model_path(model_path)
        out = Path(output_dir).expanduser() if output_dir else _default_output_dir()
        result = _do_slice(
            model, machine_profile, process_profile, filament_profile,
            out, process_overrides, filament_overrides, plate, timeout_seconds,
        )
        warning = _sandbox_path_warning(out)
        if warning:
            result["warning"] = warning
        return result
    except (FileNotFoundError, ValueError, SlicerNotFoundError, SlicerError) as e:
        return _err(e)


def _sandbox_path_warning(path: Path) -> str | None:
    """Flag paths a Flatpak-sandboxed slicer likely cannot see."""
    try:
        cmd = find_slicer()
        if cmd.kind != "flatpak":
            return None
    except SlicerNotFoundError:
        return None
    app_id = cmd.flatpak_app_id or "<orcaslicer-app-id>"
    p = str(path.resolve())
    tmp = str(Path(tempfile.gettempdir()).resolve())
    if p.startswith(tmp) or p.startswith("/var/tmp") or p.startswith("/run/"):
        return (
            f"Output dir '{p}' is under a directory the Flatpak sandbox does "
            "not share with the host — the slicer's output may not appear "
            "there. Use a path under your home directory instead, or grant "
            f"access with: flatpak override --user --filesystem=/tmp {app_id}"
        )
    return None


@mcp.tool()
def parameter_sweep(
    model_path: str,
    machine_profile: str,
    process_profile: str,
    filament_profile: str,
    sweep_parameter: str,
    sweep_values: list,
    parameter_target: str = "process",
    output_dir: str | None = None,
    plate: int = 1,
    timeout_seconds: int = 600,
) -> dict:
    """Slice the same model once per value of a single parameter and compare.

    Args:
        model_path: Path to the model file.
        machine_profile / process_profile / filament_profile: As in slice_model.
        sweep_parameter: Setting key to vary (e.g. 'fan_max_speed',
            'slow_down_layer_time', 'wall_loops').
        sweep_values: List of values to try, e.g. [20, 40, 60, 80].
        parameter_target: 'process' or 'filament' — which profile the
            parameter belongs to.
        output_dir: Base output directory.
        plate: Plate index (default 1).
        timeout_seconds: Per-slice timeout.

    Returns one entry per value with G-code path and estimates, so results
    can be diffed (e.g. with analyze_gcode_layers over the same Z window).
    """
    if parameter_target not in ("process", "filament"):
        return _err(ValueError("parameter_target must be 'process' or 'filament'"))
    if not sweep_values:
        return _err(ValueError("sweep_values must be a non-empty list"))
    if len(sweep_values) > 12:
        return _err(ValueError("Refusing to sweep more than 12 values in one call"))

    try:
        model = _validate_model_path(model_path)
    except (FileNotFoundError, ValueError) as e:
        return _err(e)
    out = Path(output_dir).expanduser() if output_dir else _default_output_dir()

    runs = []
    for value in sweep_values:
        overrides = {sweep_parameter: value}
        try:
            r = _do_slice(
                model, machine_profile, process_profile, filament_profile, out,
                overrides if parameter_target == "process" else None,
                overrides if parameter_target == "filament" else None,
                plate, timeout_seconds,
            )
            runs.append({"value": value, "ok": True, **{
                "gcode": r["gcode_files"][0] if r["gcode_files"] else None,
                "estimates": r.get("estimates"),
                "output_dir": r["output_dir"],
            }})
        except (FileNotFoundError, ValueError, SlicerNotFoundError, SlicerError) as e:
            runs.append({"value": value, "ok": False, **_err(e)})

    comparison = [
        {
            "value": r["value"],
            "time_s": (r.get("estimates") or {}).get("estimated_time_seconds"),
            "filament_g": (r.get("estimates") or {}).get("filament_used_g"),
        }
        for r in runs if r.get("ok")
    ]
    return {
        "sweep_parameter": sweep_parameter,
        "parameter_target": parameter_target,
        "runs": runs,
        "comparison": comparison,
    }


@mcp.tool()
def get_slice_estimates(gcode_path: str) -> dict:
    """Parse print time, filament usage, and key settings from a G-code file.

    Args:
        gcode_path: Path to a .gcode file produced by OrcaSlicer.
    """
    p = Path(gcode_path).expanduser()
    if not p.is_file():
        return _err(FileNotFoundError(f"G-code file not found: {p}"))
    try:
        return gcode_mod.parse_estimates(p)
    except OSError as e:
        return _err(e)


@mcp.tool()
def analyze_gcode_layers(
    gcode_path: str,
    z_min: float | None = None,
    z_max: float | None = None,
    max_layers: int = 300,
) -> dict:
    """Per-layer stats: Z, layer height, features, fan speed, print speed range.

    Useful for inspecting a specific region of a print, e.g. an overhang or
    shoulder transition — pass z_min/z_max bracketing the region and compare
    across sweep runs.

    Args:
        gcode_path: Path to a .gcode file.
        z_min: Only include layers at or above this Z (mm).
        z_max: Only include layers at or below this Z (mm).
        max_layers: Cap on returned layers (default 300).
    """
    p = Path(gcode_path).expanduser()
    if not p.is_file():
        return _err(FileNotFoundError(f"G-code file not found: {p}"))
    try:
        return gcode_mod.analyze_layers(p, z_min=z_min, z_max=z_max,
                                        max_layers=max_layers)
    except OSError as e:
        return _err(e)


# --------------------------------------------------------------------------------------
# Project-level tools: work on a saved 3MF (settings come from the project itself).
# --------------------------------------------------------------------------------------

_PREVIEW_RE = re.compile(r"^Metadata/plate_(\d+)\.png$")
_KEY_PROJECT_SETTINGS = [
    "printer_settings_id", "print_settings_id", "filament_settings_id",
    "layer_height", "initial_layer_print_height", "wall_loops", "top_shell_layers",
    "bottom_shell_layers", "sparse_infill_density", "sparse_infill_pattern",
    "enable_support", "support_type", "brim_type", "brim_width",
    "outer_wall_speed", "inner_wall_speed", "sparse_infill_speed", "travel_speed",
    "nozzle_temperature", "nozzle_temperature_initial_layer", "hot_plate_temp",
    "fan_max_speed", "fan_min_speed", "close_fan_the_first_x_layers",
    "retraction_length", "retraction_speed", "z_hop", "seam_position",
    "slow_down_layer_time",
]


def _bundle_dir() -> Path | None:
    """OrcaSlicer's bundled system presets: <App>.app/Contents/Resources/profiles (macOS)."""
    try:
        exe = Path(find_slicer().argv_prefix[0]).resolve()
    except SlicerNotFoundError:
        return None
    candidate = exe.parents[1] / "Resources" / "profiles"
    return candidate if candidate.is_dir() else None


def _read_project_settings(project: Path) -> dict:
    with zipfile.ZipFile(project) as z:
        if "Metadata/project_settings.config" not in z.namelist():
            raise ValueError(f"not an OrcaSlicer project (no project_settings.config): {project}")
        return json.loads(z.read("Metadata/project_settings.config"))


def _project_objects(project: Path) -> list[dict]:
    with zipfile.ZipFile(project) as z:
        if "Metadata/model_settings.config" not in z.namelist():
            return []
        xml = z.read("Metadata/model_settings.config").decode("utf-8", "ignore")
    out = []
    for m in re.finditer(r'<object id="(\d+)">(.*?)</object>', xml, re.S):
        name = re.search(r'key="name" value="([^"]*)"', m.group(2))
        out.append({
            "id": m.group(1),
            "name": name.group(1) if name else "?",
            "parts": len(re.findall(r"<part id=", m.group(2))),
        })
    return out


@mcp.tool()
def inspect_project(project_3mf: str) -> dict:
    """Summarise a saved OrcaSlicer 3MF project: presets, objects and key print settings.

    Read-only. Use it to check what a project contains before slicing or analysing it.
    """
    try:
        p = Path(project_3mf).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"project not found: {p}")
        s = _read_project_settings(p)
        return {
            "project": str(p),
            "objects": _project_objects(p),
            "settings": {k: s[k] for k in _KEY_PROJECT_SETTINGS if k in s},
            "total_keys": len(s),
        }
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as e:
        return _err(e)


@mcp.tool()
def slice_project(
    project_3mf: str,
    plate: int = 0,
    output_dir: str | None = None,
    timeout_seconds: int = 900,
) -> dict:
    """Slice a saved OrcaSlicer 3MF project and produce a sliced project you can open in OrcaSlicer.

    The settings come from the project itself (change them in the GUI and re-save first).
    The source file is copied to a run folder and never modified.

    Outputs in the run folder:
      - sliced.3mf: the project with G-code embedded. Open it in OrcaSlicer to see the sliced plate.
      - preview_plate_N.png: plate preview rendered by OrcaSlicer (model image, not toolpaths).
      - plate_N.gcode and plate_N.statistics.json: G-code and per-layer statistics.

    Args:
        project_3mf: Path to a saved .3mf project.
        plate: 0 = all plates, N = plate N.
        output_dir: Where to create the run folder (default: system temp / orca-mcp).
        timeout_seconds: Kill the slice if it exceeds this.
    """
    try:
        src = Path(project_3mf).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"project not found: {src}")
        if src.suffix.lower() != ".3mf":
            raise ValueError("slice_project needs a .3mf project file")
        out_root = Path(output_dir).expanduser() if output_dir else _default_output_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = out_root / f"{src.stem}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        work = run_dir / src.name
        shutil.copy2(src, work)

        args: list[str] = []
        config_roots = prof_mod._config_roots()
        if config_roots:
            args += ["--datadir", str(config_roots[0])]
        args += [
            "--slice", str(plate),
            "--export-statistics",
            "--outputdir", str(run_dir),
            "--export-3mf", "sliced.3mf",  # relative to --outputdir (absolute paths fail)
            str(work),
        ]
        res = run_slicer(args, timeout=timeout_seconds)

        gcode_files = sorted(run_dir.glob("*.gcode"))
        if res.returncode != 0 and not gcode_files:
            raise SlicerError("Slicing failed", returncode=res.returncode,
                              stdout=res.stdout, stderr=res.stderr)

        result: dict = {
            "returncode": res.returncode,
            "run_dir": str(run_dir),
            "gcode_files": [str(g) for g in gcode_files],
            "statistics_files": [str(s) for s in sorted(run_dir.glob("*.statistics.json"))],
            "used_xvfb": res.used_xvfb,
        }
        sliced = run_dir / "sliced.3mf"
        if sliced.is_file():
            result["sliced_3mf"] = str(sliced)
            result["previews"] = _extract_previews(sliced, run_dir)
        if gcode_files:
            result["estimates"] = gcode_mod.parse_estimates(gcode_files[0])
        stats = sorted(run_dir.glob("*.statistics.json"))
        if stats:
            st = json.loads(stats[0].read_text(encoding="utf-8"))
            result["statistics"] = {"summary": st.get("summary", {}),
                                    "objects": st.get("objects", [])}
        warnings = [w for w in (res.stdout + res.stderr).splitlines() if "got warnings" in w]
        if warnings:
            result["warnings"] = warnings
        return result
    except (FileNotFoundError, ValueError, SlicerNotFoundError, SlicerError,
            zipfile.BadZipFile) as e:
        return _err(e)


def _extract_previews(sliced_3mf: Path, run_dir: Path) -> list[str]:
    """Copy OrcaSlicer's own plate previews out of the 3MF as PNG files."""
    out = []
    with zipfile.ZipFile(sliced_3mf) as z:
        for name in z.namelist():
            m = _PREVIEW_RE.match(name)
            if m:
                dest = run_dir / f"preview_plate_{m.group(1)}.png"
                dest.write_bytes(z.read(name))
                out.append(str(dest))
    return sorted(out)


@mcp.tool()
def write_profile(
    kind: str,
    name: str,
    parent: str,
    overrides: dict | None = None,
    overwrite: bool = False,
) -> dict:
    """Create a NEW user preset (machine, process or filament) that inherits from a system preset.

    Only writes <OrcaSlicer user dir>/<kind>/<name>.json. Existing files and system presets are
    never modified. Restart OrcaSlicer afterwards to see the preset in the GUI.

    Args:
        kind: "machine", "process" or "filament".
        name: New preset name (must be unique among user presets).
        parent: System or user preset to inherit from, e.g.
            "0.20mm Standard @Creality Ender3V3SE 0.4" (process) or
            "Creality Ender-3 V3 SE 0.4 nozzle" (machine).
        overrides: Only the settings that differ from the parent, e.g.
            {"layer_height": "0.16", "wall_loops": "3"}. Per-extruder keys take lists,
            e.g. {"nozzle_temperature": ["210"], "fan_max_speed": ["100"]}.
        overwrite: Replace an existing user preset with the same name (default False).
    """
    try:
        return prof_mod.write_user_profile(
            kind, name, parent, overrides=overrides,
            overwrite=overwrite, bundle_dir=_bundle_dir(),
        )
    except (ValueError, FileNotFoundError, FileExistsError, SlicerNotFoundError) as e:
        return _err(e)
    except OSError as e:
        return _err(e)



def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
