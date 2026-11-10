"""Discovery and manipulation of OrcaSlicer profiles (machine/process/filament).

OrcaSlicer stores user profiles as JSON under a per-platform config dir:

  Linux:   ~/.config/OrcaSlicer/user/<account>/{machine,process,filament}/
  macOS:   ~/Library/Application Support/OrcaSlicer/user/<account>/...
  Windows: %APPDATA%/OrcaSlicer/user/<account>/...

System (vendor) presets live under a sibling ``system/`` tree inside the
installation's resources, and additionally get cached in the config dir on
some platforms. User profiles frequently use ``"inherits"`` to point at a
system preset; the CLI resolves inheritance itself as long as it can find its
resource tree, so we pass profiles through untouched except when the caller
requests overrides — in which case we shallow-merge overrides into a copy of
the profile JSON written to a temp file.

The flatpak config dir is also checked on Linux:
  ~/.var/app/*OrcaSlicer*/config/OrcaSlicer/... (any installed app ID)
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

PROFILE_TYPES = ("machine", "process", "filament")


@dataclass
class Profile:
    name: str
    type: str          # machine | process | filament
    path: str
    source: str        # "user" | "explicit"
    inherits: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _config_roots() -> list[Path]:
    # ORCASLICER_CONFIG_DIR overrides discovery entirely (see README) so tests
    # and non-standard installs aren't polluted by platform default locations.
    override = os.environ.get("ORCASLICER_CONFIG_DIR")
    if override:
        root = Path(override).expanduser()
        return [root] if root.is_dir() else []

    system = platform.system()
    home = Path.home()
    roots: list[Path] = []
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            roots.append(Path(appdata) / "OrcaSlicer")
    elif system == "Darwin":
        roots.append(home / "Library/Application Support/OrcaSlicer")
    else:
        roots.append(home / ".config/OrcaSlicer")
        # Flatpak sandbox config dirs — app ID varies by install era, so glob.
        var_app = home / ".var/app"
        if var_app.is_dir():
            for d in sorted(var_app.glob("*OrcaSlicer*/config/OrcaSlicer")):
                roots.append(d)
    return [r for r in roots if r.is_dir()]


def discover_profiles(profile_type: str | None = None) -> list[Profile]:
    """List user profiles found in OrcaSlicer config directories."""
    types = [profile_type] if profile_type else list(PROFILE_TYPES)
    for t in types:
        if t not in PROFILE_TYPES:
            raise ValueError(f"profile_type must be one of {PROFILE_TYPES}, got {t!r}")

    profiles: list[Profile] = []
    seen: set[str] = set()
    for root in _config_roots():
        user_root = root / "user"
        if not user_root.is_dir():
            continue
        for account_dir in sorted(user_root.iterdir()):
            for t in types:
                type_dir = account_dir / t
                if not type_dir.is_dir():
                    continue
                for f in sorted(type_dir.glob("*.json")):
                    key = f"{t}:{f.stem}"
                    if key in seen:
                        continue
                    seen.add(key)
                    inherits = None
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        inherits = data.get("inherits")
                    except (json.JSONDecodeError, OSError):
                        pass
                    profiles.append(
                        Profile(name=f.stem, type=t, path=str(f),
                                source="user", inherits=inherits)
                    )
    return profiles


def resolve_profile(name_or_path: str, profile_type: str) -> Profile:
    """Resolve a profile reference: an explicit path, or a user-profile name."""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        inherits = None
        try:
            inherits = json.loads(p.read_text(encoding="utf-8")).get("inherits")
        except (json.JSONDecodeError, OSError):
            pass
        return Profile(name=p.stem, type=profile_type, path=str(p),
                       source="explicit", inherits=inherits)

    matches = [
        prof for prof in discover_profiles(profile_type)
        if prof.name.lower() == name_or_path.lower()
    ]
    if matches:
        return matches[0]

    # Fall back to substring match so "PolyFlex TPU95" finds
    # "Generic TPU @Elegoo Centauri Carbon - PolyFlex TPU95" style names.
    partial = [
        prof for prof in discover_profiles(profile_type)
        if name_or_path.lower() in prof.name.lower()
    ]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        options = ", ".join(prof.name for prof in partial[:8])
        raise FileNotFoundError(
            f"Ambiguous {profile_type} profile '{name_or_path}'. "
            f"Matches: {options}"
        )

    available = ", ".join(
        prof.name for prof in discover_profiles(profile_type)[:12]
    ) or "(none found)"
    raise FileNotFoundError(
        f"No {profile_type} profile named '{name_or_path}' found and it is not "
        f"a file path. Available user profiles: {available}"
    )


def read_profile_json(profile: Profile) -> dict:
    return json.loads(Path(profile.path).read_text(encoding="utf-8"))


def apply_overrides(profile: Profile, overrides: dict, workdir: Path) -> Path:
    """Write a temp copy of ``profile`` with ``overrides`` merged in.

    OrcaSlicer profile JSON is flat key -> value (values are strings or lists
    of strings, one per extruder/filament). We coerce scalars to strings to
    match the native format, and wrap values in a single-element list when the
    existing key is list-typed.
    """
    data = read_profile_json(profile)

    for key, value in overrides.items():
        existing = data.get(key)
        if isinstance(existing, list):
            if isinstance(value, list):
                data[key] = [str(v) for v in value]
            else:
                data[key] = [str(value)] * max(1, len(existing))
        else:
            data[key] = str(value) if not isinstance(value, (dict, list)) else value

    # Give the derived profile a distinct name so G-code metadata reflects it.
    suffix = "+".join(f"{k}={overrides[k]}" for k in list(overrides)[:3])
    data["name"] = f"{data.get('name', profile.name)} [{suffix}]"[:120]

    workdir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f"{profile.type}_override_", suffix=".json", dir=str(workdir)
    )
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    return Path(tmp_path)


# --------------------------------------------------------------------------------------
# Writing new user presets (never touches existing files).
# --------------------------------------------------------------------------------------

# Key that OrcaSlicer uses to store the preset's own name, per profile type.
_ID_KEY = {
    "machine": "printer_settings_id",
    "process": "print_settings_id",
    "filament": "filament_settings_id",
}

# Version string written by OrcaSlicer 2.3.x/2.4.x/2.5.x user presets we inspected.
_PRESET_VERSION = "2.3.2.60"


def _user_dir_for_write(account: str | None) -> Path:
    roots = _config_roots()
    if not roots:
        # Fall back to the platform default even if the directory does not exist yet.
        roots = [Path(os.environ.get("ORCASLICER_CONFIG_DIR")
                      or Path.home() / "Library/Application Support/OrcaSlicer")]
    root = roots[0]
    if account is None:
        user_root = root / "user"
        existing = sorted(p.name for p in user_root.iterdir()) if user_root.is_dir() else []
        account = existing[0] if existing else "default"
    return root / "user" / account


def bundle_has_preset(name: str, bundle_dir: Path | None) -> bool:
    """True if the preset name appears in the OrcaSlicer system bundle (*.opc files).

    System presets are stored in binary .opc archives, not as readable JSON, so the
    check is a byte search for the name. It is a presence check, not a value check.
    """
    if bundle_dir is None or not bundle_dir.is_dir():
        return False
    needle = name.encode("utf-8")
    return any(needle in f.read_bytes() for f in bundle_dir.rglob("*.opc"))


def _coerce_value(value):
    """Orca stores most values as strings; per-extruder/filament keys as lists of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, dict):
        raise ValueError("nested objects are not valid OrcaSlicer preset values")
    return str(value)


def write_user_profile(
    kind: str,
    name: str,
    parent: str,
    overrides: dict | None = None,
    account: str | None = None,
    overwrite: bool = False,
    bundle_dir: Path | None = None,
) -> dict:
    """Create a new user preset that inherits from ``parent`` and sets only ``overrides``.

    - Refuses names that already exist as user presets (unless overwrite=True, which still
      never touches system presets).
    - Refuses a parent that cannot be found in the user presets or in the system bundle.
    - Writes only the new file ``<user>/<kind>/<name>.json``; nothing else is changed.
    Orca must be restarted before the new preset appears in the GUI.
    """
    if kind not in PROFILE_TYPES:
        raise ValueError(f"kind must be one of {PROFILE_TYPES}, got {kind!r}")
    name = (name or "").strip()
    parent = (parent or "").strip()
    if not name:
        raise ValueError("name is required")
    if any(c in name for c in '/\\:*?"<>|'):
        raise ValueError(f"name contains characters not allowed in file names: {name!r}")
    if not parent:
        raise ValueError("parent (the system preset this one inherits from) is required")

    user_parent = [p for p in discover_profiles(kind) if p.name == parent]
    if not user_parent and not bundle_has_preset(parent, bundle_dir):
        raise FileNotFoundError(
            f"parent preset {parent!r} not found in user presets or in the OrcaSlicer bundle"
        )

    target_dir = _user_dir_for_write(account) / kind
    target = target_dir / f"{name}.json"
    existing_names = {p.name for p in discover_profiles(kind)}
    if name in existing_names and not overwrite:
        raise FileExistsError(f"a {kind} preset named {name!r} already exists (overwrite=False)")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists (overwrite=False)")

    data: dict = {
        "name": name,
        "inherits": parent,
        "from": "User",
        "version": _PRESET_VERSION,
        _ID_KEY[kind]: [name] if kind == "filament" else name,
    }
    for key, value in (overrides or {}).items():
        if key in ("name", "inherits", "from", "version") or key == _ID_KEY[kind]:
            raise ValueError(f"{key!r} is managed by write_user_profile and cannot be overridden")
        data[key] = _coerce_value(value)

    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)
    return {
        "written": str(target),
        "kind": kind,
        "name": name,
        "inherits": parent,
        "overrides": sorted((overrides or {}).keys()),
        "note": "Restart OrcaSlicer to see the new preset in the GUI. "
                "The CLI cannot resolve user presets that inherit from system presets.",
    }
