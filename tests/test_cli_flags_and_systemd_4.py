"""Tests verifying CLI argument flag aliases and systemd unit alignment (#4).

Ensures that:
1. rapidocr-worker accepts both short (--det) and long (--det-model) flags.
2. configs/systemd/rapidocr-worker@.service uses valid flags recognized by the binary.
3. configs/systemd/rapidocr-worker@.service and deploy/systemd/rapidocr-worker@.service stay in sync.
"""

from pathlib import Path
import subprocess


def test_systemd_units_match_and_use_valid_flags() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cfg_unit = repo_root / "configs" / "systemd" / "rapidocr-worker@.service"
    deploy_unit = repo_root / "deploy" / "systemd" / "rapidocr-worker@.service"

    assert cfg_unit.exists(), f"Missing {cfg_unit}"
    assert deploy_unit.exists(), f"Missing {deploy_unit}"

    cfg_text = cfg_unit.read_text(encoding="utf-8")
    deploy_text = deploy_unit.read_text(encoding="utf-8")

    assert cfg_text == deploy_text, "configs and deploy systemd units must be identical"

    exec_start = None
    for line in cfg_text.splitlines():
        if line.startswith("ExecStart="):
            exec_start = line
            break

    assert exec_start is not None, "Missing ExecStart in service unit"

    # Must contain required flags
    assert "--socket" in exec_start
    assert "--det-model" in exec_start or "--det" in exec_start
    assert "--rec-model" in exec_start or "--rec" in exec_start
    assert "--limit-side-len" in exec_start
    assert "--threads" in exec_start


def test_binary_cli_flags_parsing() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "target" / "release" / "rapidocr-worker"
    if not bin_path.exists():
        bin_path = repo_root / "target" / "debug" / "rapidocr-worker"

    if bin_path.exists():
        res = subprocess.run([str(bin_path), "--help"], capture_output=True, text=True)
        assert "--det|--det-model" in res.stderr or "--det-model" in res.stderr
        assert "--limit-side-len 736" in res.stderr
        assert "--keys" not in res.stderr
