import json
from pathlib import Path

from skillforge.cli import load_seed_skill, main

TASK = str(Path(__file__).parent.parent / "tasks" / "stub_pick_cube_to_bin.json")


def test_cli_forge_with_fake_improver(tmp_path, capsys):
    rc = main([
        "forge", TASK, "--seed-skill", "pick_place", "--rounds", "2", "--rollouts", "10",
        "--improver", "fake", "--target", "1.01", "--library", str(tmp_path / "lib"), "--out", str(tmp_path / "forge"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("\nround ") == 2 and "stopped: rounds" in out
    (hist,) = (tmp_path / "forge").glob("*/history.json")
    doc = json.loads(hist.read_text())
    assert doc["meta"]["improver"] == "fake" and len(doc["rounds"]) == 2
    assert doc["meta"]["gate"] == "paired" and doc["gate"]["alpha"] == 0.1
    assert "paired vs 0.2.0 +" in out and " p=" in out  # the round line shows the paired counts
    assert (tmp_path / "lib" / "stub_pick_cube_to_bin" / "0.2.0" / "skill.py").exists()


def test_cli_claude_without_key_exits_cleanly(tmp_path, monkeypatch, capsys):
    for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(v, raising=False)
    rc = main(["forge", TASK, "--improver", "claude", "--library", str(tmp_path), "--out", str(tmp_path)])
    assert rc == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_seed_skill_resolution(tmp_path):
    s = load_seed_skill("pick_place")
    assert (s.name, s.version, s.entrypoint) == ("pick_place", "0.2.0", "PickPlace")
    s = load_seed_skill("pick_place_v01")
    assert (s.name, s.version, s.entrypoint) == ("pick_place", "0.1.0", "PickPlaceV01")
    f = tmp_path / "my_skill.py"
    f.write_text("class A:\n    def reset(self, t): pass\n    def act(self, o): pass\n")
    assert load_seed_skill(str(f)).entrypoint == "A"
