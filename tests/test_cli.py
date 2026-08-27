"""Tests for the nextaire_tools command-line interface."""

from __future__ import annotations

import pytest

from nextaire_tools.cli import main


def test_info_command(capsys):
    rc = main(["info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nextaire_tools" in out


def test_version_flag():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_preprocess_command(aq_csv, tmp_path):
    out = tmp_path / "clean.csv"
    rc = main(["preprocess", str(aq_csv), str(out), "--time-col", "timestamp", "--set-time-index"])
    assert rc == 0
    assert out.exists()


def test_no_command_prints_help(capsys):
    rc = main([])
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()
