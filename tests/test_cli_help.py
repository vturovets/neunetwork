import subprocess, sys, shutil, os, pathlib, pytest

@pytest.mark.skip("skeleton: enable after implementation")
def test_cli_help():
    assert shutil.which("neunet") or os.path.exists(os.path.join(os.getcwd(), "neunet", "cli.py"))
