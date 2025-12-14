import subprocess
import sys


def run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "sreejita.cli"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_cli_help():
    result = run_cli(["--help"])
    assert result.returncode == 0


def test_cli_version():
    result = run_cli(["--version"])
    assert result.returncode == 0
