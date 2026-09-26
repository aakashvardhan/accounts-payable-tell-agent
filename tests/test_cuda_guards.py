"""CPU-only tests for the Part 4 CUDA-availability guard fix. No model, no
GPU load; `torch.cuda.*` is mocked where behavior-under-CUDA needs proving
without a real GPU.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path("/home/hp5/tell")
GUARDED_FILES = {
    "src/tell/agent/loop.py": 1,
    "src/tell/agent/conditional_retrieval.py": 1,
    "src/tell/agent/memory_loop.py": 1,
}
GUARD_PATTERN = re.compile(r"if torch\.cuda\.is_available\(\):\s*\n\s*torch\.cuda\.reset_peak_memory_stats\(\)")
UNGUARDED_CALL_PATTERN = re.compile(r"^(?!\s*if\s+torch\.cuda\.is_available\(\)\s*:).*torch\.cuda\.reset_peak_memory_stats\(\)")


def test_every_previously_unguarded_call_site_now_guarded():
    for rel, expected_count in GUARDED_FILES.items():
        text = (REPO / rel).read_text()
        matches = GUARD_PATTERN.findall(text)
        assert len(matches) == expected_count, f"{rel}: expected {expected_count} guarded reset_peak_memory_stats call(s), found {len(matches)}"


def test_no_remaining_unguarded_reset_peak_memory_stats_in_agent_package():
    # tell.agent.local_model is intentionally excluded: ModelRuntime.load()
    # is the actual Qwen3 GPU loader, which has no CPU code path at all --
    # it guards by raising RuntimeError("CUDA is not available...") a few
    # lines earlier in the same function if CUDA is absent, rather than
    # wrapping this one call in an `if`. That is a different, equally
    # valid guard idiom (pre-existing, not part of Part 4's fix), not an
    # unguarded call; see the file for the raise.
    excluded = {REPO / "src" / "tell" / "agent" / "local_model.py"}
    for path in (REPO / "src" / "tell" / "agent").glob("*.py"):
        if path in excluded:
            assert "CUDA is not available" in path.read_text()
            continue
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if "torch.cuda.reset_peak_memory_stats()" not in line:
                continue
            prev = lines[i - 1].strip() if i > 0 else ""
            assert prev == "if torch.cuda.is_available():", f"{path}:{i + 1} is not guarded"


def test_no_conftest_or_cuda_shim_exists_in_repo():
    # Part 4 forbids a global monkeypatch or scratchpad plugin. Assert none
    # exists anywhere this repo's own test suite could pick up.
    assert not list((REPO / "tests").rglob("conftest.py"))
    assert not list(REPO.glob("conftest.py"))
    pyproject = (REPO / "pyproject.toml").read_text()
    assert "addopts" not in pyproject
    assert "PYTEST_PLUGINS" not in os.environ


def test_full_cpu_suite_passes_with_cuda_hidden_and_no_plugin():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         "tests/test_agent_loop.py", "tests/test_conditional_retrieval.py", "tests/test_memory_pilot.py", "tests/test_prompt_profiles.py"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


# ---------------------------------------------------------------------
# 17: CUDA behavior unchanged when available (mocked -- this environment's
# installed PyTorch build does not support this host's GPU compute
# capability, so we cannot rely on a real CUDA kernel dispatch here; the
# guard idiom itself, exactly as used at all three call sites, is what's
# under test).
# ---------------------------------------------------------------------


def _guarded_call(torch_module) -> None:
    """The exact two-line idiom used at all three call sites (see
    GUARD_PATTERN above) -- tested in isolation against a mocked
    torch.cuda so both branches are exercised deterministically."""
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()


def test_guard_calls_reset_when_cuda_available():
    import torch

    with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "reset_peak_memory_stats") as mock_reset:
        _guarded_call(torch)
    mock_reset.assert_called_once()


def test_guard_skips_reset_when_cuda_unavailable():
    import torch

    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(torch.cuda, "reset_peak_memory_stats") as mock_reset:
        _guarded_call(torch)
    mock_reset.assert_not_called()
