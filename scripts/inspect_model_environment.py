"""Read-only model-runtime readiness inspector.

Bounded audit tool: inspects machine resources, discovers existing Python
environments, and (only where PyTorch is already installed) runs a tiny,
immediately-released CUDA allocate+matmul smoke test. Never installs,
upgrades, or removes any package; never downloads a model or dataset;
never prints token/credential values.

Writes:
  - results/runtime/model_environment_report.json (machine-readable)
This script does not write the .md report; that is authored separately so
it can cite external documentation (Qwen3-8B model card, PyTorch wheel
index) alongside these local findings.

Safe to re-run at any time, including after a future step installs
PyTorch into one of the candidate environments -- at that point the CUDA
smoke-test fields will populate for that environment instead of reporting
"not installed".
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/home/hp5/tell")
REPORT_JSON_PATH = PROJECT_ROOT / "results" / "runtime" / "model_environment_report.json"

CANDIDATE_ENVS: dict[str, Path] = {
    "system": Path("/usr/bin/python3"),
    ".venv": PROJECT_ROOT / ".venv" / "bin" / "python",
    ".venv-inspect": PROJECT_ROOT / ".venv-inspect" / "bin" / "python",
}

PACKAGES_TO_CHECK = [
    "transformers", "accelerate", "safetensors", "huggingface_hub",
    "peft", "bitsandbytes", "sentencepiece",
]

_PROBE_SCRIPT = """
import importlib.metadata as m
import json
import sys

info = {"executable": sys.executable, "python_version": sys.version.split()[0]}

try:
    import torch
    info["torch_installed"] = True
    info["torch_version"] = torch.__version__
    info["torch_cuda_version"] = torch.version.cuda
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{major}.{minor}"
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        props = torch.cuda.get_device_properties(0)
        info["total_memory_bytes"] = int(props.total_memory)
        try:
            a = torch.ones((256, 256), device="cuda", dtype=torch.float32)
            b = torch.ones((256, 256), device="cuda", dtype=torch.float32)
            c = a @ b
            info["cuda_tiny_matmul_ok"] = bool(torch.all(c == 256.0).item())
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            info["cuda_tiny_matmul_ok"] = False
            info["cuda_tiny_matmul_error"] = str(exc)
except ImportError:
    info["torch_installed"] = False
except Exception as exc:  # noqa: BLE001
    info["torch_installed"] = "error"
    info["torch_import_error"] = str(exc)

for pkg in %(packages)r:
    try:
        info[f"pkg_{pkg}"] = m.version(pkg)
    except m.PackageNotFoundError:
        info[f"pkg_{pkg}"] = None

print(json.dumps(info))
""" % {"packages": PACKAGES_TO_CHECK}


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (result.stdout or "") + (result.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"<unavailable: {exc}>"


def inspect_machine() -> dict:
    uname = platform.uname()
    os_release = {}
    os_release_path = Path("/etc/os-release")
    if os_release_path.exists():
        for line in os_release_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os_release[k] = v.strip('"')

    du = shutil.disk_usage(str(PROJECT_ROOT))
    mem_total_kb = mem_avail_kb = None
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        for line in meminfo_path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_avail_kb = int(line.split()[1])

    return {
        "hostname": socket.gethostname(),
        "cpu_architecture": uname.machine,
        "kernel_release": uname.release,
        "os_pretty_name": os_release.get("PRETTY_NAME"),
        "mem_total_bytes": mem_total_kb * 1024 if mem_total_kb else None,
        "mem_available_bytes": mem_avail_kb * 1024 if mem_avail_kb else None,
        "swap_and_free_raw": _run(["free", "-h"]),
        "disk_home_hp5_total_bytes": du.total,
        "disk_home_hp5_used_bytes": du.used,
        "disk_home_hp5_free_bytes": du.free,
        "nvidia_smi_raw": _run(["nvidia-smi"]),
        "nvidia_smi_query_raw": _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used,memory.free",
            "--format=csv",
        ]),
        "nvcc_version_raw": _run(["nvcc", "--version"]),
    }


def discover_environments() -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name, python_path in CANDIDATE_ENVS.items():
        if not python_path.exists():
            results[name] = {"exists": False, "path": str(python_path)}
            continue
        proc = subprocess.run(
            [str(python_path), "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, timeout=60,
        )
        entry: dict = {"exists": True, "path": str(python_path)}
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                entry.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except json.JSONDecodeError:
                entry["probe_error"] = proc.stdout + proc.stderr
        else:
            entry["probe_error"] = proc.stdout + proc.stderr
        results[name] = entry
    return results


def inspect_hf_cache() -> dict:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    result = {
        "hub_cache_dir": str(hub_dir),
        "hub_cache_dir_exists": hub_dir.exists(),
        "token_file_present": token_path.exists(),  # existence only, never contents
        "qwen3_8b_snapshots": [],
    }
    if hub_dir.exists():
        for entry in sorted(hub_dir.iterdir()):
            if "qwen3-8b" in entry.name.lower():
                size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                result["qwen3_8b_snapshots"].append({"path": entry.name, "size_bytes": size})
    return result


def main() -> None:
    report = {
        "machine": inspect_machine(),
        "python_environments": discover_environments(),
        "huggingface_cache": inspect_hf_cache(),
    }
    REPORT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_JSON_PATH}")

    for name, env in report["python_environments"].items():
        status = "torch installed" if env.get("torch_installed") is True else "no torch"
        print(f"  {name}: exists={env.get('exists')}, {status}")
    cache = report["huggingface_cache"]
    print(f"  HF cache present: {cache['hub_cache_dir_exists']}, Qwen3-8B snapshots: {len(cache['qwen3_8b_snapshots'])}")


if __name__ == "__main__":
    main()
