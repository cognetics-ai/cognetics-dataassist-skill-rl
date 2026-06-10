#!/usr/bin/env python3
"""Report whether the current host can run SkillSQL-RL GRPO training.

macOS is useful for catalog work, rollouts, rewards, and artifact generation.
The verl/vLLM weight-update path is expected to run on a Linux host/container
with an NVIDIA CUDA GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SkillSQL-RL training runtime")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if CUDA GPU is unavailable",
    )
    parser.add_argument(
        "--require-verl",
        action="store_true",
        help="Fail if verl/vLLM are unavailable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report()

    failures: list[str] = []
    if args.require_gpu and not report["gpu"]["cuda_visible"]:
        failures.append("CUDA GPU is not visible")
    if args.require_verl and not report["python_packages"]["verl"]["installed"]:
        failures.append("verl is not importable")
    if args.require_verl and not report["python_packages"]["vllm"]["installed"]:
        failures.append("vLLM is not importable")

    report["ok"] = not failures
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


def build_report() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine()
    torch_info = _torch_info()
    vllm_info = _module_info("vllm")
    verl_info = _module_info("verl")
    docker = _command_probe(["docker", "version", "--format", "{{.Server.Version}}"])
    nvidia_smi = _command_probe(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])

    cuda_visible = bool(nvidia_smi["ok"] and torch_info.get("cuda_available"))
    training_ready = bool(cuda_visible and vllm_info["installed"] and verl_info["installed"])

    return {
        "host": {
            "system": system,
            "machine": machine,
            "python": sys.version.split()[0],
            "is_macos": system == "Darwin",
            "is_linux": system == "Linux",
        },
        "docker": docker,
        "gpu": {
            "nvidia_smi": nvidia_smi,
            "torch_cuda_available": torch_info.get("cuda_available"),
            "torch_cuda_device_count": torch_info.get("cuda_device_count"),
            "cuda_visible": cuda_visible,
        },
        "python_packages": {
            "torch": torch_info,
            "vllm": vllm_info,
            "verl": verl_info,
        },
        "mode": "verl-vllm-training" if training_ready else "dry-run-artifact-only",
        "training_ready": training_ready,
        "recommendation": _recommendation(
            system=system,
            docker_ok=bool(docker["ok"]),
            cuda_visible=cuda_visible,
            vllm_installed=bool(vllm_info["installed"]),
            verl_installed=bool(verl_info["installed"]),
        ),
    }


def _module_info(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {"installed": spec is not None, "origin": spec.origin if spec else None}


def _torch_info() -> dict[str, Any]:
    info = _module_info("torch")
    if not info["installed"]:
        return {**info, "cuda_available": False, "cuda_device_count": 0}

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {
            **info,
            "import_error": str(exc),
            "cuda_available": False,
            "cuda_device_count": 0,
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
    return {
        **info,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": devices,
    }


def _command_probe(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return {"ok": False, "command": cmd[0], "error": "not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "command": cmd[0], "error": "timeout"}

    output = (result.stdout or result.stderr or "").strip()
    return {
        "ok": result.returncode == 0,
        "command": cmd[0],
        "returncode": result.returncode,
        "output": output[:500],
    }


def _recommendation(
    *,
    system: str,
    docker_ok: bool,
    cuda_visible: bool,
    vllm_installed: bool,
    verl_installed: bool,
) -> str:
    if cuda_visible and vllm_installed and verl_installed:
        return "This host can run the verl/vLLM policy-update path."
    if system == "Darwin":
        return (
            "Use this Mac for dry-runs and GRPO artifact generation. Run verl/vLLM "
            "weight updates on a Linux NVIDIA CUDA GPU host or container."
        )
    if not docker_ok:
        return "Install or start Docker before using containerized training."
    if not cuda_visible:
        return "Expose an NVIDIA CUDA GPU to the host/container before running verl/vLLM training."
    return "Install the optional verl and vLLM packages in the GPU training environment."


if __name__ == "__main__":
    raise SystemExit(main())
