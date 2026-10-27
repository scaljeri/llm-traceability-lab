"""Explicit GGUF registration; inference only rechecks cached file/process bindings."""

import argparse
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


def file_identity(value) -> dict:
    return {key: getattr(value, "st_" + key) for key in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def hash_artifact(path: str) -> dict:
    """Hash the entire regular GGUF once, detecting replacement during the read."""
    resolved = Path(path).expanduser().resolve(strict=True)
    with resolved.open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or source.read(4) != b"GGUF":
            raise ValueError("Select a regular GGUF artifact file.")
        source.seek(0)
        digest = hashlib.sha256()
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(source.fileno())
    if file_identity(before) != file_identity(after) or file_identity(after) != file_identity(
        resolved.stat()
    ):
        raise ValueError("Artifact changed during hashing; verify again.")
    return {
        "path": str(resolved),
        "filename": resolved.name,
        "sha256": digest.hexdigest(),
        "size_bytes": after.st_size,
        "file_stat": file_identity(after),
        "verified_at": datetime.now(UTC).isoformat(),
    }


def process_binding(path: str, pid: int, server_url: str, *, proc_root: Path = Path("/proc")) -> dict:
    """Require a Linux mapping of this inode and ownership of the local API listener."""
    url = urlparse(server_url)
    if url.hostname not in {"127.0.0.1", "::1"}:
        raise ValueError("Local process verification requires a loopback server URL.")
    root = proc_root / str(pid)
    if os.readlink(root / "ns/net") != os.readlink(proc_root / "self/ns/net"):
        raise ValueError("The API listener must be in the application's network namespace.")
    state = (root / "stat").read_text().rsplit(")", 1)[1].split()
    if state[0] in {"Z", "X", "x"}:
        raise ValueError("The server process is not running.")
    start = state[19]  # Field 22; state starts at field 3.
    value = Path(path).stat()
    mapped = False
    for line in (root / "maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) < 6 or fields[5].endswith(" (deleted)"):
            continue
        major, minor = (int(part, 16) for part in fields[3].split(":"))
        if int(fields[4]) == value.st_ino and (major, minor) == (
            os.major(value.st_dev),
            os.minor(value.st_dev),
        ):
            mapped = True
    if not mapped:
        raise ValueError(
            "The artifact inode is not mapped by this process; arguments alone are insufficient."
        )
    sockets = {os.readlink(entry) for entry in (root / "fd").iterdir()}
    port = url.port or (443 if url.scheme == "https" else 80)
    listening = False
    for protocol in ("tcp",) if url.hostname == "127.0.0.1" else ("tcp6",):
        table = root / "net" / protocol
        if not table.exists():
            continue
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            allowed_addresses = (
                {"00000000", "0100007F"}
                if protocol == "tcp"
                else {"0" * 32, "00000000000000000000000001000000"}
            )
            if (
                fields[1].split(":")[0] in allowed_addresses
                and fields[3] == "0A"
                and int(fields[1].split(":")[1], 16) == port
                and f"socket:[{fields[9]}]" in sockets
            ):
                listening = True
    if not listening:
        raise ValueError("The process does not own the configured server listener.")
    arguments = (root / "cmdline").read_bytes().decode(errors="replace").split("\0")
    allowed = {
        "-m",
        "--model",
        "-c",
        "--ctx-size",
        "-ngl",
        "--gpu-layers",
        "--n-gpu-layers",
        "-np",
        "--parallel",
        "-t",
        "--threads",
        "--tensor-split",
        "--split-mode",
        "--chat-template-file",
        "--flash-attn",
    }
    selected = {}
    for index, argument in enumerate(arguments):
        flag, separator, inline = argument.partition("=")
        if flag in allowed:
            selected[flag] = (
                inline if separator else arguments[index + 1] if index + 1 < len(arguments) else None
            )
    if (root / "stat").read_text().rsplit(")", 1)[1].split()[19] != start or file_identity(
        Path(path).stat()
    ) != file_identity(value):
        raise ValueError("Process or artifact changed during association checking.")
    return {
        "pid": pid,
        "start_time_ticks": start,
        "boot_id": (proc_root / "sys/kernel/random/boot_id").read_text().strip(),
        "executable": os.readlink(root / "exe"),
        "inference_arguments": selected,
    }


def recheck_artifact(receipt: dict, profile: dict, props: dict | None) -> dict:
    """Cheap stat/process checks only. Never hash a GGUF during inference."""
    try:
        if receipt["url"] != profile["url"] or receipt["api_model_id"] != profile["model"]:
            raise ValueError("Profile routing changed; explicit re-verification required.")
        if file_identity(Path(receipt["path"]).stat()) != receipt["file_stat"]:
            raise ValueError("Artifact file changed; explicit re-verification required.")
        old = receipt.get("process")
        if not old:
            return {
                "status": "file_verified",
                "reason": receipt.get("association_error")
                or "File hash verified at registration; loaded process not associated.",
            }
        current = process_binding(receipt["path"], old["pid"], profile["url"])
        if current != old:
            raise ValueError("Server process/restart changed; explicit re-verification required.")
        if not props or props.get("is_sleeping") or props.get("model_path") != receipt["path"]:
            return {
                "status": "file_verified",
                "reason": "Current server model path/loading state could not be associated.",
            }
        return {
            "status": "loaded_verified",
            "process": current,
            "reason": "Unchanged hashed file, mapped inode, process instance and local API listener verified.",
        }
    except (OSError, ValueError, KeyError, IndexError) as error:
        return {"status": "stale", "reason": str(error)}


def main():
    parser = argparse.ArgumentParser(
        description="Hash a complete GGUF explicitly and emit a portable deployment manifest."
    )
    parser.add_argument("path")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--engine", help="Declared engine name, only when known")
    parser.add_argument("--revision")
    arguments = parser.parse_args()
    receipt = hash_artifact(arguments.path)
    if arguments.pid:
        receipt["server_url"] = arguments.server_url
        try:
            receipt["process"] = process_binding(receipt["path"], arguments.pid, arguments.server_url)
        except (OSError, ValueError, IndexError) as error:
            receipt["association_error"] = str(error)
    if file_identity(Path(receipt["path"]).stat()) != receipt["file_stat"]:
        raise ValueError("Artifact changed during registration; verify again.")
    print(
        json.dumps(
            {
                "artifact": {key: receipt[key] for key in ("filename", "sha256", "size_bytes")},
                "runtime": {
                    "engine": arguments.engine,
                    "revision": arguments.revision,
                    "configuration": {
                        "process_arguments": (receipt.get("process") or {}).get("inference_arguments", {})
                    },
                },
                "provenance": "explicit inference-host GGUF registration",
                "verification": receipt,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
