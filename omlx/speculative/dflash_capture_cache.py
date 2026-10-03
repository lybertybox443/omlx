# SPDX-License-Identifier: Apache-2.0
"""Bounded, versioned DFlash hidden-capture sidecars at exact KV boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path


def checkpoint_identity(path):
    """Invalidate snapshots when the local checkpoint or its configuration changes."""
    root = Path(path).resolve()
    files = sorted([*root.glob("*.safetensors"), *root.glob("*.json")])
    return [
        str(root),
        [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files],
    ]


class DFlashCaptureStore:
    def __init__(
        self,
        identity,
        *,
        max_entries=4,
        max_bytes=8 * 1024**3,
        directory=None,
        disk_bytes=20 * 1024**3,
    ):
        self.identity = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self.disk_bytes = max(0, int(disk_bytes))
        self.root = Path(directory) if directory else None
        self.directory = self.root / self.identity if self.root is not None else None
        self.entries = OrderedDict()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._prune_disk()

    def _key(self, tokens, boundary, media):
        if boundary <= 0 or boundary > len(tokens):
            return None
        try:
            data = json.dumps(
                [1, self.identity, list(tokens[:boundary]), media], sort_keys=True
            )
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(data.encode()).hexdigest()

    def _remember(self, key, snapshot):
        size = sum(value.nbytes for value in snapshot.values())
        if not self.max_entries or size > self.max_bytes:
            return
        self.entries[key] = snapshot
        self.entries.move_to_end(key)
        while (
            len(self.entries) > self.max_entries
            or sum(
                value.nbytes
                for entry in self.entries.values()
                for value in entry.values()
            )
            > self.max_bytes
        ):
            self.entries.popitem(last=False)

    def _prune_disk(self):
        files = sorted(
            self.root.glob("*/*.safetensors"), key=lambda p: p.stat().st_mtime_ns
        )
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self.disk_bytes:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)

    def put(self, tokens, boundary, media, snapshot):
        import mlx.core as mx

        key = self._key(tokens, boundary, media)
        if key is None:
            return
        snapshot = {name: value + 0 for name, value in snapshot.items()}
        mx.eval(list(snapshot.values()))
        self._remember(key, snapshot)
        if self.directory is None or not self.disk_bytes:
            return
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(suffix=".safetensors", dir=self.directory)
            os.close(fd)
            mx.save_safetensors(temporary, snapshot, {"version": "1", "key": key})
            os.replace(temporary, self.directory / (key + ".safetensors"))
            self._prune_disk()
        except (OSError, ValueError, RuntimeError):
            # A cache write must not invalidate otherwise correct generation.
            pass
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def clear(self, *, memory=True, disk=False):
        cleared = len(self.entries) if memory else 0
        if memory:
            self.entries.clear()
        deleted = self.clear_disk(self.root) if disk and self.root is not None else 0
        return {"capture_hot_cleared": cleared, "capture_ssd_deleted": deleted}

    @staticmethod
    def clear_disk(directory):
        """Clear all configurations in this dedicated capture-cache directory."""
        root = Path(directory)
        deleted = 0
        for namespace in root.iterdir() if root.exists() else ():
            if namespace.is_symlink() or not namespace.is_dir():
                continue
            for path in namespace.glob("*.safetensors"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                deleted += 1
        return deleted

    def get(self, tokens, boundary, media):
        import mlx.core as mx

        key = self._key(tokens, boundary, media)
        if key is None:
            return None
        snapshot = self.entries.get(key)
        if snapshot is not None:
            self.entries.move_to_end(key)
        elif self.directory is not None:
            path = self.directory / (key + ".safetensors")
            try:
                if path.stat().st_size > self.disk_bytes:
                    return None
                snapshot, metadata = mx.load(str(path), return_metadata=True)
                if metadata != {"version": "1", "key": key}:
                    return None
                mx.eval(list(snapshot.values()))
                os.utime(path, None)
                self._remember(key, snapshot)
            except (OSError, ValueError, RuntimeError):
                return None
        return (
            None
            if snapshot is None
            else {name: value + 0 for name, value in snapshot.items()}
        )
