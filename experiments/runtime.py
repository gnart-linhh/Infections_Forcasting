"""Atomic progress files and streaming checksums for long experiments."""
import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def acquire_run_lock(path):
    """Keep the returned handle alive for the lifetime of the run."""
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=path.open('a+')
    try:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'Another process is already using {path.parent}') from None
    handle.seek(0);handle.truncate();handle.write(str(os.getpid())+'\n');handle.flush()
    return handle
