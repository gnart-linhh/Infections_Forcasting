"""Ordered process-based generation and compact checkpoints without pickle."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path

import numpy as np
from generator import SyntheticCurve, gen_curve

FAMILIES = ('sir_rollercoaster', 'sir_rollercoaster_wiggle', 'seasonal')
PUBLISHED_CURVES_PER_FAMILY = 18387


def generate_one(task):
    family_index, index, key = task # family_index: chỉ số của họ đường cong, index: chỉ số của đường cong trong họ, key: khóa ngẫu nhiên để tạo ra đường cong
    curve = gen_curve(FAMILIES[family_index], np.random.default_rng(np.random.SeedSequence(key))) 
    valid = curve is not None and len(curve.ts) > 0 and np.all(np.isfinite(curve.ts)) and np.all(curve.ts > 0) #Kiểm tra xem đường cong có hợp lệ hay không: không rỗng, tất cả các giá trị là hữu hạn và dương
    return curve if valid else None


def generate_ordered(tasks, workers=1):
    if workers == 1: 
        yield from map(generate_one, tasks)
    else:
        # LSODA is not thread-safe; independent processes keep RNG and solver isolated.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            yield from pool.map(generate_one, tasks, chunksize=32)


def save_packed(curves, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.r_[0, np.cumsum([len(c.ts) for c in curves])]
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, values=np.concatenate([c.ts for c in curves]), offsets=offsets,
                            curve_type=[c.curve_type for c in curves], time_cadence=[c.time_cadence for c in curves],
                            scale=[c.scale for c in curves], seasonal=[c.seasonal for c in curves])
    temporary.replace(path)


def load_packed(path):
    with np.load(path, allow_pickle=False) as data:
        values = data['values']; offsets = data['offsets']
        return [SyntheticCurve(values[offsets[i]:offsets[i+1]], str(family), str(cadence), str(scale), bool(seasonal))
                for i, (family, cadence, scale, seasonal) in enumerate(zip(
                    data['curve_type'], data['time_cadence'], data['scale'], data['seasonal']))]


def write_audit(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with gzip.open(temporary, 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    temporary.replace(path)


def read_audit(path):
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle]
