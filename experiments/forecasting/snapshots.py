"""Strict historical snapshots: never substitute final truth for a missing vintage."""
from __future__ import annotations
import csv
from datetime import date
from pathlib import Path
import numpy as np


class SnapshotTruth:
    def __init__(self, path: Path):
        self.snapshots = {}
        with path.open() as handle:
            for row in csv.DictReader(handle):
                vintage = date.fromisoformat(row['as_of'])
                target = date.fromisoformat(row['target_end_date'])
                value = float(row['value'])
                if target > vintage or not np.isfinite(value) or value < 0:
                    raise ValueError('Snapshot targets must be <= as_of; values finite and nonnegative')
                snapshot = self.snapshots.setdefault(vintage, {})
                if target in snapshot:
                    raise ValueError(f'Duplicate snapshot target: {vintage}, {target}')
                snapshot[target] = max(1.0, value)
        if not self.snapshots:
            raise ValueError('Empty snapshot archive')

    def history(self, origin, expected_dates):
        # as_of denotes the exact information cutoff, not the retrieval date.
        # Requiring an exact vintage prevents unannounced stale/future substitutions.
        if origin not in self.snapshots:
            raise ValueError(f'Missing exact as_of snapshot for {origin}')
        snapshot = self.snapshots[origin]
        if set(snapshot) != set(expected_dates):
            raise ValueError(f'Snapshot history must match all weekly dates through {origin}')
        return np.asarray([snapshot[d] for d in expected_dates])


def calibration_rows(prior, origin, history_dates, history):
    """Refresh matured outcomes from the current vintage, not from scoring truth."""
    observed = {d.isoformat(): float(v) for d, v in zip(history_dates, history)}
    cutoff = origin.isoformat()
    result = []
    for row in prior:
        if row['target_end_date'] <= cutoff:
            if row['target_end_date'] not in observed:
                raise ValueError('Matured forecast outcome missing from current vintage')
            result.append(dict(row, true_value=observed[row['target_end_date']]))
        else:
            # Keep issued row count for the six-origin warmup switch. Its value
            # cannot enter dispersion because target_end_date is after origin.
            result.append(dict(row, true_value=float('nan')))
    return result
