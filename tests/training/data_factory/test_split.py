"""Ticket 020 split helper — 7:2:1 by task_id hash, reproducible, no overlap."""

from __future__ import annotations

from src.training.data_factory.build import split_by_task_id


def test_split_is_stable_and_partitions():
    ids = [f"log--log--train-alba--{i:06d}" for i in range(200)]
    first = [split_by_task_id(task_id) for task_id in ids]
    second = [split_by_task_id(task_id) for task_id in ids]
    assert first == second
    buckets = {"train": [], "holdout": [], "loss_val": []}
    for task_id, name in zip(ids, first):
        buckets[name].append(task_id)
    assert set(buckets) == {"train", "holdout", "loss_val"}
    seen = []
    for rows in buckets.values():
        seen.extend(rows)
    assert sorted(seen) == sorted(ids)
    assert len(set(seen)) == len(ids)
    # roughly 7:2:1
    assert 0.55 <= len(buckets["train"]) / len(ids) <= 0.85
    assert 0.05 <= len(buckets["holdout"]) / len(ids) <= 0.35
    assert 0.02 <= len(buckets["loss_val"]) / len(ids) <= 0.25
