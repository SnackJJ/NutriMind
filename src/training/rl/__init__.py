"""NutriMind v2.0 RL stage (spec: ``.scratch/nutrimind-rl/spec.md``).

Consumes factory ``TaskPackage`` / ``EpisodeResult``. Does not author tasks.
The package import is side-effect-free and does not import ``nutrienv``.
"""

from .exam_gate import (
    ExamGateError,
    assert_disjoint_from_exam,
    assert_exam_byte_identical,
    before_eval_rollout,
    exam_task_ids,
)

__all__ = [
    "ExamGateError",
    "assert_disjoint_from_exam",
    "assert_exam_byte_identical",
    "before_eval_rollout",
    "exam_task_ids",
]
