"""NutriMind v2.0 data factory (spec: ``.scratch/nutrimind-v2/spec.md``).

A task/data artifact factory, not an experiment runner (spec §4.1): author →
gate → materialize ``TaskPackage`` → per-target materialize (SFT record / RLVR
export / eval task). This package imports with no side effects and never imports
``nutrienv`` at module level — the stage modules that need nutrienv import it
lazily, so importing the factory never requires the benchmark to be installed.

Layout (grown ticket by ticket; ticket 003 ships the skeleton):

- ``concepts`` — the shared seam concept types (logic-free).
- ``config``   — the typed schema + strict loader for ``configs/data_factory.yaml``.
"""

from .concepts import (
    AttemptRecord,
    EpisodeResult,
    GateResult,
    Provenance,
    RewardSemantics,
    RolloutCache,
    TaskCatalogRef,
    TaskEnvironment,
    TaskOracle,
    TaskPackage,
    TaskVerifierRef,
    Termination,
    TurnMeta,
    VerificationResult,
)
from .config import (
    ConfigError,
    DataFactoryConfig,
    ExpanderConfig,
    FamilyConfig,
    NutriEnvPin,
    TeacherConfig,
    config_from_dict,
    load_config,
)

__all__ = [
    "AttemptRecord",
    "ConfigError",
    "DataFactoryConfig",
    "EpisodeResult",
    "ExpanderConfig",
    "FamilyConfig",
    "GateResult",
    "NutriEnvPin",
    "Provenance",
    "RewardSemantics",
    "RolloutCache",
    "TaskCatalogRef",
    "TaskEnvironment",
    "TaskOracle",
    "TaskPackage",
    "TaskVerifierRef",
    "TeacherConfig",
    "Termination",
    "TurnMeta",
    "VerificationResult",
    "config_from_dict",
    "load_config",
]
