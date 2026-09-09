"""Typed config schema + loader for ``configs/data_factory.yaml`` (spec §7).

The loader is strict: a missing key, a mistyped key, or an unknown key raises
``ConfigError`` with the dotted path of the offending key — config errors fail the
whole run before any teacher spend (spec §4.1). Validation logic lives HERE, not
in the concept-types module.

Single-sourcing notes:

- ``nutrienv_rev`` (spec §7) is the ``rev`` of the ticket-001 ``nutrienv:`` pin
  block, exposed as ``DataFactoryConfig.nutrienv_rev``. There is deliberately no
  top-level ``nutrienv_rev`` key — two spellings of the pin would drift.
- ``catalog_path`` / ``exam_split_path`` may be ``null``: the build stage resolves
  them to the nutrienv public defaults (``load_catalog()``'s gold catalog and
  ``nutrienv.bench.EXAM_SPLIT_PATH``). This module never imports nutrienv.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "DataFactoryConfig",
    "ExpanderConfig",
    "FamilyConfig",
    "NutriEnvPin",
    "TeacherConfig",
    "config_from_dict",
    "load_config",
]


class ConfigError(ValueError):
    """A data-factory config is missing a key, mistyped, or unknown."""


_REV_RE = re.compile(r"^[0-9a-f]{40}$")

_TARGETS = {"sft", "rlvr", "eval", "all"}
_BUDGET_ACTIONS = {"warn", "stop"}
_THINKING_TYPES = {"enabled", "disabled"}

_TOP_LEVEL_KEYS = {
    "nutrienv",
    "target",
    "teacher",
    "expander",
    "families",
    "max_seq_tokens",
    "plan_max_tokens",
    "tokenizer_name",
    "max_intents",
    "usd_budget",
    "on_budget",
    "output_dir",
    "rubric_version",
    "reward_version",
    "catalog_path",
    "exam_split_path",
}
_TEACHER_KEYS = {
    "model_id",
    "endpoint",
    "credential_env",
    "thinking",
    "temperature_first",
    "temperature_retry",
    "per_turn_timeout_s",
}
_EXPANDER_KEYS = {
    "model_id",
    "endpoint",
    "credential_env",
    "thinking",
    "timeout_s",
    "parse_retries",
}
_FAMILY_KEYS = {"target_n", "teacher_k", "over_generate_x", "gram_anchor"}
_FAMILY_OPTIONAL_KEYS = {"amount_path_weights"}
_NUTRIENV_KEYS = {"repo", "rev", "install", "note"}

# spec §16: k is a per-family max-attempts value in 1..6.
_TEACHER_K_MIN, _TEACHER_K_MAX = 1, 6


# --------------------------------------------------------------------------- #
# typed schema
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class NutriEnvPin:
    """The ticket-001 dependency pin block — byte-frozen in the yaml."""

    repo: str
    rev: str
    install: str
    note: str | None = None


@dataclasses.dataclass(frozen=True)
class TeacherConfig:
    """Teacher call shape (ADR-011 amended 2026-09-09: ark/deepseek-v4-flash on
    api/plan/v3, ``thinking: {"type": "enabled"}`` as the length control)."""

    model_id: str
    endpoint: str
    credential_env: str
    thinking: dict[str, str]
    temperature_first: float  # attempt 1
    temperature_retry: float  # attempts 2..k
    per_turn_timeout_s: float


@dataclasses.dataclass(frozen=True)
class ExpanderConfig:
    """Expander call shape — same endpoint + credential as the teacher (one
    provider); ``thinking: {"type": "disabled"}`` (structured JSON output)."""

    model_id: str
    endpoint: str
    credential_env: str
    thinking: dict[str, str]
    timeout_s: float
    parse_retries: int  # attempts = parse_retries + 1 (spec §16)


@dataclasses.dataclass(frozen=True)
class FamilyConfig:
    """Per-family authoring knobs (spec §7). ``target_n`` counts accepted Pass
    ``task_id``s after intra-family ``semantic_key`` dedup (spec §10)."""

    target_n: int
    teacher_k: int  # max teacher attempts per task_id, 1..6 (spec §16)
    over_generate_x: float
    gram_anchor: bool
    amount_path_weights: dict[str, float] | None = None


@dataclasses.dataclass(frozen=True)
class DataFactoryConfig:
    """Resolved, validated data-factory config (spec §7)."""

    target: str
    teacher: TeacherConfig
    expander: ExpanderConfig
    families: dict[str, FamilyConfig]
    max_seq_tokens: int
    plan_max_tokens: int
    tokenizer_name: str | None
    max_intents: int
    usd_budget: float
    on_budget: str
    output_dir: str
    rubric_version: str
    reward_version: str
    catalog_path: str | None
    exam_split_path: str | None
    nutrienv: NutriEnvPin

    @property
    def nutrienv_rev(self) -> str:
        """Spec §7's ``nutrienv_rev`` — single-sourced from the pin block."""
        return self.nutrienv.rev

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #


def _err(source: str, path: str, message: str) -> None:
    raise ConfigError(f"{source}: {path}: {message}")


def _mapping(value: Any, source: str, path: str) -> dict:
    if not isinstance(value, dict):
        _err(source, path, f"expected a mapping, got {type(value).__name__}")
    return value


def _str(value: Any, source: str, path: str) -> str:
    if not isinstance(value, str) or not value:
        _err(source, path, f"expected a non-empty string, got {value!r}")
    return value


def _str_or_none(value: Any, source: str, path: str) -> str | None:
    return None if value is None else _str(value, source, path)


def _number(value: Any, source: str, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _err(source, path, f"expected a number, got {type(value).__name__} ({value!r})")
    return float(value)


def _int(value: Any, source: str, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _err(source, path, f"expected an int, got {type(value).__name__} ({value!r})")
    return value


def _bool(value: Any, source: str, path: str) -> bool:
    if not isinstance(value, bool):
        _err(source, path, f"expected a bool, got {type(value).__name__} ({value!r})")
    return value


def _no_unknown_keys(data: dict, allowed: set[str], source: str, path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        _err(source, path, f"unknown key(s): {', '.join(unknown)}")


def _required(data: dict, key: str, source: str, parent: str) -> Any:
    """Assert ``key`` is present; report at the full dotted path of the key."""
    if key not in data:
        _err(source, f"{parent}.{key}" if parent else key, "missing required key")
    return data[key]


def _thinking(value: Any, source: str, path: str) -> dict[str, str]:
    thinking = _mapping(value, source, path)
    _no_unknown_keys(thinking, {"type"}, source, path)
    kind = _str(_required(thinking, "type", source, path), source, f"{path}.type")
    if kind not in _THINKING_TYPES:
        _err(source, f"{path}.type", f"expected one of {sorted(_THINKING_TYPES)}, got {kind!r}")
    return {"type": kind}


# --------------------------------------------------------------------------- #
# section parsers
# --------------------------------------------------------------------------- #


def _parse_nutrienv(value: Any, source: str) -> NutriEnvPin:
    path = "nutrienv"
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _NUTRIENV_KEYS, source, path)
    rev = _str(_required(block, "rev", source, path), source, f"{path}.rev")
    if not _REV_RE.match(rev):
        _err(source, f"{path}.rev", f"expected a 40-hex git rev, got {rev!r}")
    return NutriEnvPin(
        repo=_str(_required(block, "repo", source, path), source, f"{path}.repo"),
        rev=rev,
        install=_str(_required(block, "install", source, path), source, f"{path}.install"),
        note=_str_or_none(block.get("note"), source, f"{path}.note"),
    )


def _parse_teacher(value: Any, source: str) -> TeacherConfig:
    path = "teacher"
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _TEACHER_KEYS, source, path)
    timeout = _number(
        _required(block, "per_turn_timeout_s", source, path), source, f"{path}.per_turn_timeout_s"
    )
    if timeout <= 0:
        _err(source, f"{path}.per_turn_timeout_s", f"must be > 0, got {timeout}")
    return TeacherConfig(
        model_id=_str(_required(block, "model_id", source, path), source, f"{path}.model_id"),
        endpoint=_str(_required(block, "endpoint", source, path), source, f"{path}.endpoint"),
        credential_env=_str(
            _required(block, "credential_env", source, path), source, f"{path}.credential_env"
        ),
        thinking=_thinking(_required(block, "thinking", source, path), source, f"{path}.thinking"),
        temperature_first=_number(
            _required(block, "temperature_first", source, path),
            source,
            f"{path}.temperature_first",
        ),
        temperature_retry=_number(
            _required(block, "temperature_retry", source, path),
            source,
            f"{path}.temperature_retry",
        ),
        per_turn_timeout_s=timeout,
    )


def _parse_expander(value: Any, source: str) -> ExpanderConfig:
    path = "expander"
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _EXPANDER_KEYS, source, path)
    timeout = _number(_required(block, "timeout_s", source, path), source, f"{path}.timeout_s")
    if timeout <= 0:
        _err(source, f"{path}.timeout_s", f"must be > 0, got {timeout}")
    parse_retries = _int(
        _required(block, "parse_retries", source, path), source, f"{path}.parse_retries"
    )
    if parse_retries < 0:
        _err(source, f"{path}.parse_retries", f"must be >= 0, got {parse_retries}")
    return ExpanderConfig(
        model_id=_str(_required(block, "model_id", source, path), source, f"{path}.model_id"),
        endpoint=_str(_required(block, "endpoint", source, path), source, f"{path}.endpoint"),
        credential_env=_str(
            _required(block, "credential_env", source, path), source, f"{path}.credential_env"
        ),
        thinking=_thinking(_required(block, "thinking", source, path), source, f"{path}.thinking"),
        timeout_s=timeout,
        parse_retries=parse_retries,
    )


def _parse_family(name: str, value: Any, source: str) -> FamilyConfig:
    path = f"families.{name}"
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _FAMILY_KEYS | _FAMILY_OPTIONAL_KEYS, source, path)
    target_n = _int(_required(block, "target_n", source, path), source, f"{path}.target_n")
    if target_n < 0:
        _err(source, f"{path}.target_n", f"must be >= 0, got {target_n}")
    teacher_k = _int(_required(block, "teacher_k", source, path), source, f"{path}.teacher_k")
    if not _TEACHER_K_MIN <= teacher_k <= _TEACHER_K_MAX:
        _err(
            source,
            f"{path}.teacher_k",
            f"must be in {_TEACHER_K_MIN}..{_TEACHER_K_MAX} (spec §16), got {teacher_k}",
        )
    over_generate_x = _number(
        _required(block, "over_generate_x", source, path), source, f"{path}.over_generate_x"
    )
    if over_generate_x <= 0:
        _err(source, f"{path}.over_generate_x", f"must be > 0, got {over_generate_x}")
    weights_raw = block.get("amount_path_weights")
    weights = None
    if weights_raw is not None:
        weights = {}
        for key, weight in _mapping(weights_raw, source, f"{path}.amount_path_weights").items():
            w = _number(weight, source, f"{path}.amount_path_weights.{key}")
            if w < 0:
                _err(source, f"{path}.amount_path_weights.{key}", f"must be >= 0, got {w}")
            weights[_str(key, source, f"{path}.amount_path_weights")] = w
    return FamilyConfig(
        target_n=target_n,
        teacher_k=teacher_k,
        over_generate_x=over_generate_x,
        gram_anchor=_bool(
            _required(block, "gram_anchor", source, path), source, f"{path}.gram_anchor"
        ),
        amount_path_weights=weights,
    )


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #


def config_from_dict(data: Any, *, source: str = "<dict>") -> DataFactoryConfig:
    """Validate a parsed config mapping into a ``DataFactoryConfig``."""
    root = _mapping(data, source, "")
    _no_unknown_keys(root, _TOP_LEVEL_KEYS, source, "")

    target = _str(_required(root, "target", source, ""), source, "target")
    if target not in _TARGETS:
        _err(source, "target", f"expected one of {sorted(_TARGETS)}, got {target!r}")

    families_raw = _mapping(_required(root, "families", source, ""), source, "families")
    if not families_raw:
        _err(source, "families", "must list at least one family")
    families = {
        _str(name, source, "families"): _parse_family(name, value, source)
        for name, value in families_raw.items()
    }

    max_seq_tokens = _int(_required(root, "max_seq_tokens", source, ""), source, "max_seq_tokens")
    if max_seq_tokens <= 0:
        _err(source, "max_seq_tokens", f"must be > 0, got {max_seq_tokens}")
    plan_max_tokens = _int(
        _required(root, "plan_max_tokens", source, ""), source, "plan_max_tokens"
    )
    if plan_max_tokens <= 0:
        _err(source, "plan_max_tokens", f"must be > 0, got {plan_max_tokens}")
    max_intents = _int(_required(root, "max_intents", source, ""), source, "max_intents")
    if max_intents <= 0:
        _err(source, "max_intents", f"must be > 0, got {max_intents}")
    usd_budget = _number(_required(root, "usd_budget", source, ""), source, "usd_budget")
    if usd_budget < 0:
        _err(source, "usd_budget", f"must be >= 0, got {usd_budget}")
    on_budget = _str(_required(root, "on_budget", source, ""), source, "on_budget")
    if on_budget not in _BUDGET_ACTIONS:
        _err(source, "on_budget", f"expected one of {sorted(_BUDGET_ACTIONS)}, got {on_budget!r}")

    return DataFactoryConfig(
        target=target,
        teacher=_parse_teacher(_required(root, "teacher", source, ""), source),
        expander=_parse_expander(_required(root, "expander", source, ""), source),
        families=families,
        max_seq_tokens=max_seq_tokens,
        plan_max_tokens=plan_max_tokens,
        tokenizer_name=_str_or_none(root.get("tokenizer_name"), source, "tokenizer_name"),
        max_intents=max_intents,
        usd_budget=usd_budget,
        on_budget=on_budget,
        output_dir=_str(_required(root, "output_dir", source, ""), source, "output_dir"),
        rubric_version=_str(
            _required(root, "rubric_version", source, ""), source, "rubric_version"
        ),
        reward_version=_str(
            _required(root, "reward_version", source, ""), source, "reward_version"
        ),
        catalog_path=_str_or_none(root.get("catalog_path"), source, "catalog_path"),
        exam_split_path=_str_or_none(root.get("exam_split_path"), source, "exam_split_path"),
        nutrienv=_parse_nutrienv(_required(root, "nutrienv", source, ""), source),
    )


def load_config(path: str | pathlib.Path) -> DataFactoryConfig:
    """Load + validate ``configs/data_factory.yaml`` (or any config of its schema)."""
    path = pathlib.Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:  # unreadable / missing file
        raise ConfigError(f"{path}: cannot read config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return config_from_dict(raw, source=str(path))
