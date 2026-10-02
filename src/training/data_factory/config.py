"""Typed config schema + loader for ``configs/data_factory.yaml`` (spec §7).

The loader is strict: a missing key, a mistyped key, or an unknown key raises
``ConfigError`` with the dotted path of the offending key — config errors fail the
whole run before any teacher spend (spec §4.1). Validation logic lives HERE, not
in the concept-types module.

Single-sourcing notes:

- ``nutrienv_rev`` (spec §7) is the ``rev`` of the ``nutrienv:`` pin block
  (ticket 023 lab SHA), exposed as ``DataFactoryConfig.nutrienv_rev``. There is
  deliberately no top-level ``nutrienv_rev`` key — two spellings of the pin
  would drift.
- ``catalog_path`` / ``exam_split_path`` may be ``null``: the build stage resolves
  them to the nutrienv public defaults (``load_catalog()``'s gold catalog and
  ``nutrienv.bench.EXAM_SPLIT_PATH``). This module never imports nutrienv.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import re
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "ChannelConfig",
    "DataFactoryConfig",
    "ExpanderConfig",
    "FamilyConfig",
    "NutriEnvPin",
    "Pricing",
    "RateCard",
    "TeacherConfig",
    "TokenRates",
    "config_from_dict",
    "load_config",
]


class ConfigError(ValueError):
    """A data-factory config is missing a key, mistyped, or unknown."""


_REV_RE = re.compile(r"^[0-9a-f]{40}$")

_TARGETS = {"sft", "rlvr", "eval", "all"}
_BUDGET_ACTIONS = {"warn", "stop"}
_THINKING_TYPES = {"enabled", "disabled"}
_CURRENCIES = {"CNY", "USD"}

_TOP_LEVEL_KEYS = {
    "nutrienv",
    "target",
    "teacher",
    "expander",
    "families",
    "max_seq_tokens",
    "plan_max_tokens",
    "final_plan_max_tokens",
    "seed_offset",
    "tokenizer_name",
    "max_intents",
    "usd_budget",
    "on_budget",
    "pricing",
    "output_dir",
    "rubric_version",
    "reward_version",
    "catalog_path",
    "exam_split_path",
    "commandcode",
}
_CHANNEL_KEYS = {"model_id", "endpoint", "credential_env"}
_PRICING_TIER_KEYS = {"teacher", "expander"}
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
_PRICING_KEYS = {
    "currency",
    "cny_per_usd",
    "fx_as_of",
    "teacher",
    "expander",
    "off_peak",
    "peak",
    "peak_hours_utc",
}
_RATE_KEYS = {"input_per_mtok", "cached_input_per_mtok", "output_per_mtok"}

# spec §16: k is a per-family max-attempts value in 1..6.
_TEACHER_K_MIN, _TEACHER_K_MAX = 1, 6


# --------------------------------------------------------------------------- #
# typed schema
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class NutriEnvPin:
    """The ``nutrienv:`` dependency pin block in the yaml (ticket 023 lab SHA)."""

    repo: str
    rev: str
    install: str
    note: str | None = None


@dataclasses.dataclass(frozen=True)
class TeacherConfig:
    """Teacher call shape. Production channel is the DeepSeek official API
    (``deepseek-flash``, ``thinking: {"type": "enabled"}``)."""

    model_id: str
    endpoint: str
    credential_env: str
    thinking: dict[str, str]
    temperature_first: float  # attempt 1
    temperature_retry: float  # attempts 2..k
    per_turn_timeout_s: float


@dataclasses.dataclass(frozen=True)
class ChannelConfig:
    """Optional non-production channel (Command Code). Thinking and temperature
    stay on the teacher / expander blocks; this block only swaps the wire."""

    model_id: str
    endpoint: str
    credential_env: str


@dataclasses.dataclass(frozen=True)
class ExpanderConfig:
    """Expander call shape. Production uses the same official endpoint and
    credential as the teacher; ``thinking: {"type": "disabled"}``."""

    model_id: str
    endpoint: str
    credential_env: str
    thinking: dict[str, str]
    timeout_s: float
    parse_retries: int  # attempts = parse_retries + 1 (spec §16)


@dataclasses.dataclass(frozen=True)
class TokenRates:
    """List price per million tokens for one priced role, in ``Pricing.currency``.
    ``input`` is the cache-miss prompt price; ``output`` covers every completion
    token, reasoning included."""

    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float


@dataclasses.dataclass(frozen=True)
class RateCard:
    """One peak or off-peak card. ``teacher`` / ``expander`` are the same list
    price unless a channel prices the two roles differently."""

    teacher: TokenRates
    expander: TokenRates


@dataclasses.dataclass(frozen=True)
class Pricing:
    """The ``pricing:`` block. ``teacher`` / ``expander`` are the single card
    when no peak split is configured, and the off-peak card when it is.

    A run that crosses a peak boundary reprices the accumulated meter at the
    tier in force at the check. The meter does not stamp a rate per call.
    Chinese public holidays are not subtracted from the weekday peak windows."""

    currency: str  # CNY | USD
    cny_per_usd: float | None  # required for CNY; how many CNY one USD buys
    fx_as_of: str | None
    teacher: TokenRates
    expander: TokenRates
    off_peak: RateCard | None = None
    peak: RateCard | None = None
    peak_hours_utc: tuple[str, ...] = ()

    @property
    def usd_per_unit(self) -> float:
        return 1.0 if self.currency == "USD" else 1.0 / self.cny_per_usd

    def is_peak(self, when: dt.datetime) -> bool:
        """Weekday UTC clock inside ``peak_hours_utc`` (half-open)."""
        if self.peak is None or when.weekday() >= 5:
            return False
        minutes = when.hour * 60 + when.minute
        for span in self.peak_hours_utc:
            start, end = span.split("-")
            sh, sm = (int(part) for part in start.split(":"))
            eh, em = (int(part) for part in end.split(":"))
            if sh * 60 + sm <= minutes < eh * 60 + em:
                return True
        return False

    def rates(self, role: str, when: dt.datetime) -> TokenRates:
        card = self.peak if self.is_peak(when) else self.off_peak
        if card is None:
            card_teacher, card_expander = self.teacher, self.expander
        else:
            card_teacher, card_expander = card.teacher, card.expander
        if role == "teacher":
            return card_teacher
        if role == "expander":
            return card_expander
        raise KeyError(role)


@dataclasses.dataclass(frozen=True)
class FamilyConfig:
    """Per-family authoring knobs (spec §7). ``target_n`` is the accepted-Pass
    quota. The build draws one wave, then tops a family up until that count,
    the per-family attempt cap, or the budget (spec §10 dedup still applies)."""

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
    pricing: Pricing
    output_dir: str
    rubric_version: str
    reward_version: str
    catalog_path: str | None
    exam_split_path: str | None
    nutrienv: NutriEnvPin
    commandcode: ChannelConfig | None = None
    # The hand-in turn's plan budget (ADR-011 amendment): the turn whose
    # reasoning carries the verdict. None = plan_max_tokens, as batch 1.
    final_plan_max_tokens: int | None = None
    # Added to every intent seed (and task_id), so a later batch draws new
    # pools instead of re-authoring an earlier batch's tasks.
    seed_offset: int = 0

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


def _parse_rates(value: Any, source: str, path: str) -> TokenRates:
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _RATE_KEYS, source, path)
    rates = {}
    for key in sorted(_RATE_KEYS):
        rate = _number(_required(block, key, source, path), source, f"{path}.{key}")
        if rate < 0:
            _err(source, f"{path}.{key}", f"must be >= 0, got {rate}")
        rates[key] = rate
    return TokenRates(**rates)


def _parse_rate_card(value: Any, source: str, path: str) -> RateCard:
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _PRICING_TIER_KEYS, source, path)
    return RateCard(
        teacher=_parse_rates(_required(block, "teacher", source, path), source, f"{path}.teacher"),
        expander=_parse_rates(
            _required(block, "expander", source, path), source, f"{path}.expander"
        ),
    )


_HOUR_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def _parse_peak_hours(value: Any, source: str, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _err(source, path, "expected a non-empty list of HH:MM-HH:MM windows")
    hours = []
    for index, item in enumerate(value):
        text = _str(item, source, f"{path}[{index}]")
        match = _HOUR_RE.match(text)
        if match is None:
            _err(source, f"{path}[{index}]", f"expected HH:MM-HH:MM, got {text!r}")
        sh, sm, eh, em = (int(part) for part in match.groups())
        if not (0 <= sh <= 23 and 0 <= eh <= 23 and 0 <= sm <= 59 and 0 <= em <= 59):
            _err(source, f"{path}[{index}]", f"clock out of range: {text}")
        if sh * 60 + sm >= eh * 60 + em:
            _err(source, f"{path}[{index}]", f"window must be non-empty and not wrap: {text}")
        hours.append(text)
    return tuple(hours)


def _parse_pricing(value: Any, source: str) -> Pricing:
    path = "pricing"
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _PRICING_KEYS, source, path)
    currency = _str(_required(block, "currency", source, path), source, f"{path}.currency")
    if currency not in _CURRENCIES:
        _err(source, f"{path}.currency", f"expected one of {sorted(_CURRENCIES)}, got {currency!r}")
    cny_per_usd = block.get("cny_per_usd")
    if currency == "CNY":
        cny_per_usd = _number(
            _required(block, "cny_per_usd", source, path), source, f"{path}.cny_per_usd"
        )
        if cny_per_usd <= 0:
            _err(source, f"{path}.cny_per_usd", f"must be > 0, got {cny_per_usd}")
    elif cny_per_usd is not None:
        _err(source, f"{path}.cny_per_usd", "only meaningful when currency is CNY")
    flat = "teacher" in block or "expander" in block
    tiered = "off_peak" in block or "peak" in block or "peak_hours_utc" in block
    if tiered:
        off_peak = _parse_rate_card(
            _required(block, "off_peak", source, path), source, f"{path}.off_peak"
        )
        peak = _parse_rate_card(_required(block, "peak", source, path), source, f"{path}.peak")
        hours = _parse_peak_hours(
            _required(block, "peak_hours_utc", source, path), source, f"{path}.peak_hours_utc"
        )
        teacher, expander = off_peak.teacher, off_peak.expander
        # asdict round-trips the off-peak card under teacher/expander as well.
        if "teacher" in block:
            _parse_rates(block["teacher"], source, f"{path}.teacher")
        if "expander" in block:
            _parse_rates(block["expander"], source, f"{path}.expander")
    else:
        off_peak = peak = None
        hours = ()
        teacher = _parse_rates(
            _required(block, "teacher", source, path), source, f"{path}.teacher"
        )
        expander = _parse_rates(
            _required(block, "expander", source, path), source, f"{path}.expander"
        )
    return Pricing(
        currency=currency,
        cny_per_usd=cny_per_usd,
        fx_as_of=_str_or_none(block.get("fx_as_of"), source, f"{path}.fx_as_of"),
        teacher=teacher,
        expander=expander,
        off_peak=off_peak,
        peak=peak,
        peak_hours_utc=hours,
    )


def _parse_channel(value: Any, source: str) -> ChannelConfig | None:
    if value is None:
        return None
    path = "commandcode"
    block = _mapping(value, source, path)
    _no_unknown_keys(block, _CHANNEL_KEYS, source, path)
    return ChannelConfig(
        model_id=_str(_required(block, "model_id", source, path), source, f"{path}.model_id"),
        endpoint=_str(_required(block, "endpoint", source, path), source, f"{path}.endpoint"),
        credential_env=_str(
            _required(block, "credential_env", source, path), source, f"{path}.credential_env"
        ),
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
    final_plan_max_tokens = root.get("final_plan_max_tokens")
    if final_plan_max_tokens is not None:
        final_plan_max_tokens = _int(final_plan_max_tokens, source, "final_plan_max_tokens")
        if final_plan_max_tokens <= 0:
            _err(source, "final_plan_max_tokens", f"must be > 0, got {final_plan_max_tokens}")
    seed_offset = _int(root.get("seed_offset", 0), source, "seed_offset")
    if seed_offset < 0:
        _err(source, "seed_offset", f"must be >= 0, got {seed_offset}")
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
        pricing=_parse_pricing(_required(root, "pricing", source, ""), source),
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
        commandcode=_parse_channel(root.get("commandcode"), source),
        final_plan_max_tokens=final_plan_max_tokens,
        seed_offset=seed_offset,
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
