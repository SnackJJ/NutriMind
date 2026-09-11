"""Ticket 003 — config schema / loader for configs/data_factory.yaml.

Covers the acceptance list: every spec §7 key present in the shipped yaml; the
loader returns a typed object and raises a clear error on a missing / mistyped /
unknown key; the ticket-023 NutriEnv lab pin is the first 40-hex in the yaml; the pinned
design numbers (spec §2.1) are asserted so they cannot drift silently.
"""

from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from src.training.data_factory import ConfigError, DataFactoryConfig, load_config

REPO = pathlib.Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO / "configs" / "data_factory.yaml"

NUTRIENV_PIN = "0ee68eaa6c246e8079915761c95fc986c53d4979"
CATALOG_SHA = "57184b2bbce4519076b4238a8d64861950db46fdc793d0e43055f07f43c28b5f"


@pytest.fixture(scope="module")
def shipped() -> DataFactoryConfig:
    return load_config(CONFIG_PATH)


@pytest.fixture()
def shipped_dict() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


# --------------------------------------------------------------------------- #
# the shipped config: complete + pinned values
# --------------------------------------------------------------------------- #


def test_shipped_config_loads_typed(shipped):
    assert isinstance(shipped, DataFactoryConfig)


def test_every_spec7_key_present(shipped):
    # spec §7 keys (nutrienv_rev is single-sourced from the pin block)
    for field in (
        "target", "teacher", "expander", "families", "max_seq_tokens",
        "plan_max_tokens", "tokenizer_name", "max_intents", "usd_budget",
        "on_budget", "output_dir", "rubric_version", "reward_version",
        "catalog_path", "exam_split_path",
    ):
        assert hasattr(shipped, field), field
    assert shipped.nutrienv_rev == NUTRIENV_PIN


def test_teacher_block_pinned(shipped):
    # user-pinned teacher config (ADR-011 amended 2026-09-09)
    assert shipped.teacher.model_id == "ark/deepseek-v4-flash"
    assert shipped.teacher.endpoint == "api/plan/v3/chat/completions"
    assert shipped.teacher.credential_env == "ARK_API_KEY"
    assert shipped.teacher.thinking == {"type": "enabled"}
    assert shipped.teacher.temperature_first == 0.0
    assert shipped.teacher.temperature_retry == 0.7
    assert shipped.teacher.per_turn_timeout_s == 60


def test_expander_shares_endpoint_and_credential(shipped):
    # one provider, one credential for teacher + expander
    assert shipped.expander.model_id == "ark/deepseek-v4-flash"
    assert shipped.expander.endpoint == shipped.teacher.endpoint
    assert shipped.expander.credential_env == shipped.teacher.credential_env
    assert shipped.expander.thinking == {"type": "disabled"}
    assert shipped.expander.timeout_s == 60
    assert shipped.expander.parse_retries == 1


def test_scalar_keys(shipped):
    assert shipped.max_seq_tokens == 20000      # ADR-011
    assert shipped.plan_max_tokens == 80        # ADR-011 ~80-token plan cap
    assert shipped.tokenizer_name is None
    assert shipped.on_budget == "warn"
    assert shipped.output_dir == "data/student/"  # v2-only output dir (spec §14)
    assert shipped.rubric_version == "v2-r1"
    assert shipped.reward_version == "v2-r1"


def test_families_have_full_schema(shipped):
    assert shipped.families, "at least one family must be configured"
    for name, family in shipped.families.items():
        assert isinstance(family.target_n, int) and family.target_n > 0, name
        assert 1 <= family.teacher_k <= 6, name  # spec §16
        assert family.over_generate_x > 0, name
        assert isinstance(family.gram_anchor, bool), name


def test_pinned_design_numbers(shipped):
    """Numbers preserved in spec §2.1 — do not edit these casually (they are the
    design contract, not tunables)."""
    three_leg = shipped.families["composite_update_log_recommend"]
    assert three_leg.target_n == 40            # 3-leg target = 40 accepted Pass
    assert three_leg.teacher_k == 6            # teacher k = 6
    assert three_leg.gram_anchor is True       # §6 ladder on from the start (spec §22.13)

    total = sum(f.target_n for f in shipped.families.values())
    assert 400 <= total <= 440, f"Batch-1 total {total} drifted from ~420"

    # family mix (spec §2.1): composite ~57 / recommend ~17 / evaluate ~13 /
    # log ~10 / update ~3 (percent of the ~420 total)
    composite = (
        shipped.families["composite"].target_n + three_leg.target_n
    )
    shares = {
        "composite": composite,
        "recommend": shipped.families["recommend"].target_n,
        "evaluate": shipped.families["evaluate"].target_n,
        "log": shipped.families["log"].target_n,
        "update": shipped.families["update"].target_n,
    }
    for name, approx in (
        ("composite", 0.57), ("recommend", 0.17), ("evaluate", 0.13),
        ("log", 0.10), ("update", 0.03),
    ):
        assert abs(shares[name] / total - approx) <= 0.03, (name, shares[name] / total)


# --------------------------------------------------------------------------- #
# loader errors: missing / mistyped / unknown keys
# --------------------------------------------------------------------------- #


def _load(mutate, dict_fixture):
    data = copy.deepcopy(dict_fixture)
    mutate(data)
    from src.training.data_factory import config_from_dict

    return config_from_dict(data, source="test-config")


def test_error_on_missing_teacher_block(shipped_dict):
    del shipped_dict["teacher"]
    with pytest.raises(ConfigError, match="teacher"):
        _load(lambda d: d, shipped_dict)


def test_error_on_missing_nested_key(shipped_dict):
    del shipped_dict["teacher"]["per_turn_timeout_s"]
    with pytest.raises(ConfigError, match=r"teacher.*per_turn_timeout_s.*missing"):
        _load(lambda d: d, shipped_dict)


def test_error_on_mistyped_key(shipped_dict):
    shipped_dict["teacher"]["per_turn_timeout_s"] = "sixty"
    with pytest.raises(ConfigError, match=r"per_turn_timeout_s.*expected a number.*str"):
        _load(lambda d: d, shipped_dict)


def test_error_on_mistyped_gram_anchor(shipped_dict):
    shipped_dict["families"]["log"]["gram_anchor"] = "yes"
    with pytest.raises(ConfigError, match=r"gram_anchor.*expected a bool"):
        _load(lambda d: d, shipped_dict)


def test_error_on_teacher_k_out_of_range(shipped_dict):
    shipped_dict["families"]["log"]["teacher_k"] = 7
    with pytest.raises(ConfigError, match=r"teacher_k.*1\.\.6"):
        _load(lambda d: d, shipped_dict)
    shipped_dict["families"]["log"]["teacher_k"] = 0
    with pytest.raises(ConfigError, match=r"teacher_k.*1\.\.6"):
        _load(lambda d: d, shipped_dict)


def test_error_on_bad_target(shipped_dict):
    shipped_dict["target"] = "train"
    with pytest.raises(ConfigError, match="target"):
        _load(lambda d: d, shipped_dict)


def test_error_on_bad_thinking_type(shipped_dict):
    shipped_dict["teacher"]["thinking"] = {"type": "sometimes"}
    with pytest.raises(ConfigError, match=r"thinking.*type"):
        _load(lambda d: d, shipped_dict)


def test_error_on_unknown_top_level_key(shipped_dict):
    shipped_dict["techer"] = {}  # typo'd duplicate
    with pytest.raises(ConfigError, match="unknown key"):
        _load(lambda d: d, shipped_dict)


def test_error_on_unknown_family_key(shipped_dict):
    shipped_dict["families"]["log"]["over_generate_factor"] = 3.0
    with pytest.raises(ConfigError, match=r"families\.log.*unknown key"):
        _load(lambda d: d, shipped_dict)


def test_error_on_missing_nutrienv_pin(shipped_dict):
    del shipped_dict["nutrienv"]["rev"]
    with pytest.raises(ConfigError, match=r"nutrienv.*rev.*missing"):
        _load(lambda d: d, shipped_dict)


def test_error_on_bad_nutrienv_rev(shipped_dict):
    shipped_dict["nutrienv"]["rev"] = "main"
    with pytest.raises(ConfigError, match=r"rev.*40-hex"):
        _load(lambda d: d, shipped_dict)


def test_error_on_bad_on_budget(shipped_dict):
    shipped_dict["on_budget"] = "ignore"
    with pytest.raises(ConfigError, match="on_budget"):
        _load(lambda d: d, shipped_dict)


def test_error_on_empty_families(shipped_dict):
    shipped_dict["families"] = {}
    with pytest.raises(ConfigError, match="families.*at least one"):
        _load(lambda d: d, shipped_dict)


def test_load_config_missing_file():
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(REPO / "configs" / "does_not_exist.yaml")


def test_amount_path_weights_accepted(shipped_dict):
    shipped_dict["families"]["log"]["amount_path_weights"] = {
        "named_measure": 0.5, "unspecified": 0.3, "explicit_grams": 0.2,
    }
    cfg = _load(lambda d: d, shipped_dict)
    assert cfg.families["log"].amount_path_weights == {
        "named_measure": 0.5, "unspecified": 0.3, "explicit_grams": 0.2,
    }


# --------------------------------------------------------------------------- #
# the ticket-023 lab pin
# --------------------------------------------------------------------------- #


def _pin_block(text: str) -> str:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("nutrienv:"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line and not line[0].isspace() and not line.startswith("#"):
            end = i
            break
    return "\n".join(lines[start:end]).rstrip()


def test_nutrienv_pin_is_lab_rev():
    """v2 pin is ``../nutri-env-lab`` at the ticket-023 SHA; CI reads the first 40-hex."""
    text = CONFIG_PATH.read_text()
    block = _pin_block(text)
    assert NUTRIENV_PIN in block
    assert "../nutri-env-lab" in block
    import re

    first_hex = re.search(r"[0-9a-f]{40}", text).group(0)
    assert first_hex == NUTRIENV_PIN
