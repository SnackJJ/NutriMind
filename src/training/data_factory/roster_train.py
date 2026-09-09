"""TRAIN_ROSTER — the NutriMind-side people the data factory authors for.

The single isolation point between train and exam (spec §2, US-16): a
``train-*`` ``user_id`` prefix (the exam roster is ``roster-*``) plus body
facts chosen so ``derive_profile_windows`` output is **provably disjoint**
from every nutri-env ``ROSTER`` person's windows. A template-family oracle
authored on these people therefore cannot collide with the frozen v1.0 exam.
Prior art: ``docs/plans/nutrienv_student_data.md`` §3 (its "21 人" count is
outdated — the exam roster is 23 — and its numbers are superseded).

Window math the design exploits (``nutrienv.world.daily_windows``):
``derive_profile_windows`` computes a Mifflin-St Jeor × PAL energy
requirement (EER); the kcal window is ``(EER, EER)`` for maintain/muscle and
``(EER-300, EER-300)`` for cut; carb/fat/fiber windows are the FDA Daily
Values scaled by ``EER / 2000`` (injective in EER); the protein window is
``(0.8 g/kg x weight, max(EER/40, lo))`` for maintain/cut and
``(1.6 g/kg x weight, max(EER/40, lo))`` for muscle. Disjointness strategy:

- **everyday weights sit at half-kilos** (55.5, 73.5, ...) off the exam
  roster's integer-kilo grid, so no everyday protein window can coincide with
  an exam one: a same-regime coincidence needs an identical weight, and a
  cross-regime one (0.8 vs 1.6 g/kg) needs an exam weight at exactly 2x or
  1/2x a TRAIN weight — impossible across the grid shift.
- **gym and cut weights are integers off the exam weight set** (gym
  69/75/84/96 vs exam gym {58, 66, 70, 80, 85}; cut 71/78/94), each placed so
  neither 2x nor 1/2x of it is an exam weight either — no same- or
  cross-regime protein tuple can coincide.
- **heights/ages/activities spread every EER off all 23 exam EERs** (and cut
  people's EER-300 off every exam kcal value), which separates kcal, carb,
  fat, and fiber simultaneously — they are all functions of EER alone.

``tests/training/data_factory/test_roster_train.py`` proves all of this
exhaustively rather than trusting the arithmetic above.

``sodium_mg`` is **excluded** from the disjointness contract on purpose:
``derive_profile_windows`` returns the constant ``(0.0, 2300.0)`` for every
person in both rosters, so it cannot separate anyone (documented in the
isolation test as well).

Persona mix: 13 everyday / 4 gym / 3 cut — the provisional 65/20/15
everyday/gym/cut mix (spec §23 OQ-9; the NHANES-or-diet-app citation is
deferred and non-blocking — it affects this roster, not the pipeline), exact
at n=20. Gym people carry ``phase="muscle"`` (the 1.6 g/kg protein floor
naturally produces the higher protein windows a post-gym profile needs) with
active/very_active PALs, so the ``rec-post-gym`` shell — which hard-requires
the gym persona — always has plausible speakers.

Allergies: the catalog's ``allergen_tags`` vocabulary is **exactly** the exam
roster's nine tags (egg, fish, gluten, milk, peanut, shellfish, soy,
tree_nut, wheat) — there is no supported allergy value outside the exam set.
The pre-spec draft's "sesame" suggestion is unsupported: the catalog's
"Sesame *" foods carry no ``allergen_tags``, so
``generate_one(family="update", shell="upd-add-allergy-short",
slots={"allergen": "sesame"})`` rejects with ``no_allergen_food``. TRAIN
allergies therefore draw from the supported vocabulary; train/exam isolation
is carried by the ``train-*`` prefix and the window disjointness, not by the
allergy vocabulary. Several **allergy-free** people exist in each persona so
the ``upd-add-allergy-short`` composite always has a clean-S0 speaker.

This is a stage module: it imports ``nutrienv`` at module level (allowed by
spec §18), while the package ``__init__`` stays nutrienv-free — do not add
this module to ``src/training/data_factory/__init__.py`` imports. Windows are
derived via ``profile_for`` / ``derive_profile_windows``, never stored here.
"""

from __future__ import annotations

from nutrienv.bench.pipeline.roster import RosterPerson

__all__ = ["TRAIN_ROSTER"]

# Shape mirrors nutri-env's ROSTER (a tuple of RosterPerson), one entry per
# person, alphabetical user_ids.
TRAIN_ROSTER: tuple[RosterPerson, ...] = (
    # ---- everyday (13) — phase "maintain", every PAL level represented,
    # ages 20-71. Half-kilo weights off the exam integer grid (module
    # docstring): each protein window lo is 0.8 g/kg x (k + 0.5), which no
    # exam 0.8/1.6 x integer weight can equal; heights/ages/activities land
    # EER off all 23 exam EERs, separating kcal/carb/fat/fiber in one move.
    # Allergy-free people are spread through the group so the
    # upd-add-allergy-short composite leg always has a clean-S0 speaker.
    RosterPerson("train-alba", "female", 23, 161.0, 55.5, "light", "maintain", (), "everyday", "balanced"),
    RosterPerson("train-bruno", "male", 26, 176.0, 73.5, "moderate", "maintain", (), "everyday", "balanced"),
    RosterPerson("train-cleo", "female", 35, 157.0, 53.5, "sedentary", "maintain", ("milk",), "everyday", "dairy_free"),
    RosterPerson("train-dario", "male", 41, 183.0, 92.5, "active", "maintain", (), "everyday", "balanced"),
    RosterPerson("train-elena", "female", 48, 168.0, 65.5, "moderate", "maintain", ("peanut", "tree_nut"), "everyday", "mediterranean"),
    RosterPerson("train-farid", "male", 57, 169.0, 78.5, "light", "maintain", (), "everyday", "heart_healthy"),
    RosterPerson("train-greta", "female", 66, 154.0, 56.5, "sedentary", "maintain", ("egg",), "everyday", "vegetarian"),
    RosterPerson("train-hugo", "male", 71, 172.0, 75.5, "light", "maintain", ("gluten",), "everyday", "gluten_free"),
    RosterPerson("train-iris", "female", 29, 174.0, 67.5, "very_active", "maintain", (), "everyday", "balanced"),
    RosterPerson("train-jonas", "male", 20, 186.0, 82.5, "very_active", "maintain", (), "everyday", "balanced"),
    RosterPerson("train-kanya", "female", 52, 162.0, 71.5, "light", "maintain", ("shellfish",), "everyday", "mediterranean"),
    RosterPerson("train-lucas", "male", 33, 179.0, 87.5, "moderate", "maintain", (), "everyday", "high_protein"),
    RosterPerson("train-mira", "female", 60, 159.0, 58.5, "moderate", "maintain", ("soy", "wheat"), "everyday", "low_sodium"),
    # ---- gym (4) — phase "muscle" (the 1.6 g/kg protein floor naturally
    # yields the higher protein windows a post-gym profile needs) with
    # active/very_active PALs, so the rec-post-gym shell — which hard-requires
    # the gym persona — always has plausible speakers. Integer weights sit
    # off the exam gym weights {58, 66, 70, 80, 85} and away from 2x/1/2x
    # coincidences with exam maintain/cut weights (all <= 90 kg).
    RosterPerson("train-noor", "male", 25, 181.0, 84.0, "active", "muscle", (), "gym", "high_protein"),
    RosterPerson("train-omar", "female", 31, 170.0, 69.0, "very_active", "muscle", (), "gym", "high_protein"),
    RosterPerson("train-pia", "male", 38, 187.0, 96.0, "active", "muscle", ("milk", "egg"), "gym", "high_protein"),
    RosterPerson("train-quinn", "male", 22, 178.0, 75.0, "very_active", "muscle", ("peanut",), "gym", "high_protein"),
    # ---- cut (3) — phase "cut" (kcal window at EER-300), light/moderate/
    # active PALs. Integer weights off the exam weight set with 2x/1/2x also
    # off it; each EER-300 kcal value is verified off every exam kcal tuple
    # (maintain EERs and cut EER-300s alike) by the isolation test. Two of
    # three start allergy-free so this persona too has clean-S0 speakers for
    # upd-add-allergy-short.
    RosterPerson("train-rhea", "female", 44, 158.0, 71.0, "light", "cut", (), "cut", "low_sodium"),
    RosterPerson("train-stefan", "male", 49, 175.0, 94.0, "moderate", "cut", ("fish",), "cut", "mediterranean"),
    RosterPerson("train-talia", "female", 27, 165.0, 78.0, "active", "cut", (), "cut", "low_carb"),
)
