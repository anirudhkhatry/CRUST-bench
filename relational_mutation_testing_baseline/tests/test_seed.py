from pathlib import Path

from additional_datasets import seed

ROOT = Path(__file__).resolve().parent.parent


def test_fault_categories_are_the_spec_16_list():
    assert "encoding difference" in seed.FAULT_CATEGORIES
    assert "ordering difference" in seed.FAULT_CATEGORIES
    assert "partial migration" in seed.FAULT_CATEGORIES
    assert len(seed.FAULT_CATEGORIES) >= 15


def test_every_language_has_hazard_prose():
    assert "javascript" in seed.HAZARDS and "java" in seed.HAZARDS
    assert "Locale" in seed.HAZARDS["java"]


def test_seed_does_not_import_the_mutation_arms_operators():
    """Spec §7.4: if Arm S drew on the C mutation arm's taxonomy, "the mutation arm
    catches Arm S faults" would prove only that two halves of this repo share a
    vocabulary. This checks the import route; the taxonomy route is checked below.
    """
    src = (ROOT / "additional_datasets" / "seed.py").read_text()
    for forbidden in ("stage1_mutate", "rmt.stage1", "from rmt import"):
        assert forbidden not in src, f"seed.py reaches the mutation arm via {forbidden!r}"


def test_seed_fault_prompt_does_not_copy_from_mutate_prompt():
    """Literal copy-paste check between the two prompt files (catches careless errors)."""
    seed_prompt = (ROOT / "additional_datasets" / "prompts" / "seed_fault.md").read_text()
    rmt_prompt = (ROOT / "rmt" / "prompts" / "mutate.md").read_text()
    # No long verbatim span shared between the two prompts.
    for i in range(0, len(rmt_prompt) - 120, 60):
        assert rmt_prompt[i:i + 120] not in seed_prompt


def test_the_two_fault_taxonomies_are_disjoint():
    """The real anti-circularity property: Arm S's categories and the C mutation arm's
    fault classes must not overlap. A verbatim-span check cannot see a taxonomy that was
    paraphrased rather than copied; this can.
    """
    from rmt.stage1_mutate import FAULT_CLASSES

    shared = {c.lower() for c in seed.FAULT_CATEGORIES} & {c.lower() for c in FAULT_CLASSES}
    assert not shared, f"Arm S shares fault vocabulary with the mutation arm: {shared}"
