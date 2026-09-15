from pathlib import Path

from additional_datasets import seed


def test_fault_categories_are_the_spec_16_list():
    assert "encoding difference" in seed.FAULT_CATEGORIES
    assert "ordering difference" in seed.FAULT_CATEGORIES
    assert "partial migration" in seed.FAULT_CATEGORIES
    assert len(seed.FAULT_CATEGORIES) >= 15


def test_every_language_has_hazard_prose():
    assert "javascript" in seed.HAZARDS and "java" in seed.HAZARDS
    assert "Locale" in seed.HAZARDS["java"]


def test_seed_does_not_import_the_mutation_arms_operators():
    """Spec §7.4: sharing a taxonomy with stage1_mutate would make Arm S circular."""
    src = Path("additional_datasets/seed.py").read_text()
    assert "stage1_mutate" not in src
    assert "from rmt import" not in src or "stage1" not in src
    prompt = Path("additional_datasets/prompts/seed_fault.md").read_text()
    rmt_prompt = Path("rmt/prompts/mutate.md").read_text()
    # No long verbatim span shared between the two prompts.
    for i in range(0, len(rmt_prompt) - 120, 60):
        assert rmt_prompt[i:i + 120] not in prompt
