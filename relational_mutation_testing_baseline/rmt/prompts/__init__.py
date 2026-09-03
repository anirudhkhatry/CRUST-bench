"""Prompt texts, kept out of the pipeline code so they can be diffed run to run.

Substitution is plain `<<NAME>>` replacement, not str.format -- the prompts embed C and
Rust source, which is full of braces.
"""
from pathlib import Path

_DIR = Path(__file__).parent


def load(name, **subs):
    text = (_DIR / f"{name}.md").read_text()
    for k, v in subs.items():
        text = text.replace(f"<<{k.upper()}>>", str(v))
    return text
