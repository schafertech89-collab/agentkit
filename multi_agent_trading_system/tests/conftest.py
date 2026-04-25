"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

# Never let tests execute real orders.
os.environ.setdefault("MATS_LIVE", "0")
os.environ.setdefault("COINBASE_SANDBOX", "1")
os.environ.setdefault("GOVERNANCE_HASH_SEED", "test-seed")


@pytest.fixture
def tmp_vault(tmp_path):
    vault = tmp_path / "vault"
    (vault / ".governance").mkdir(parents=True)
    (vault / ".raw").mkdir(parents=True)
    (vault / "STM").mkdir(parents=True)
    (vault / "MTM" / "episodic").mkdir(parents=True)
    (vault / "LTM" / "semantic").mkdir(parents=True)
    return vault
