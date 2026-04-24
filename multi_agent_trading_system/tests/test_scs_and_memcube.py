import asyncio

import pytest

from mats.core.memcube import MemCube, MemoryType, Tier, Provenance
from mats.core.scs import InMemorySCS, session_key


@pytest.mark.asyncio
async def test_scs_basic_roundtrip():
    scs = InMemorySCS()
    await scs.set("k", {"v": 1})
    assert await scs.get("k") == {"v": 1}
    await scs.append("list", 1)
    await scs.append("list", 2)
    assert await scs.get("list") == [1, 2]
    keys = await scs.keys("*")
    assert set(keys) == {"k", "list"}


def test_memcube_decays_and_promotes():
    cube = MemCube(title="t", activation_strength=1.0, decay_rate=0.1)
    cube.decay(days=1)
    assert cube.activation_strength < 1.0
    cube.promote(Tier.MTM)
    assert cube.tier == Tier.MTM


def test_memcube_frontmatter_roundtrip():
    cube = MemCube(
        title="t", body="body",
        provenance=Provenance.for_payload(source="agent", payload="body"),
    )
    fm = cube.to_frontmatter()
    assert fm["memcube_id"] == cube.memcube_id
    assert fm["provenance"]["hash"]


def test_session_key_hash_tagged():
    assert session_key("abc", "scratch") == "scs:{abc}:scratch"
