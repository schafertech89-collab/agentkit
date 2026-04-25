# MATS vault constitution

This file is the constitution every write-capable MATS agent must obey. It is
the single source of truth for the memory lifecycle; violations are rejected
at the SSGM write boundary and the offending write never reaches disk.

## Invariants

1. **`.raw/` is immutable.** Once a raw source (market snapshot, ticker
   capture, orderbook dump) is written it must never be modified. If the
   underlying source updates, a new file with a fresh content hash is created.

2. **Episodic memories (`MTM/episodic/`) are append-only.** They can be linked
   to from semantic memories but must never be rewritten. They are only moved
   — either promoted to `LTM/` via the ConsolidateAgent, or soft-demoted to
   `.decay/` by the memory policy.

3. **Importance decrements only through ConsolidateAgent.** Agent code must
   never manually lower `importance` on an existing memcube. ConsolidateAgent
   is the sole authorised writer.

4. **Every write carries an audit trail.** The `governance.audit_chain` list
   must be a SHA-256 hash chain linking the current version to its parent.
   On read, corrupt chains trigger a governance halt.

5. **No agent writes into a tier it doesn't own.**
   - `episodic-mcp` owns `MTM/episodic/`
   - `semantic-mcp` owns `LTM/semantic/`
   - `procedural-mcp` owns `LTM/procedural/`
   - `profile-mcp` owns `LTM/profile/`
   Writes outside the owning tier are rejected by filesystem ACL.

## Frontmatter schema

Every markdown file carries a YAML frontmatter block produced by
`MemCube.to_frontmatter()`. The fields are:

- `memcube_id` — stable UUID-derived identifier
- `memory_type` — parametric | activation | plaintext
- `tier` — STM | MTM | LTM | DECAY | RAW
- `importance`, `activation_strength`, `decay_rate` ∈ [0, 1]
- `replay_count` — integer, monotonically non-decreasing
- `consolidated_to` — optional memcube_id pointer
- `provenance` — {source, hash, immutable, source_path}
- `governance` — {version, last_modified_by, audit_chain}
- `policy_signal` — {utility_score, interference, regret}

## Trading-domain conventions

- Every *decision* MemCube links to its source *analysis* cubes via
  `[[wikilinks]]` in its body and via `DERIVED_FROM` edges in the graph.
- Every *execution receipt* links to the *decision* that triggered it.
- Every *outcome* (realized P&L) links back to the *decision*, closing the
  reflection loop.
- Allora submissions are persisted as MemCubes with `memory_type: plaintext`
  in `MTM/episodic/allora/`.

## Live-trading gate

No agent may issue a live trade until the audit chain verifies clean on boot
*and* `MATS_LIVE=1`. In paper mode the connectors return simulated receipts
but the full governance chain still records them. Do not disable this gate.
