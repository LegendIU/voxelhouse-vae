# Failure Gallery

This gallery tracks representative generation failures for transparent error analysis.

Use one section per failure case. Keep examples from:

- unconditional mode
- conditional mode
- constraint-guided mode (if still failing)

---

## Case FG-001: short title

- **Run ID**: `run_xxx`
- **Regime**: `unconditional | conditional | constraint_guided`
- **Condition case**: `none | two_story_compact | wide_lowrise_sloped | custom`
- **Sample artifact**: `path/to/projection_or_mesh`
- **Observed failure type**: `disconnected structure | floating mass | wrong roof type | etc`

### Metrics snapshot

- `connectedness`:
- `unsupported_mass`:
- `component_count`:
- `symmetry_proxy`:
- `plausibility_score`:
- `attribute_match_rate`:

### Why this failed (hypothesis)

- short hypothesis 1
- short hypothesis 2

### Potential fix

- data fix:
- model fix:
- sampling/guidance fix:

### Severity

- `high | medium | low`

---

## Failure taxonomy (recommended labels)

- **FG-DISCONNECT**: multi-component disconnected mass
- **FG-UNSUPPORTED**: large floating unsupported volume
- **FG-COLLAPSE**: oversimplified blob-like collapse
- **FG-MISMATCH-ATTR**: misses requested condition attributes
- **FG-OVERGUIDED**: valid but repetitive low-diversity outputs

Keeping this file updated closes the loop between metrics, visual evidence, and next actions.
