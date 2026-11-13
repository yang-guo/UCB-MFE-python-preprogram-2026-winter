# Research output provenance

## November 13 structural corrections

Existing CSVs predate the corrected next-session-entry/maturity forward-return
calculation and benchmark-calendar quarantine. They are historical artifacts,
not validated outputs of the corrected pipeline. Incomplete 90-day targets must
be censored; off-calendar observations and cross-gap daily returns are excluded.
`Research.ipynb` outputs have been cleared. Regeneration and interpretation are
pending the research validation pass; no 2016–2026 portfolio evaluation was run
as part of these corrections.
