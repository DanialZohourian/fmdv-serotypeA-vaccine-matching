# dataset/

Curated virus–antiserum records and the search logs behind them.

| File | Contents |
|---|---|
| `master_pairs.csv` | 248 retained virus–antiserum pairs; 190 of these carry sequences for both partners and enter the model |
| `master_excluded.csv` | every screened record that was excluded, with the reason |
| `master_contradictions.csv` | pairs reported more than once with discordant r1 values |
| `master_assay_discordance.csv` | pairs where assays other than 2D-VNT disagreed |
| `unresolved_r1.csv` | records whose r1 value could not be resolved to a number |
| `sequences.csv`, `viruses.csv` | sequence and isolate metadata |
| `genbank_search_log.csv`, `genbank_search_*.json` | GenBank search audit trail |
| `wrl_pairs_raw.json`, `pairs.csv` | raw extraction from WRLFMD reports before curation |
| `scanned_reports_not_digitized.txt` | reports available only as scans, excluded from extraction |
| `master_audit.txt`, `NOTES.txt` | curation decisions and running notes |

`master_pairs_1.csv` and `master_excluded_1.csv` are earlier exports kept for provenance.
