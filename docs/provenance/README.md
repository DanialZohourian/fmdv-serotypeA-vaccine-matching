# Provenance

Material documenting where the curated dataset came from. None of it is required to run the
pipeline; it is kept so the extraction can be audited.

| Path | Contents |
|---|---|
| `wrl_vmr_urls.txt` | the 241 WRLFMD Vaccine Matching Report URLs that were screened |
| `image_pdfs.txt` | the 157 reports available only as scanned images, which required OCR |
| `extracted/` | text extracted from reports and papers: `samples/` and `text_heads/` are per-report extractions, `ocr_*.txt` are OCR outputs for scanned reports, and the remaining files are tables and supplementary text from Upadhyaya, Mahapatra, Bari and the G-VII and Azimi papers |

The exclusion decisions made from this material are recorded in `dataset/master_excluded.csv`
and `dataset/master_audit.txt`.
