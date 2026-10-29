# v5 integrity logs

These logs were produced on 29 Oct 2026 by running the v5 pipeline in-repo
(`code/v5/`, repository root as working directory, Python 3.12.16 /
numpy 2.1.3 / scipy 1.14.1 / scikit-learn 1.5.2, pdftotext 24.x) against the
committed data in `results/{v5,s3,a1,a2}/` and the v5.0.1 manuscript
(release `paper-v5.0.1`; `V5_PAPER_DIR` pointed at the unpacked source
archive).

| Log | What it shows |
|---|---|
| `check_s3_integrity.log` | I1–I6 on the four S3 cells: seed sets, shared-member bit-identity, cross-k Single bit-identity, config stamps, version stamps, member metrics — **ALL CHECKS PASS** |
| `check_a1_integrity.log` | I1–I7 on the three A1 cells, incl. 360 bit-level anchors vs the S1 single records — **ALL PASS** |
| `analyze_v5.log` / `analyze_s3.log` / `analyze_a1a2.log` | full re-computation of every analysis artifact; all ten outputs **byte-identical** to the committed ones |
| `postcheck_v5_paper.log` | 51-check manuscript-vs-data verification (structure, 28 table cells, headline statistics, study-scale claims N21a–g) — **ALL PASS** |
| `audit_v501_scale_claims.log` | 33-check deep audit of countable/scale claims — **ALL PASS** |

To reproduce any of them, see README Part I.4 ("Verification and
reproducing").
