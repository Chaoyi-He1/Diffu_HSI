# Task 19a — Phase 3 results page (report)

Commit 6b3b856 on worktree-phase1-plan (not pushed, nothing published). It adds 7 files: docs/analysis/phase3_page/{extract_page_data.py, body.html, charts.js, build_page.py, page_data.json} and docs/phase3_sensor_results{.src,}.html.

## Pipeline
- `extract_page_data.py [--verdict FILE]` reads every source and writes the compact `page_data.json`. It runs from any cwd through a sys.path shim. Its output is byte-identical on unchanged data: `generated` is the newest record or verdict `created` stamp, not the wall clock. It prints the B count and the PENDING list.
- `build_page.py [--extract]` is the same mechanism as Phase 2a. It copies the Phase 1a′ head (retitled "Phase 3 Results") and helpers by text markers, then adds body.html and charts.js, and injects the data into `/*__DATA__*/null`. It writes the .src.html and .html files, plus a standalone copy to `$PAGE_RENDER` when that is set. It prints the pending list.
- Default verdict: the first of verdict__e123.json, verdict__e123__report.json and verdict__e123_noB__report.json that exists; otherwise the newest verdict__*.json.
- Rebuild once the B records land: `python docs/analysis/phase3_page/build_page.py --extract` (add `--verdict ...` if needed).

## Data bindings
- **Settings:** E1 device1/seq at 60/50/40/30/20; E2 device2 and ph5/seq at 40/30/20; E3 device1/snapshot at 60/40/30/20. For each setting the extractor reads the means of L, L_demosaic, D, V, C and C1 from the `_main` and `_C` records.
- **B records:** B is per setting with status final, pending or dropped (PH5 is dropped by design). A record is marked stale if its prediction type is not x0, read from provenance.b_prediction_type or the checkpoint's prediction_type. The x0 B blocks carry:
  - paired comparisons on the common files with `scripts/paired_stats.cluster_se`: B against L, D, V, L_demosaic, C and C1, for RMSE and CRPS;
  - a T1-style or T4-style bound, using sd_verdict's ratio_2se formula with limit 1.05 against D in E1 and 0.85 against L_demosaic at snapshot 30 dB;
  - the other methods' means on B's 50 cubes.
- **Archived v-prediction B records** (`eval/vpred_prior/`, 60 and 50 dB) appear as "B, v-prediction prior (archived)".
- **Stage 0:** summary.json (L curves, raw curve, shape_check, principal angles with the non-zero ones extracted) plus the L__*.json means table (L and L_demosaic).
- **Sensor:** `load_sensor(dev, DATA_ROOT)` gives the 31×30 normalised R for device1 and the singular values, number below 1 % and condition number for all three devices.
- **Toy record:** not in SENSOR_DIR. It exists only at the worktree's `results/residual_warmstart/sensor/toy_guidance.json`, so the extractor falls back to that path.
- **Config:** P_y, training_sha, family, and e2_demosaic min/median/max.
- **Runs:** train_log and args for the 5 runs and 3 pilots, giving epochs, mean s/epoch, hours and val_loss at the selected epoch.
- **Budget:** eval GPU-h by record kind (provenance.seconds), the mean B record cost, and the projection for 12 records.
- **B-prior probe curves:** a literal constant `B_PRIOR_PROBE` that cites the results-note section.
- **Prose numbers:** every number in body.html is a `data-v="path:fmt"` span, resolved from page_data at render time, or a JS slot that builds the sentence from the data. `facts` holds the derived numbers. Sentences that depend on B are slots, conditional on which B records exist, and say "pending" otherwise.

## Page structure
Header (claim, the three experiments, the pre-registration statement, a data-built status box, KV tiles, pipeline figure, glossary) → 01 sensor (R columns, singular values, Stage 0 MMSE vs SNR with the raw curve, normalisation check, principal angles) → 02 figure (i) (RMSE vs SNR log-y with L/D/C/C1/B/B-vpred, the floor reference and T1 markers; CRPS vs SNR) → 03 figure (ii) (CRPS bars at 30 and 20 dB, coverage dots with the T2 band, AUSE bars, T2 markers and text) → 04 E2 (grouped bars per sensor with a 40/30/20 toggle, T3 text, Device 2 / PH5 readings, B on Device 2) → 05 E3 (RMSE vs SNR with the T4 target, uncertainty table, T4 text, the poor-centre remark, B in snapshot) → 06 B story (x̂₀-error vs t log-y with the mean-cube line and saturation numbers, toy table, B record table for all 12 settings plus the 2 archived, "where B stands") → 07 clause table (T1 per SNR, T2 at 30 and 20, T3, T4, sanity, counts; C and B columns) → 08 runs table and budget → 09 deviations and rulings (6 items) → footer with sources.

## Pending (at build time; records as of 2026-11-05 23:51)
- **B records:** 2 of 12 x0 records are present, E1 device1 60 dB and E3 snapshot 40 dB. Pending: E1 50/40/30/20, E2 device2 40/30/20, E3 snapshot 60/30/20.
- **Verdict:** the current verdict is `verdict__e123_noB__report.json`. Its B clauses read the archived v-prediction records, so the page marks them invalid ("e1_dev1_seq_50_B is not in eval/ ..."). In the B column the page shows "pending" or an amber "preview" cell: the page's own paired computation under the verdict's rule. These become verdict cells automatically when a verdict whose manifest names only x0 B records written before it is selected.
- **Early B readings (data-driven, so they will update):** x0 B at 60 dB has a mean RMSE of 2.39 % against D's 1.05 % on the same cubes, cov90 0.49, which is under-dispersed. At snapshot 40 dB it is 5.96 % against L_demosaic's 5.60 % (ratio 1.06 ± 0.42 cluster SE).

## Checks run
1. Extractor ×2 from /tmp and /: page_data.json md5 identical. A full rebuild (`build_page.py --extract` from /) leaves all three outputs md5-identical.
2. Headless Chrome (google-chrome --headless=new --dump-dom, virtual-time-budget 15 s) on the standalone render copy, with a Python check of the dumped DOM:
   - every data-fig (1), data-chart (11), data-table (5), data-slot (23) and data-legend (4) id in body.html has a renderer in charts.js;
   - all 108 slot and data-v elements are non-empty after JS;
   - no NaN, undefined or Infinity in the rendered (non-script) DOM;
   - 25 SVGs, 23 "pending" pills, 5 "preview" pills.
3. Full-page screenshots (1000 px wide) inspected section by section. Layout fixes applied: label positions, the narrow-column AUSE chart, the two-line snapshot table cells, the toy-table order, and min-SNR γ hidden for direct modes.
4. A Python HTMLParser parse of both outputs; `<title>Phase 3 Results</title>` is present.
5. No Phase 2a page text was copied. Two deliberate scientific references to Phase 2a remain, both taken from the results-note readings: C's CRPS win is "the opposite of Phase 2a", and "a diffusion centred on D's prediction, Phase 2a's construction". Remove them if a strict "no Phase 2a" rule is meant.

## Concerns
- The toy JSON lives only in this worktree's git-ignored results/ copy, not under RESULTS_ROOT/sensor. The extractor falls back to it.
- B preview cells reproduce sd_verdict's formula and are not the verdict. They are labelled as previews on the page.
- The B-prior probe curves are literals (as instructed). The prose "worse than the mean cube at t = …" is computed from those literals (150, 200, 299, 400). The note's wording says "t = 100–400", but at t = 100 the v-prediction prior's error is 14.2 %, below the 17.15 % mean-cube error.
- The runs table index binding in the glossary (`runs.2/3.selected`) depends on RUN_TABLE order.

---
# Fix pass after the two adversarial checks (commit 6d2db08)

Built with `build_page.py --extract --verdict .../verdict__e123.json`. The B clauses are now verdict cells (b_valid = true).

The only record still pending is e4_device2_snapshot_30_B. It shows "pending" in the E4 cells and the B table, plus a one-line amber notice that hides itself once nothing is pending.

## Checks
- Headless Chrome dump-dom: every one of the 31 slots, 11 charts, 5 tables, 1 figure and 4 legends has a renderer, and all 116 slot elements are filled after render. No NaN, undefined or Infinity. No "preview", "so far", "records present" or "starred" text remains.
- Screenshots: 1200 px light and dark, and 390 px light and dark (via an iframe), inspected section by section.
- A rebuild from / is byte-identical.

## Coordinator items
1. Outcome panel (header, slot `outcome`) and the §7 lead (`mainline`): **fixed.** They give B 1 (T3) and C 1 (T2 at 30 dB) of 5; the tie-break B −1.60 (50 cubes) vs C −0.84 (200); main line B; B wins on mean accuracy (0.964 / 0.935 × D), not calibration (cov90 0.288 / 0.365); both lines reported in full; and the three-way bullets. Every number is read from the verdict or D.
2. D "best at every SNR" removed from the lead, `dbest` and the aria; D is best at 60–40 dB and B at 30–20 dB, computed on the common cubes (`facts.e1_best_common`). The CRPS arias are fixed, and "wrong at low SNR" now says C failed at every SNR: **fixed.**
3. Clause-count rule "of 5" with the verdict's counting (T1 once, T2 per SNR, T3, T4), the passes named, and a main-line row with the tie-break: **fixed.**
4. The sanity row shows monotone yes/no with the 60→20 dB sequence and the floor value for B and C, plus "prior-limited, flat within noise, +0.09 from 40 to 30 dB": **fixed.**
5. Progress artefacts removed (status box, "n of n records", starred/preview sentence, "so far"; "total" in the budget): **fixed.** A pending notice appears only while records are missing.
6. E4 added: an extractor setting; a row in the snapshot uncertainty table, the E3 chart data table and the B table; an E4 bullet whose wording is conditional on the data; and an item in §9. The B cell shows pending: **fixed.**
7. "60 dB noise floor (σ² = P_y/10⁶), not noise-free" is now in the glossary and the lede. The legacy noise-free 0.266 % is quoted once, with the ratio, in the Stage 0 caption: **fixed.**
8. The coverage axis now runs from 0, so B's cov50 is visible. A "scroll sideways for the full chart/table →" cue is added by JS wherever a .scroll or .tbl overflows (verified at 390 px). The snapshot table is compacted to three-line cells. Dark-mode colours were checked: **fixed.**
9. Probe caption: **fixed**, with the numbers computed from the literals. The real gaps are 0.3 and 0.6 at t = 25 / 50, so the page says "within 0.6", not the requested "0.1", and 1.2–2.5 from t = 100 to 200; the two pilots are within 0.8.
10. One-line items, below.

## Checker 1 (pagecheck1)
- #1–6: fixed, as above.
- #7 (monotone): fixed.
- #8 (cov90 ≥ 0.99): fixed; now 3 decimals, ≥ 0.985.
- #9 (coverage axis): fixed.
- #10 ("(50 cubes)" tags): fixed in the AUSE and coverage charts and their captions.
- #11 (x₀ rule): fixed. The `x0rule` slot computes the drop per t from the literals: met at t = 50–299, 25 % at t = 25. It also states that the rule was set after the failure and that B's verdict scores an amended line.
- #12 (epoch counts): fixed ("four of 100 epochs and the variance head's 40").
- #13 (brec snapshot columns): fixed (L exact / L demosaic / D / C).
- #14 (eval hours include E4): fixed. E4 is now on the page and the budget says so.
- #15 (disclosure of 50 vs 200 cubes in the tie-break): fixed in the outcome panel, the main-line lead and §9.

## Checker 2 (pagecheck2)
- #1, #2: fixed.
- #3–7: fixed.
- #8: fixed. The KPI now uses C at 30 dB (its T2 pass), §3 is retitled "uncertainty", and `t2text` explains why CRPS can favour an over-confident sampler with a good mean.
- #9–12: fixed.
- #13: fixed with the scroll cue. The SVGs keep a minimum width rather than shrinking, to keep text legible.
- #14: fixed.
- #15: fixed (bold readings for B on Device 2 and in snapshot, including "B cannot demosaic").
- #16: fixed (PH5: "only V's CRPS stays level with L's", with the numbers).
- #17: fixed (the Phase 2a CRPS sentence moved to the CRPS caption; AUSE given to 3 decimals, 0.302–0.355).
- #18: fixed.
- #19: fixed (pairs ordered 60 then 50 dB, "still" dropped).
- #20: partly fixed. The dash patterns now differ, the pilots are drawn in --muted, the x₀ pilot is no longer faint, and the mean-cube label moved to t ≈ 520. Thin curves still cross the label slightly.
- #21: fixed (labels staggered, T4 segment neutral, E3 grid noted in the lead).
- #22: fixed (compact cells).
- #23: fixed (the SV caption notes the clamp).
- #24: partly fixed. The glossary defines mean-of-32 as the sampler's posterior-mean estimate and L exact vs L demosaic. The remaining "posterior mean" wording stays.
- #25: fixed (§8 states seed 42 and one run per network, sample k under torch seed k, crc32 noise seeds; the budget gives the total ≈ 72 GPU-h).
- #26: fixed (header and §9 call it an amendment set after the failure).
- #27: fixed (the toy path is now relative, `results/residual_warmstart/sensor/toy_guidance.json`).
- #28: fixed ("about 11–13"; the 13 is static, taken from the spec).

## Not fixed, and why
- Two scientific references to Phase 2a remain on purpose, both from the results note: C's CRPS win is the opposite of Phase 2a, and Phase 2a's D-centred construction is named as the fairer snapshot test.
- Probe gap at t = 25 / 50: the request said 0.1, but the literals give 0.3 and 0.6, so the page reports the data.
- The mean-cube label in the probe chart is still lightly crossed by curves (cosmetic).
- The run-table index binding (`runs.2/3`) still depends on RUN_TABLE order.
