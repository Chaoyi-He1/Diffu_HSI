# Residual warm-start Phase 1a: ledger (as built)

This is the tracked copy of the controller's SDD ledger, `.superpowers/sdd/2026-10-09-residual-warm-start-phase1a/progress.md`. That file is git-ignored, and `git worktree remove` deletes it. The copy was made in the final-review fix wave on 2026-10-12, and it keeps what the plan and the RESULTS note do not:
- every controller ruling, with its why and cost;
- the deferred minor findings, which form the backlog for Phase 1b and for the driver;
- the final review's corrections and its parked items;
- the fix-wave commits.

Snapshot of progress.md: 86 lines, sha256 `912650e1d89aa497…`. `L<n>` below is the line number in that snapshot. Quoted lines are verbatim; text outside the quotes is added here.

Related documents: the plan `2026-10-09-residual-warm-start-phase1a.md` (see its "As built (2026-10-12)" block), the results `2026-10-09-residual-warm-start-phase1a-RESULTS.md`, and the spec `docs/superpowers/specs/2026-10-08-residual-warm-start-design.md`.

## 1. Rulings (verbatim)

These are all the ledger lines that carry a ruling. A ruling made inside a task note (L70, L74) is quoted with its whole line.

- **L36.** Ruling (Finding A): Tasks 3 and 6 run from the WORKTREE, not the main checkout, with the absolute `RESULTS_ROOT` default (all artefacts land in /data/chaoyi_he/HSI/Diffu/results/residual_warmstart, outside the worktree). Execution order 1, 2, 4, 5, 3, 6. — why: a mid-plan merge to master would bypass the whole-branch review and the finishing step; the plan's real requirement (artefacts survive worktree removal) is met by the absolute root — cost if wrong: none, provided the worktree stays until Phase 1a finishes (the running pilot imports code from it).
  - *Note (added here):* Recorded in the RESULTS note, Observation 7 ("Execution location"): training ran at 7da6b93 and the evaluations at 2fd2bee, both on the branch worktree-phase1-plan.
- **L37.** Ruling: EMA = none (user default accepted by "approve"; recorded so it is not re-asked).
- **L44.** Ruling (Task 2 fix round 1): on resume refuse (ValueError) when ck[meta] differs from checkpoint_meta(args) in mode, d, base_channels, prediction_type, sigma_d (|diff| >= 1e-9) or prior_sha256; a num_epochs != scheduler T_max mismatch is a loud printed warning, not a refusal (the Task 3 smoke test deliberately resumes 2 -> 3 epochs; the pilot never changes num_epochs) — why: provenance must not drift; the schedule caveat is documented — cost if wrong: one more resume flag later. Promoted minor: make_datasets asserts all 200 gate files were found and removed (mirrors dataset_for).
- **L51.** Ruling (Task 4 fix round 1): fix all five; refuse unequal final epochs by default (spec: same budget) with --allow_unequal_epochs overriding and the verdict annotated; default output name gate_1a__<B run_name>[__nodone][__nfe<N>][__k<K>].json; promoted minors: strict grid key (missing grid refused), refuse when ddim10 or ddpm1000 rows are missing (a full grid always has them), refuse n < 2 cubes, missing provenance keys -> ValueError — why: the gate is the research decision; its record must be unambiguous — cost if wrong: one fix round
- **L58.** Ruling (Task 5 fix round 1): count EVERY launch (launches file, cap 1 + MAX_RELAUNCHES); per-action failure markers with MAX_ACTION_FAILURES = 2 then SKIP + loud log; the loop exits when the official gate file exists or when a pass is STUCK (no actions possible, no run alive); try/except per pass with traceback logged; lenient DONE parsing; subprocess timeouts (eval_full 8 h, eval_quick 2 h, launch/gate 1 h); exact gate filename; quick evals also after DONE; heartbeat line per idle pass; exit right after the gate pass — why: this runs 16 h unattended — cost if wrong: one fix round
  - *Note (added here):* The plan text still says `relaunches`; the as-built names are in the plan's "As built (2026-10-12)" block.
- **L62.** Ruling (Task 5 STUCK policy): the loop exits rc 2 when no action is executable and none of our runs is alive, even if a foreign job holds the GPUs or nvidia-smi fails — accepted for the pilot: the GPUs are dedicated to it, the exit is loud (STUCK line) and the driver is re-invocable — cost if wrong: a manual restart of the driver
  - *Note (added here):* Amended by L84 (STUCK amendment, commit e6404f0): a failed nvidia-smi (empty GPU status) now means wait, logged as SMI-FAIL on every pass; rc 2 is kept when foreign jobs hold the GPUs.
- **L68.** Ruling (pre-launch, from the Task 3 review): Task 6 expectations re-baselined to 306 s/epoch (8.5 h per run, ~20-21 h critical path incl. ~3.2 h full-grid evals); before launching, probe the eval batch size at base 64 (8/16/32/64) on an idle GPU and set --batch_size in eval_milestone.sh to the largest size that gives >= 1.5x per-cube speed-up (same value for quick and full and for all three arms; the noise draw changes with batch size but all arms stay paired) — why: the full-grid evals are on the critical path — cost if wrong: milestone-vs-final evals not bitwise comparable (irrelevant to the gate)
  - *Note (added here):* The 306 s/epoch baseline was itself measured without the thread limit and is about 19 % high; with the limit a lone base-64 run takes 0.227 s/step, about 256 s/epoch (final-review item 9; RESULTS Observation 5).
- **L70.** Task 6: first launch at 465 s/epoch (expected 306): 16 loader workers x ~34 BLAS threads, load ~50 on 32 cores, GPUs 45-50 % busy; data wait 1 %. Ruling: stop the driver and both runs, delete the two run dirs, add OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 to launch_run.sh (inherited by the workers), relaunch — why: ~1.5x per-epoch speed-up for a 16-minute loss; the pilot would otherwise take 26-28 h — cost if wrong: 16 minutes and one extra commit
  - *Note (added here):* Fix-wave item 5 (6ceff1f, e980f7c) moved the limit into the Python DataLoaders as well (`worker_init_fn=limit_worker_threads`); the shell exports stay.
- **L74.** Task 6: B and A DONE at epoch 100 (B final loss 0.1060 / val 0.1032; A 0.0747 / 0.0732; 9.0 h each). Driver started B full eval on GPU 0 at 23:29 but did not launch A-prime (GPU 1 still busy with A exiting on that pass) and then blocked in the synchronous eval. Ruling: controller launched A-prime manually at 00:54:33 with launch_run.sh (same script the driver uses; idempotent) — why: gained ≈ 1.9 h on the critical path (CORRECTED by the final review: at the 23:29:22 pass A had not yet written DONE — checkpoint 23:29:44.71, DONE 23:29:44.85 — so the policy correctly did not launch A-prime; the driver then sat in B's synchronous eval until 02:38) — cost if wrong: none (driver sees it alive). Deferred for the driver: launch while an eval runs (final-review item 7) or run evals asynchronously.
  - *Note (added here):* Corrected by the controller from final-review item 8; see section 3.
- **L79.** Ruling (2026-10-12, session switch to Opus 5.5 + ultracode at the user request): commit trailer from now on is "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" per the updated session attribution; earlier commits keep their Fable 5.1 trailers (correct when made); the remaining review steps (results-note review, whole-branch review, fix-wave re-review) run as multi-agent workflows with adversarial verification — why: user instruction "use opus 5.5 with ultracode effort to continue" — cost if wrong: none
- **L80.** Ruling (review process): the Task 6 task review (results note + the two BLAS-limit commits 7da6b93/2fd2bee made during the pilot) and the whole-branch final review run in ONE workflow (parallel lenses, adversarial verification of every Critical/Important finding, one synthesis); their findings go to ONE fix wave and one scoped re-review — why: the note and the branch are reviewed against the same sources; avoids a separate fix loop for a docs task — cost if wrong: a finding about the note is fixed in the same wave as code findings rather than before the final review
- **L84.** Ruling (final-review fix wave): accept all 16 items as specified by the synthesis; resolve the five severity/scope splits as the synthesis recommends — Observation 1 rewrite important (item 2 wording); wall-clock semantics item 6 in full; as-built record as a tracked LEDGER.md (item 16); driver stopgap item 7 (launch while a synchronous eval runs + heartbeat + flock), async evals deferred to the driver generalisation; STUCK amendment (overturns part of the Task 5 STUCK ruling): gpu_status()=={} (nvidia-smi failure) means wait and log SMI-FAIL each pass, rc 2 kept for foreign jobs holding the GPUs. Post-review gate output goes to a NEW file gate_1a__residual_d4_b64__postreview.json; the official gate record is never overwritten. — why: every item is confirmed by two independent verifiers; the note must not misstate the science before the user decides — cost if wrong: a larger fix wave than strictly needed
- **L85.** Ruling (fix-wave execution): one workflow, four fix agents on disjoint file sets with explicit-path commits (git commit -F msg -- <paths>): B training/loaders/paths (items 5, 6-training, 13, 15) first; then C harness/gate/y-swap script (items 1, 10, 11, 12 code) in parallel with D driver (items 6-driver, 7, 14, STUCK amendment); then A docs (note items 1-4, 8-11, plan as-built + tracked LEDGER.md, item 16 minus the push); then one scoped re-review workflow stage (3 lenses + full suite). Controller pushes after the re-review. — why: the one-fix-agent rule exists to avoid per-finding context rebuilds; four scoped agents on disjoint files keep each context focused without that cost — cost if wrong: a merge-order slip between C and D (disjoint files make it unlikely)

## 2. Deferred minor findings, grouped

Each item is one clause of a `minor (deferred)` line, quoted verbatim. Re-joined with "; ", the clauses of a line give the whole line; section 2.5 quotes the whole lines. The status column records where each item stands after the final review and its fix wave. "Parked" means the final review parked it (section 4).

### 2.1 Phase 1b hazards

Section 4 also lists the parked pre-1b items: the full-size FSDP save/resume probe, refusing a resume after a change of seed, loss, SNR weighting or batch size, and the disk and RAM sizing.

| source | deferred item (verbatim) | status after the fix wave |
|---|---|---|
| L45 (Task 2) | duplicate epoch lines in train_log.jsonl after a crash-resume and args.json overwritten on relaunch | fixed in the fix wave: item 13 (c) and (d) (c59d31e): train_log.jsonl is trimmed to the restored epoch, and every launch writes args_launch<N>.json |
| L45 (Task 2) | non-finite grad norm not skipped | fixed in the fix wave: item 13 (a) (c59d31e): a non-finite gradient norm skips the step and counts toward the max_nonfinite abort |
| L45 (Task 2) | CPU exact resume needs num_workers=0 (persistent-worker base seed drawn once; GPU unaffected) | parked (moot on GPU: noise comes from the device RNG) |
| L45 (Task 2) | Phase 1b sizing notes (11 GB checkpoints x5 per run, every rank torch.loads the file, NCCL 10-min default timeout during the rank-0 write, save_every 10 loses <= 2.5 h) | partly fixed in the fix wave: the NCCL timeout is 60 min (item 13 (b), c59d31e). Parked as a pre-1b GPU task: a full-size FSDP save/resume probe, mmap=True torch.load, save_every 5. The final review corrects the loss: save_every 10 loses up to 2.89 h at batch 4, not 2.5 h. Checkpoints are 11.30 GB each (section 4, "Phase 1b disk and RAM sizing") |
| L45 (Task 2) | single-GPU and FSDP checkpoints not interchangeable for resume | parked (1b starts FSDP from scratch; the model state loads in both) |

### 2.2 Driver follow-ups

| source | deferred item (verbatim) | status after the fix wave |
|---|---|---|
| L59 (Task 5) | pid_alive could also check /proc/<pid>/cmdline | parked (driver cosmetics) |
| L59 (Task 5) | no single-instance lock (flock) between --once and --loop | fixed in the fix wave: item 7 (accd02d), flock on runs/pilot_driver.lock and a second driver exits 3; launch_run.sh takes .launch.lock (item 14, 1c60541) |
| L59 (Task 5) | status.py val only every 5th epoch + spiky ETA + no __main__ guard | parked (driver cosmetics); status.py does not show STOPPED either (agent D) |
| L59 (Task 5) | quick evals share the training GPU (plan-specified; an OOM would cost a relaunch) | parked (no OOM in 6 quick evaluations; they cost +49–75 % on 4–5 epochs each) |
| L59 (Task 5) | nvidia-smi --id filter for --query-compute-apps unverified while a process runs (Task 3 smoke will show) | moot (verified in production) |
| L63 (Task 5) | Ctrl-C on an interactive driver no longer kills the child eval (own session) | parked (driver cosmetics) |
| L63 (Task 5) | failure keys never expire (two transient <3 GB refusals permanently cap a launch key; an OOM-capped eval_quick is never retried after DONE) | parked (driver cosmetics). Since item 7 a launch attempted from an evaluation tick retries every 60 s, so two refusals within about 2 min cap a launch key (agent D) |
| L63 (Task 5) | eval_full failure key includes the GPU so a capped GPU-0 action is not retried on GPU 1 while GPU 0 is idle | parked (never fired: there is no action_failures.json; fix with the driver generalisation) |
| L63 (Task 5) | log_event inside the except handler can itself raise (ENOSPC) | parked (driver cosmetics) |
| L74 (Task 6) | Deferred for the driver: launch while an eval runs (final-review item 7) or run evals asynchronously. | stopgap fixed in the fix wave: item 7 (accd02d); asynchronous evaluations are deferred to the driver generalisation (ruling L84) |

### 2.3 Test gaps

| source | deferred item (verbatim) | status after the fix wave |
|---|---|---|
| L40 (Task 1) | checkpoint_eval wiring tests do not cover grid=quick pass-through, the getattr defaults or the run_name fallback | parked (Task 1 cosmetics; behaviour verified in the pilot) |
| L40 (Task 1) | reconstruct/yswap tests loop over residual/standard only (no standard_xhat case) | parked (Task 1 cosmetics; the pilot's A′ swap columns differ) |
| L40 (Task 1) | no test that an A-prime checkpoint with meta is rejected under --mode residual | parked (Task 1 cosmetics; 9/9 evaluations meta_checked) |
| L40 (Task 1) | build_residual_model test does not assert m.training | parked (Task 1 cosmetics) |
| L45 (Task 2) | test gaps (wall-clock stop, DONE contents, --resume auto through build_everything; test_run_epochs_calls_save_on_every_rank uses enabled=False; unused `before`) | partly fixed in the fix wave: wall-clock stop and DONE contents (item 6, 90ab7d6); --resume auto through build_everything (item 13, c59d31e). Still open: the save-on-every-rank test uses enabled=False; unused `before` |
| L48 (Task 2) | test_resume_meta_mode_mismatch_raises match="mode" is vacuous because tmp_path contains "mode" (use "mode: checkpoint") | parked (the guard is still pinned; tighten to match='mode: checkpoint' when the file is next touched); still match='mode' after the fix wave |
| L48 (Task 2) | no sigma_d wiring test through build_everything | parked (Task 2 cosmetics) |
| L52 (Task 4) | test gaps (prediction_type refusal, ablation_blind for A-prime, prereg/intersection/b_setting contents, CLI) | open (not taken up by the final review) |
| L59 (Task 5) | test gaps (epoch-50 quick, gate with 2/3 JSONs, two idle GPUs -> two evals, A-prime not launching while A runs, execute/gpu_status untested) | moot (final review: Task 5 test gaps now covered) |
| L63 (Task 5) | run_cmd timeout test asserts the return value only | fixed in the fix wave: item 14 (1c60541), the test asserts that os.killpg(pgid, 0) raises ProcessLookupError |
| L66 (Task 3) | smoke refusal check swallows stderr (grep "does not match this evaluation" if reused) | parked (the refusal is pinned at unit level) |

### 2.4 Code and record cosmetics

| source | deferred item (verbatim) | status after the fix wave |
|---|---|---|
| L40 (Task 1) | target_for does not validate mode (build_residual_model does) | parked (final review: Task 1 cosmetics) |
| L40 (Task 1) | run_name default expression duplicated (eval_warmstart.py:272,313) | parked (Task 1 cosmetics) |
| L40 (Task 1) | mid-file imports in the two test files | parked (Task 1 cosmetics) |
| L40 (Task 1) | yswap_check/reconstruct comments say residual/standard only | parked (Task 1 cosmetics) |
| L40 (Task 1) | module docstrings do not mention the absolute RESULTS_ROOT | parked (fix the obsolete layout at eval_warmstart.py:4-8 when the file is next touched); still open after the fix wave |
| L40 (Task 1) | time_train_step.py and compute_residual_scale.py keep a repo-relative PRIOR (same bytes today) | fixed in the fix wave: item 15 (25780bc), one definition in data_loader/warmstart_paths.py |
| L45 (Task 2) | test-only tuple branch + one-line meta fallback in run_epochs | parked (Task 2 cosmetics) |
| L45 (Task 2) | file_sha256/PRIOR/SUBSET duplicated from eval_warmstart | fixed in the fix wave: item 15 (25780bc, e980f7c) |
| L45 (Task 2) | --save_every 0 ZeroDivisionError | parked (Task 2 cosmetics) |
| L45 (Task 2) | --keep_last 0 keeps all | parked (Task 2 cosmetics) |
| L45 (Task 2) | dry_run StopIteration when steps+warm-up > len(loader) and skips the free-memory check | parked (Task 2 cosmetics) |
| L48 (Task 2) | make_datasets uses assert (stripped under -O) | parked (Task 2 cosmetics) |
| L52 (Task 4) | vs-baseline trusts stored paired_rmse_diff/se rather than recomputing from baseline_per_item | parked (recomputed from baseline_per_item, equal to 1e-8 on the pilot); the fix-wave gate recomputes only the clustered SE |
| L52 (Task 4) | CLI does not explain PASS_NOT_OVER_APRIME | parked (gate cosmetics) |
| L52 (Task 4) | b_setting_applied_to is an opaque name | parked (gate cosmetics) |
| L52 (Task 4) | a/aprime lack rmse_pct_se | parked (gate cosmetics) |
| L52 (Task 4) | "baseline (200 cubes)" hard-coded | parked (gate cosmetics) |
| L52 (Task 4) | open() without with (check_done, main) | moot (fixed before the final review) |
| L52 (Task 4) | unused sys.path insert | moot since the fix wave: the insert now serves `from scripts.paired_stats import cluster_se` (215946c) |
| L55 (Task 4) | default_out_name does not encode --allow_unequal_epochs (an unequal-epoch re-run overwrites the official file; inputs/final_epochs inside the JSON make it visible) | parked (gate cosmetics) |
| L55 (Task 4) | --no_require_done peeks of different epochs share a filename | open (not taken up by the final review) |
| L63 (Task 5) | the plan doc still says relaunches | fixed by the plan's "As built (2026-10-12)" block (fix wave, docs commit) |
| L66 (Task 3) | +17 % step time vs Phase 0 attributed to GPU compute without evidence (likelier host contention from 8 workers; the Task 6 data<=10 % check is the guard) | resolved by final-review item 9: the Task 3 record ran without the thread limit and is about 19 % high (RESULTS Observation 5) |
| L66 (Task 3) | record names/schema deviate from the brief (eval_quick_b16_smoke with sec; 6th width-probe record kept as provenance) | parked (the records describe themselves) |
| L66 (Task 3) | base-64 quick-grid time derived not measured | moot (measured in the pilot: 26.5–27.5 min) |
| L66 (Task 3) | plan line "48 NFE / 45x" wrong (quick grid is 78 NFE per cube-seed) | fixed by the plan's "As built (2026-10-12)" block (fix wave, docs commit) |

### 2.5 The deferred lines as recorded

- **L40.** Task 1: minor (deferred): target_for does not validate mode (build_residual_model does); run_name default expression duplicated (eval_warmstart.py:272,313); checkpoint_eval wiring tests do not cover grid=quick pass-through, the getattr defaults or the run_name fallback; reconstruct/yswap tests loop over residual/standard only (no standard_xhat case); no test that an A-prime checkpoint with meta is rejected under --mode residual; mid-file imports in the two test files; build_residual_model test does not assert m.training; yswap_check/reconstruct comments say residual/standard only; module docstrings do not mention the absolute RESULTS_ROOT; time_train_step.py and compute_residual_scale.py keep a repo-relative PRIOR (same bytes today)
- **L45.** Task 2: minor (deferred): test-only tuple branch + one-line meta fallback in run_epochs; file_sha256/PRIOR/SUBSET duplicated from eval_warmstart; duplicate epoch lines in train_log.jsonl after a crash-resume and args.json overwritten on relaunch; non-finite grad norm not skipped; --save_every 0 ZeroDivisionError; --keep_last 0 keeps all; dry_run StopIteration when steps+warm-up > len(loader) and skips the free-memory check; CPU exact resume needs num_workers=0 (persistent-worker base seed drawn once; GPU unaffected); Phase 1b sizing notes (11 GB checkpoints x5 per run, every rank torch.loads the file, NCCL 10-min default timeout during the rank-0 write, save_every 10 loses <= 2.5 h); single-GPU and FSDP checkpoints not interchangeable for resume; test gaps (wall-clock stop, DONE contents, --resume auto through build_everything; test_run_epochs_calls_save_on_every_rank uses enabled=False; unused `before`)
- **L48.** Task 2: minor (deferred): test_resume_meta_mode_mismatch_raises match="mode" is vacuous because tmp_path contains "mode" (use "mode: checkpoint"); make_datasets uses assert (stripped under -O); no sigma_d wiring test through build_everything
- **L52.** Task 4: minor (deferred): vs-baseline trusts stored paired_rmse_diff/se rather than recomputing from baseline_per_item; CLI does not explain PASS_NOT_OVER_APRIME; b_setting_applied_to is an opaque name; a/aprime lack rmse_pct_se; "baseline (200 cubes)" hard-coded; open() without with (check_done, main); unused sys.path insert; test gaps (prediction_type refusal, ablation_blind for A-prime, prereg/intersection/b_setting contents, CLI)
- **L55.** Task 4: minor (deferred): default_out_name does not encode --allow_unequal_epochs (an unequal-epoch re-run overwrites the official file; inputs/final_epochs inside the JSON make it visible); --no_require_done peeks of different epochs share a filename
- **L59.** Task 5: minor (deferred): pid_alive could also check /proc/<pid>/cmdline; no single-instance lock (flock) between --once and --loop; status.py val only every 5th epoch + spiky ETA + no __main__ guard; quick evals share the training GPU (plan-specified; an OOM would cost a relaunch); test gaps (epoch-50 quick, gate with 2/3 JSONs, two idle GPUs -> two evals, A-prime not launching while A runs, execute/gpu_status untested); nvidia-smi --id filter for --query-compute-apps unverified while a process runs (Task 3 smoke will show)
- **L63.** Task 5: minor (deferred): Ctrl-C on an interactive driver no longer kills the child eval (own session); failure keys never expire (two transient <3 GB refusals permanently cap a launch key; an OOM-capped eval_quick is never retried after DONE); eval_full failure key includes the GPU so a capped GPU-0 action is not retried on GPU 1 while GPU 0 is idle; log_event inside the except handler can itself raise (ENOSPC); run_cmd timeout test asserts the return value only; the plan doc still says relaunches
- **L66.** Task 3: minor (deferred): +17 % step time vs Phase 0 attributed to GPU compute without evidence (likelier host contention from 8 workers; the Task 6 data<=10 % check is the guard); record names/schema deviate from the brief (eval_quick_b16_smoke with sec; 6th width-probe record kept as provenance); base-64 quick-grid time derived not measured; smoke refusal check swallows stderr (grep "does not match this evaluation" if reused); plan line "48 NFE / 45x" wrong (quick grid is 78 NFE per cube-seed)

## 3. Corrections from the final review (item 8)

The ledger's account of the manual A′ launch had the cause and the times wrong. The controller corrected L74 in place (section 1 quotes the corrected line). The corrected facts, from the file mtimes and `pilot_events.log`, are:
- At the 23:29:22 pass A had not yet written DONE. It was finishing epoch 100: the checkpoint is from 23:29:44.71 and DONE from 23:29:44.85. So the policy, which launches A′ only after A's DONE (pilot_driver.py:96 in the pilot's code), correctly did not launch A′.
- The same pass started B's full evaluation on GPU 0. Actions run synchronously by design (plan Task 5), so the driver sat in that evaluation until 02:38:33 and did not observe again until 02:48:33.
- The controller launched A′ by hand at **00:54:33**, the mtime of its `launches` and `pid` files; the ledger had said 00:0x. The `MANUAL` line in `pilot_events.log` is logged at 00:54:44, with pid 646601. GPU 1 had sat idle for 1.41 h.
- Without the manual launch the driver would have started A′ at 02:48:33, so the launch gained **≈ 1.9 h**; the ledger had said ~3 h.
- The driver was inside synchronous evaluations for 5.78 h of A′'s 7.71 h of training. A crash in those windows would have waited up to ≈ 3 h for a relaunch; none occurred.
- The deferred fix is now "launch while an eval runs (item 7) or run evals asynchronously". It replaces "launch actions should be attempted again after the synchronous eval", because that retry would only have launched A′ at 02:38:33. The stopgap is done (section 5).

The RESULTS note carries the same corrections in the Runs table, its notes, and Observation 7.

## 4. Parked by the final review (28 entries, verbatim from the synthesis)

- **Rejected sub-claim: 'the best ≤ 10-NFE row is structurally the least stochastic (warm50 for A and A′)'.** Both verifiers rejected it. For A and A′, ddim1 is the same single call yet scores 13.83 and 8.14: warm50 wins on its centre (x̂₀), not on low spread. Several B rows with lower |seed0 − seed1| than ddim1 still lose.
- **Rejected sub-claim: 'DDIM-10 spread alone (4.33 pct²) exceeds 3.99 pct², so it fails 1b even with a perfect centre'.** 4.33 is DDIM-10's excess mean square over ddim1, not a measured spread. The gate averages per-cube RMSE, so the threshold is ≈ 4.14–4.40 pct², not 3.99. A calibrated sampler with a perfect centre has zero spread.
- **Rejected sub-claim (yswap-threshold): 'B ignores the sensor tokens'.** The paired sensor_only − true difference is +0.0183 ± 0.0037 at t = 999 (5 SE), so B uses the sensor path weakly rather than not at all. Item 1's wording reflects this.
- **Rejected sub-claim: 'at t = 300 B uses its condition less than the Bayes optimum (0.983 vs 0.943)'.** The per-element Gaussian model ignores spatial correlation in r, which pushes the Bayes ratio toward 1, so under-use is not established.
- **Partly rejected sub-claim (sync-evals): 'a restarted driver re-issues the in-flight B eval'.** This needs a contrived state. Real duplicates arise for an in-flight quick eval, and for a full eval when the other GPU is free. Eval locks come with the async rewrite.
- **Partial-information oracle test for a re-specified y-swap rule.** Waits until the user approves a new rule. Spec §9 requires only that a condition-blind stub fails.
- **Gate pass/fail at t = 999 on a 32-cube y-swap batch.** This is a spec §7/§8/§11 change and is on the Decision as a user question (item 4, step 3). Item 1 adds only per-cube reporting.
- **K-sample-mean, sample-spread and K-noise-averaged t = 999 diagnostics.** These need GPU and a sample-keeping script, so they are proposed to the user as Decision step 0 rather than given to the fix agent.
- **quick-eval-ignores-idle-gpu (fin-driver minor; deferred item promoted by fin-driver, parked by t6-ops).** Saved ≈ 23 min on A′ at most. Sharing the run's GPU is approved deviation 5. GPU allocation changes when evals go async, so revisit then.
- **capped-eval-starves-gpu, and the deferred 'eval_full failure key includes the GPU'.** Never fired: there is no action_failures.json. Fix together with the driver generalisation.
- **hardcoded-pilot-config (driver and helpers fixed to d4/b64, three arms, single GPU, 8 h eval timeout).** Parametrise when a sweep or 1b driver is planned, and time one base-256 quick eval before setting the timeouts.
- **filename-keyed-state-reuse: mtime check on eval JSONs.** Use a fresh RESIDUAL_WARMSTART_RESULTS root per campaign. The cheap 'refuse when the gate file exists' part is in item 14.
- **Full-size FSDP save/resume probe (base 256, 2 GPUs), mmap=True torch.load, save_every 5 for 1b.** A pre-1b GPU task. The estimate gives ≈ 25× margin on the timeout, which item 13 raises to 60 min. save_every 10 loses up to 2.89 h at batch 4, not the 2.5 h in the ledger.
- **Refuse a resume when seed, loss_type, snr_gamma or batch_size changed.** Move to the 1b plan. Item 13 adds provenance only.
- **os.environ.setdefault for the thread variables in the main process.** Not needed once the workers are limited by item 5. It would also cap the main process's torch intra-op threads.
- **Quick evals share the training GPU.** No OOM in 6 quick evals. The cost was +49–75 % on 4–5 epochs each (≈ 41 min on the A→A′ chain), acceptable for a pilot.
- **Phase 1b disk and RAM sizing.** 11.30 GB per checkpoint; 56.5 GB retained (67.8 GB at peak) against 497 GB free. Two ranks loading the full file need 2×11.3 + 2×3.8 GB against 119 GB available. The 1b plan should set its own keep_epochs.
- **Task 2 cosmetics: test-only tuple branch and meta fallback; --save_every 0 ZeroDivisionError; --keep_last 0 keeps all; dry_run StopIteration; make_datasets assert under -O; no sigma_d wiring test.** No caller reaches them, and the pilot outputs are correct (meta_checked true; val = 800).
- **test_resume_meta_mode_mismatch_raises uses a weak match='mode'.** The guard is still pinned: removing it gives DID NOT RAISE. Tighten to match='mode: checkpoint' when the file is next touched.
- **Single-GPU and FSDP checkpoints are not interchangeable for resume.** 1b starts FSDP from scratch, and the model state loads in both.
- **CPU exact resume needs num_workers=0.** Moot on GPU: noise comes from the device RNG and the image-mode dataset is deterministic.
- **Gate trusts the stored paired_rmse_diff/se.** Recomputed from baseline_per_item; it equals the stored values to 1e-8 on the pilot.
- **Gate cosmetics: a/aprime lack rmse_pct_se; 'baseline (200 cubes)' is hard-coded; the CLI does not explain PASS_NOT_OVER_APRIME; the b_setting_applied_to name; an unused sys.path insert; default_out_name ignores --allow_unequal_epochs.** All the values are recorded in the JSON or explained in the note.
- **Task 1 cosmetics: target_for mode validation; duplicated run_name expression; mid-file test imports; no m.training assert; comments and docstrings (the obsolete results layout at eval_warmstart.py:4-8, fix if the file is touched); reconstruct/yswap tests lack standard_xhat; no A′-meta-under-residual test.** Behaviour verified in the pilot (A′ swap columns differ; 9/9 meta_checked).
- **Driver cosmetics: pid_alive could check /proc cmdline; Ctrl-C no longer kills the child eval; failure keys never expire; log_event can raise on ENOSPC; status.py ETA, val cadence and missing __main__ guard.** None fired in 22 h, and status.py ran correctly on the final artefacts.
- **smoke_test.sh refusal check swallows stderr.** The refusal is pinned at unit level (test_check_meta_names_every_mismatch). The smoke test was a one-off and passed.
- **Task 3 timing record names and schema deviate from the brief.** The records describe themselves (sec_basis, note fields). The oversubscribed single-GPU record is corrected in the note by item 9.
- **Moot items.** Resolved by the pilot or by later commits:
  - nvidia-smi --id filtering, verified in production
  - Task 5 test gaps, now covered
  - meta always written: 9/9 meta_checked
  - quick-grid wiring: 6 quick JSONs are correct
  - base-64 quick time: measured at 26.5–27.5 min
  - eval batch probe: flat
  - EMA none: deviation 4
  - Task 2/4/5 fix-round rulings: implemented
  - Opus trailer ruling: verified
  - open() without with: fixed

## 5. Final-review fix wave (2026-10-12)

Review and dispatch, verbatim from the ledger:
- **L82.** Final review dispatched as workflow wl46iv0mr (run wf_3a69d9f0-363): lenses t6-numbers, t6-science, t6-ops, fin-train, fin-harness-gate, fin-driver, fin-integration; confirm+refute per Critical/Important; one synthesis
- **L83.** Final review result (workflow wf_3a69d9f0-363, 32 agents): 12 Critical/Important findings, all CONFIRMED by both confirm and refute agents (0 disputed, 0 refuted), 27 minors; synthesis -> 16 fix items + 28 parked (saved at /home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/final_review_synthesis.json); full suite 142 passed on CPU. Key science finding: a post-hoc CPU y-swap at t=999 shows B's one-step gain is cube-specific (swap_full - true +0.2225 +- 0.0136 on 32 cubes) — INVALID comes from the 10 % threshold's lack of power at a 4.6 % gain, not from an unused condition; the sqrt(2) argument does not discriminate (a calibrated posterior sampler also pays sqrt(2) under RMSE).
- **L86.** Final-review fix wave dispatched as workflow we0ueswhd (run wf_adf5bccc-8d7): B -> (C || D) -> A -> re-review R1/R2/R3; FIX_BASE a9bb5c9

The controller's rulings for the wave are L84 and L85 in section 1. The commits, all on worktree-phase1-plan and made with explicit paths:

| agent | scope | commit | subject |
|---|---|---|---|
| B | training, loaders, paths (items 5, 6 training, 13, 15) | 6ceff1f | fix(warmstart): one BLAS thread per DataLoader worker via worker_init_fn (fix item 5) |
| B | training, loaders, paths (items 5, 6 training, 13, 15) | 25780bc | fix(warmstart): define RESULTS_ROOT, PRIOR, SUBSET and file_sha256 once in data_loader/warmstart_paths.py (fix item 15) |
| B | training, loaders, paths (items 5, 6 training, 13, 15) | 90ab7d6 | fix(warmstart): --max_hours defaults to no cap; STOPPED record, no stale DONE, refuse a resume past the cap (fix item 6, training part) |
| B | training, loaders, paths (items 5, 6 training, 13, 15) | c59d31e | fix(warmstart): skip non-finite grad norms, 60 min NCCL timeout, resume-safe train_log, per-launch args, git_head in meta (fix item 13) |
| B | training, loaders, paths (items 5, 6 training, 13, 15) | dcbeb2b | docs(warmstart): --max_nonfinite help names both skip kinds (fix item 13 follow-up) |
| C | harness, gate, y-swap script (items 1 code, 5 and 15 harness, 10, 11, 12) | e980f7c | fix(warmstart): harness reads the shared paths module and builds every loader with one BLAS thread per worker (fix items 15 and 5, harness part) |
| C | harness, gate, y-swap script (items 1 code, 5 and 15 harness, 10, 11, 12) | 215946c | feat(warmstart): cluster-robust (CR1, source image) SEs in aggregate_row and every gate comparison; the k rule keeps the iid SE (fix item 10) |
| C | harness, gate, y-swap script (items 1 code, 5 and 15 harness, 10, 11, 12) | 6002c26 | feat(warmstart): per-cube y-swap values and files (additive) and scripts/yswap_one_step.py, a CPU one-step y-swap at any t with paired contrasts (fix item 1 (a), (b)) |
| C | harness, gate, y-swap script (items 1 code, 5 and 15 harness, 10, 11, 12) | 85b0416 | feat(gate): flag ablations worse than the baseline, self-explaining INVALID records (verdict_if_valid, invalid_reason, banner) and require 2 noise seeds unless --allow_single_seed (fix items 11, 12) |
| D | driver and launcher (items 6 driver, 7, 14, STUCK amendment) | d10e544 | fix(warmstart): pilot driver emits eval_full and gate only for runs that are no longer alive (fix item 6, driver part) |
| D | driver and launcher (items 6 driver, 7, 14, STUCK amendment) | e6404f0 | fix(warmstart): pilot driver waits on a failed nvidia-smi (SMI-FAIL each pass) instead of exiting STUCK; rc 2 kept for GPUs held by foreign jobs (STUCK amendment) |
| D | driver and launcher (items 6 driver, 7, 14, STUCK amendment) | accd02d | fix(warmstart): pilot driver launches runs while a synchronous eval runs (60 s poll), BUSY heartbeat every 10 min, single-instance flock (fix item 7) |
| D | driver and launcher (items 6 driver, 7, 14, STUCK amendment) | 1c60541 | fix(warmstart): launch_run.sh lock + DONE/live-pid refusal; driver gate path from gate_1a.default_out_name; refuse a used results root; assert the group kill (fix item 14) |
| D | driver and launcher (items 6 driver, 7, 14, STUCK amendment) | db77eb3 | style(warmstart): pilot driver ROOT spacing and docstring wrap (fix item 14 follow-up) |
| A | documents (items 1 note, 2, 3, 4, 8, 9, 10 note, 11 note, 16) | the docs commit that adds this file | RESULTS note, plan as-built block, this ledger |

Post-review gate. The fix-wave gate was re-run on the three pilot evaluations and wrote the new file `results/residual_warmstart/eval/residual/d4/gate_1a__residual_d4_b64__postreview.json`. It reads INVALID with `verdict_if_valid` PASS and `invalid_reason` t100 0.99493, t300 0.98477, threshold 0.9, and it gives cluster-robust SEs for every comparison. The official `gate_1a__residual_d4_b64.json` was not rewritten: its sha256 is `75a4e286…` before and after.

Tests (CPU, `CUDA_VISIBLE_DEVICES= python -m pytest tests -q`): 142 before the wave, 167 after agent B, and 204 after agents C and D. The docs commit adds no tests.

Raised by the fix agents and left for the controller (not ruled):
- `gate_1a.check_done` reads only DONE. A run stopped by the wall-clock cap writes STOPPED and then DONE, so it would be gated as final at the epoch where it stopped; the unequal-epochs refusal is the only guard. Raised by agents B, C and D.
- `status.py` does not show STOPPED (agent D).
- Launches attempted from an evaluation tick retry every 60 s, so two transient refusals within about 2 min permanently cap a launch key (agent D; see 2.2).
- The push of worktree-phase1-plan is the controller's step after the re-review. Merging into master is the user's decision, and it must be a fast-forward or a merge commit so that 2fd2bee, which is recorded in all nine evaluation JSONs, stays reachable.

## 6. Final review and fix wave (2026-10-12)

progress.md now has 88 lines, sha256 `9bd27b98ec0115f9…`. Its lines from L82 (the line starting "Final review dispatched as workflow") to the end, verbatim; L82-L86 are also quoted in sections 1 and 5 above:
- **L82.** Final review dispatched as workflow wl46iv0mr (run wf_3a69d9f0-363): lenses t6-numbers, t6-science, t6-ops, fin-train, fin-harness-gate, fin-driver, fin-integration; confirm+refute per Critical/Important; one synthesis
- **L83.** Final review result (workflow wf_3a69d9f0-363, 32 agents): 12 Critical/Important findings, all CONFIRMED by both confirm and refute agents (0 disputed, 0 refuted), 27 minors; synthesis -> 16 fix items + 28 parked (saved at /home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/final_review_synthesis.json); full suite 142 passed on CPU. Key science finding: a post-hoc CPU y-swap at t=999 shows B's one-step gain is cube-specific (swap_full - true +0.2225 +- 0.0136 on 32 cubes) — INVALID comes from the 10 % threshold's lack of power at a 4.6 % gain, not from an unused condition; the sqrt(2) argument does not discriminate (a calibrated posterior sampler also pays sqrt(2) under RMSE).
- **L84.** Ruling (final-review fix wave): accept all 16 items as specified by the synthesis; resolve the five severity/scope splits as the synthesis recommends — Observation 1 rewrite important (item 2 wording); wall-clock semantics item 6 in full; as-built record as a tracked LEDGER.md (item 16); driver stopgap item 7 (launch while a synchronous eval runs + heartbeat + flock), async evals deferred to the driver generalisation; STUCK amendment (overturns part of the Task 5 STUCK ruling): gpu_status()=={} (nvidia-smi failure) means wait and log SMI-FAIL each pass, rc 2 kept for foreign jobs holding the GPUs. Post-review gate output goes to a NEW file gate_1a__residual_d4_b64__postreview.json; the official gate record is never overwritten. — why: every item is confirmed by two independent verifiers; the note must not misstate the science before the user decides — cost if wrong: a larger fix wave than strictly needed
- **L85.** Ruling (fix-wave execution): one workflow, four fix agents on disjoint file sets with explicit-path commits (git commit -F msg -- <paths>): B training/loaders/paths (items 5, 6-training, 13, 15) first; then C harness/gate/y-swap script (items 1, 10, 11, 12 code) in parallel with D driver (items 6-driver, 7, 14, STUCK amendment); then A docs (note items 1-4, 8-11, plan as-built + tracked LEDGER.md, item 16 minus the push); then one scoped re-review workflow stage (3 lenses + full suite). Controller pushes after the re-review. — why: the one-fix-agent rule exists to avoid per-finding context rebuilds; four scoped agents on disjoint files keep each context focused without that cost — cost if wrong: a merge-order slip between C and D (disjoint files make it unlikely)
- **L86.** Final-review fix wave dispatched as workflow we0ueswhd (run wf_adf5bccc-8d7): B -> (C || D) -> A -> re-review R1/R2/R3; FIX_BASE a9bb5c9
- **L87.** Fix wave result: B 5 commits (6ceff1f..dcbeb2b), C 4 (e980f7c..85b0416), D 5 (d10e544..db77eb3), A 1 (47faa87); re-review R1/R2/R3: all 16 items + the STUCK amendment ADDRESSED; full suite 204 passed, 0 skips; official gate file byte-identical (INVALID), postreview gate file reproduces; 5 new minors (3 note clarifications: floor 0.914 vs measured 0.913/0.910 at line 146, A-prime 3.29 pairing at 131/156, "added 4.3-6.1 pct2" reference at 141; driver imports gate_1a/numpy at module level; eval_tick counts every transient launch refusal toward MAX_ACTION_FAILURES)
- **L88.** Ruling (residual polish, deviates from the skill's "no second fix wave"): dispatch ONE small polish pass for the five residual minors + appending the post-A ledger rulings to the tracked LEDGER.md, followed by one scoped verification — why: the tick-count issue defeats the A->A-prime handover fix it was built for, and the note is the document the user decides on; all five are a few lines each — cost if wrong: ~30 min and one extra commit

Residual polish (the L88 ruling), all on worktree-phase1-plan with explicit paths. The documentation changes are in the docs commit that adds this section, so they cannot name its SHA:
- **Polish 1 (docs commit adding this section).** RESULTS.md Observation 2: the √(v/(2 − v)) ≈ 0.914 floor now says v comes from the 200-cube one-step gain (4.62 %, v 0.9098, floor 0.9135; ≈ 0.914 at v = 0.91). On the y-swap cubes B's t = 999 gain is larger: 6.27 % on the gate's 8 cubes (true 2.1409, baseline 2.2841; v 0.8786, floor 0.8851) and 7.09 % on the 32 (true 2.2479, baseline 2.4194; v 0.8632, floor 0.8714), so the measured 0.913 and 0.910 are consistent. Recomputed from the n8 and n32 y-swap files named in c_numbers.json.
- **Polish 2 (docs commit adding this section).** RESULTS.md Verdict and Observation 4: the DDPM-1000/DDIM-1 ratios say "(both rows on the 50 DDPM cubes)". A′ 26.8702 / 8.1552 = 3.2949; the table's ddim1 on 200 cubes would give 3.3008. A is paired the same way: 2.0832 (2.0771 with the 200-cube ddim1), 2.08 either way.
- **Polish 3 (docs commit adding this section).** RESULTS.md Observation 1: "The added energy over the one-step estimate, 4.3–6.1 pct² (mean obs² − ddim1², DDIM-10, DDIM-20, warm400_ddpm and DDPM-1000; 3.8–5.6 pct² over base²)". Recomputed from the epoch-100 eval JSON: over ddim1² 4.33 / 5.35 / 5.52 / 6.08, over base² 3.78 / 4.80 / 4.97 / 5.57 (mean of per-seed squares; the square of the seed mean differs by ≤ 0.006).
- **Polish 4 (1d7db4c).** The gate output name and its default thresholds are defined once, stdlib only, in `data_loader/warmstart_paths.py` (`gate_out_name`, `GATE_DEFAULT_MAX_NFE`, `GATE_DEFAULT_K`). `scripts/gate_1a.py` re-exports them as `default_out_name`, `DEFAULT_MAX_NFE` and `DEFAULT_K`, and `pilot_driver.py` imports `warmstart_paths` instead of `gate_1a`, so the driver and `status.py` import without numpy (verified under /usr/bin/python3, which has no numpy). Tests: the driver and status.py run in a subprocess with numpy and torch blocked; `gate_1a.default_out_name is warmstart_paths.gate_out_name`.
- **Polish 5 (1d7db4c).** `eval_tick` tries a given launch at most once per TICK_RETRY_S = 600 s from polls (module-level, across evaluations), and a failure there is logged TICK-REFUSED and not counted toward MAX_ACTION_FAILURES (`execute(..., count_failure=False)`); a pass-level launch is counted as before. This resolves the tick-retry item that section 5 lists as raised by agent D. Test: a refusing launch_run.sh stub over 15 polls gives 2 attempts 600 s apart, a failure count of 0, 2 TICK-REFUSED lines, and the next pass launches the run.
- **Polish 6 (docs commit adding this section).** Line references moved by polish 5: RESULTS.md Observation 5 cites pilot_driver.py:121 (was :116) for the A′ launch policy, and the plan's as-built block cites the driver docstring at lines 1-19 and the constants at lines 43-48 (were 1-16 and 40-44).
- **Polish 7 (docs commit adding this section).** This section.

Tests (CPU, `CUDA_VISIBLE_DEVICES= python -m pytest tests -q -rs`): 207 passed, 0 skips, after the polish code commit (204 + 3 new: the throttled tick launch, the numpy-free driver import, the gate name defined once).
