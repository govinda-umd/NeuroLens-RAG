# HCP Dataset Extension Options

> Candidate second (or further) datasets to stress-test the findings in `docs/end-to-end-report.md` §7 — specifically the hypothesis that the cross-representation TCAV and attribution convergence is a property of the MOTOR task's clean, temporally well-separated block design, not of the three representation-learning paradigms themselves. Compiled 2026-08-27, verified against the current HCP data-release documentation (humanconnectome.org), not recalled from memory alone.

## Tier 1 — nearly free: other HCP-YA task-fMRI runs, same subjects already on disk

The same HCP Young Adult release MOTOR comes from collected 6 other task paradigms per subject: **Working Memory** (N-back), **Language** (story/math), **Social Cognition** (theory-of-mind), **Relational**, **Emotion**, **Gambling**. Same subjects, same preprocessing pipeline, same `data_setup.py` loaders — only the task/run string changes, no new data engineering.

**Why it matters, not just why it's cheap**: tests whether the effector/laterality-style convergence found on MOTOR generalizes to entirely different concept types (working-memory load, semantic vs. arithmetic processing, social inference) or is specific to motor cortex's unusually clean topography. This is the most direct test of the *representation-learning-objective* claim, decoupled from the *task-design* claim that Tier 2 tests.

**Cost**: lowest of all options. No new subject recruitment, no new access agreement, no new preprocessing pipeline.

## Tier 2 — already reserved: HCP 7T Movie-Watching

Continuous, naturalistic viewing, no discrete condition blocks — the direct stress test of the block-design hypothesis itself. Already scaffolded: `data/raw/hcp_movie_watching/README.md`.

**Why it matters**: if the convergence findings weaken on continuous data, that confirms the block-structure explanation in §7 of the end-to-end report. If they hold up, the finding is more general than MOTOR alone can show.

**Cost**: moderate — 7T movie-watching is a subset of subjects (not all HCP-YA participants have it), and continuous data needs a different windowing/labeling scheme than discrete-block MOTOR (no clean `y` class labels; would need annotation-derived or embedding-derived pseudo-labels).

## Tier 3 — a different domain with equally strong ground truth: HCP 7T Retinotopy

Visual cortex has an even more precisely mapped, textbook topographic organization (polar angle / eccentricity maps) than the motor homunculus.

**Why it matters**: a second, independent cortical system with its own well-established ground truth. Tests whether the cross-representation attribution-convergence finding is a general property of well-organized cortex, or a motor-cortex-specific artifact of this project's Schaefer-300/somatomotor-network setup.

**Cost**: moderate-high — different task design (visual stimulation, not movement), different concept definitions (visual field position, not effector/laterality), likely a smaller subject subset (7T data is not universal across HCP-YA).

## Tier 4 — a different scientific question, bigger lift: HCP-Development or HCP-Aging

- **HCP-Development (HCP-D)**: ages 5–21, 652+ subjects (Release 2.0).
- **HCP-Aging (HCP-A)**: ages 36–100+, 725 subjects.

**Why it matters**: a genuinely different question from Tiers 1–3 — does the representation-learning-objective convergence found on healthy young adults hold across development or aging, or does motor cortex reorganization with age break it? Real neuroscience literature exists on both pediatric motor system maturation and age-related motor cortex reorganization, so a divergent finding here would be independently interesting, not just a null result.

**Cost**: highest of the fMRI options — different subject pool entirely (can't reuse the existing 100-subject splits or bootstrap checkpoints), different access mechanism, likely different preprocessing pipeline versions across HCP Lifespan releases.

## Tier 5 — different modality, already on the roadmap: HCP structural connectivity / diffusion data

Not a new fMRI task at all — a graph-structured input (the structural connectome) for a future GNN extension. ~958 of the current 1,013 candidate subjects already have preprocessed diffusion/DTI data (`docs/case2-3-design-plan.md`'s parked SC-graph idea) — subject availability was already checked and is not a blocker.

**Why it matters**: a structurally different extension axis (input modality, not task or population) — feeds a GNN-based Case 4, not a stress test of Cases 1–3.

**Cost**: highest engineering lift of all options — requires a new model architecture (GNN front-end), not just new data through the existing pipeline.

## Recommendation

Sequence by ratio of insight to effort: **Tier 1 before Tier 2 or 3, not instead of them.** Tier 1 reuses everything already built with zero new data engineering and directly tests the representation-learning-objective claim. Movie-watching (Tier 2, already reserved) and retinotopy (Tier 3) are the two most scientifically interesting *stress tests* of the block-design hypothesis specifically, and are worth doing once Tier 1 establishes whether the finding generalizes across concept types at all. Tiers 4 and 5 are real, worthwhile future directions but are scoped as separate projects, not quick extensions.
