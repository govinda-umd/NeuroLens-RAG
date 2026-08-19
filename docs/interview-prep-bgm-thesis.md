# Bayesian Generative Modeling (BGM) Thesis — Interview Deep-Dive Prep

Living rehearsal document for "Bayesian Generative Modeling for Latent Network Structure" (resume project, 2024-2026), covering the PhD dissertation (`Desktop/THESIS/GovindaThesis.pdf`, 189 pages, "A Probabilistic Ontology of Functional Brain Organization"). Focus: Chapter 2, "Resting state mouse functional connectome" (pages 46-96 printed / PDF index 45-95), which corresponds to the submitted paper cited on the resume ("Bayesian generative modeling reveals a multi-modal hierarchical architecture in the mouse functional connectome," *Eur. J. Neuroscience*, submitted). Chapter 1 (Introduction, printed pages 13-44) is the general Bayesian/SBM/MCMC theoretical framework both this chapter and Chapter 3 (human threat/reward, a separate paper) build on.

**Why this doc exists**: the dissertation is written for a neuroimaging audience — methodological detail an ML audience would want explicit is often compressed into a sentence or pushed into a Supplement with no equations shown in the main text. This doc translates and makes that detail explicit, verified against the actual PDF text, not recalled from memory.

## The data, precisely

- **9 mice**, C57BL/6J background, mixed sex, 6-8 weeks old at first session. Male CRE mice specifically selected to avoid CRE-expression leakage (Ai162 genotype: tTA + TITL-GCaMP6s).
- **Simultaneous wide-field calcium (WF-Ca2+) imaging + BOLD-fMRI** — a dual-imaging dataset. This chapter uses only the fMRI resting-state runs; the calcium imaging data exists in the same dataset but isn't used here.
- Lightly **anesthetized** (isoflurane 0.5-0.75%, free breathing) — not awake resting-state. Real, explicitly stated scope limitation (see Limitations below).
- Scanner: 11.7T Bruker. GE-EPI, TR=1.0s, TE=9.1ms, 0.4mm isotropic resolution, 28 slices (near-whole-brain).
- 3 longitudinal sessions x 4 resting-state runs x 10 min = **108 total runs across 9 mice** (120 min/mouse). (3 additional runs/session/mouse involved unilateral LED stimulation, not used here.)
- Preprocessing: **RABIES** (Rodent Automated BOLD Improvement of EPI Sequences, v0.4.8) — named, specific open-source rodent-fMRI pipeline. Multi-step registration to an in-house template, itself pre-registered to **Allen CCFv3** atlas space. Motion correction (6-param), framewise-displacement scrubbing (0.075mm threshold), bandpass filter 0.008-0.2Hz, 30 timepoints trimmed per run edge, CSF/WM regression, 0.4mm-sigma smoothing.
- Network construction: per-run Pearson correlation -> subject-level -> group-level, via **timepoint-weighted averaging** (accounts for variable run length after scrubbing) with Fisher r-to-z transform. **172 ROIs** (Allen CCFv3 regions merged by functional role + anatomical proximity). Top 20% edges retained (proportional threshold; 10% as robustness check), binarized -> final undirected graph.
- **Bootstrap resampling, precisely**: 9 mice resampled with replacement, 500 independent times; full network-construction pipeline re-run per resample. **The entire downstream analysis pipeline (SBM fitting through community-level multiplicity characterization) executes independently 501 times** (1 reference sample + 500 resamples) — this is the real computational/statistical engineering scale, worth having as a number.

## The model — 7 generative hypotheses, not 1

Implemented via **`graph-tool`** (Tiago Peixoto's Python library). Three architectures x with/without degree-correction (6 variants) + the assortative SBM (a-SBM) as a 7th:
- **Standard SBM (s-SBM)**: unconstrained — can express assortative, core-periphery, or bipartite structure.
- **Hierarchical SBM (h-SBM)**: nested, multi-scale, recursively partitions to reveal a hierarchy.
- **Overlapping SBM (o-SBM)**: partitions *half-edges*, not nodes — a node's membership can genuinely mix across communities.
- **Degree-correction**: pits two neurobiological hypotheses against each other — is a region's "hubness" an intrinsic property separate from its community (degree-corrected), or is high connectivity a property of the community itself, e.g. a rich-club (non-degree-corrected)?

**The elegant methodological move, worth leading with**: modularity maximization (the field's standard heuristic) is mathematically *equivalent* to maximum-likelihood fitting of a-SBM. But modularity maximization always forces out a high-scoring assortative answer regardless of whether the data actually supports it — it has no way to say "no, this isn't real structure." a-SBM, being a real generative model with a real posterior, can. So this isn't "we tried a fancier method" — it's a genuine **stress-test of whether the field's standard heuristic detects real structure or overfits to noise**, and the a-SBM comparison is specifically built to check that.

## Model fitting and selection

- **MCMC**: `graph-tool`'s guided sampler — neighborhood-informed proposals (move a node to a community it's already connected to, not randomly) + macroscopic merge-split moves (merge/split whole communities to escape local minima). **5 independent chains x 100,000 steps per model.**
- **Convergence diagnostic with real teeth**: monitor each chain's sampled description-length (Sigma) distribution; check Kolmogorov-Smirnov distance between chains, threshold 0.2. **Models failing this convergence check are excluded from all downstream analysis entirely** — not force-fit anyway. Same "validate before trusting the output" instinct that runs through NeuroLens-RAG (probe accuracy as a CAV sanity check; the CI stopping rule) — a real cross-project methodological signature, not a coincidence, worth stating explicitly if asked to compare the two projects.
- **Model selection**: Total Description Length (TDL) = negative log model evidence, integrated over the *entire* posterior ensemble of partitions (not one best-fit point). MDL/Occam's razor. Lowest TDL wins. Crucially, TDL does **not** penalize a model for genuinely finding multiple valid modes — only for complexity that doesn't buy real explanatory power.

## Multiplicity-characterization pipeline (the real novel contribution)

Summarized at the right altitude, not every individual test:
1. **Label-switching correction** ("random label model") — un-shuffle arbitrarily-permuted community-index labels across MCMC samples by aligning to a canonical template via node-overlap.
2. **Cluster the aligned partition ensemble into discrete modes** (a mixture model) — each mode gets a posterior weight (omega_k) and a soft node-membership matrix (pi^(k)).
3. **Merge near-duplicate modes** via cosine similarity against a spatially-shuffled null baseline, FDR-corrected.
4. **Align modes across all 500 bootstrap resamples** back to the reference sample via optimal bipartite matching (linear-sum-assignment / Hungarian algorithm) — enables group-level statistical testing.
5. **Friedman test** on tracked mode prevalence across resamples (non-parametric, repeated-measures — required because per-resample omega values across modes sum to 1, so they're not independent).
6. **Per-ROI reliability testing**: expected node-membership per resample (omega-weighted average of pi^(k) across modes), then test each ROI's expected-membership distribution across 500 resamples against a null of zero affiliation, FDR-corrected -> "surviving" ROIs define a community's reliable spatial boundary.
7. **Recursively, within each individual community**: extract membership vectors across all modes x bootstraps, mask to surviving ROIs, cluster via **Gaussian Mixture Models** (component count chosen by BIC, not fixed a priori) -> reveals sub-patterns within a single community.
8. **Structurally stable vs. variable ROI classification**: Mann-Whitney U / Kruskal-Wallis H-test (non-parametric ANOVA) on each ROI's assignment-probability distribution across a community's distinct patterns, correcting for intra-resample dependency (median first), FDR-corrected.

## Results, with real numbers

- **Model comparison**: non-degree-corrected hierarchical SBM (nd-h-SBM) wins outright (lowest TDL). Purely assortative (a-SBM, = modularity-maximization equivalent) is worst. Non-degree-corrected beats degree-corrected across every architecture. Robust to 20% vs. 10% edge-density threshold.
- **Communities are real, not artifacts**: no spatial information given during inference, yet communities emerged anatomically contiguous and bilaterally symmetric, corroborated against axonal tracing and lesion-study literature (not just RSN comparison). Three hierarchy levels: ground (>20 compact communities), middle (9 communities, matches RSN spatial scale, the focus of analysis), upper (broad macro-systems, cortico-striatal forebrain vs. brainstem/hindbrain).
- **9 named middle-level communities**: Olfactory-Prefrontal-Thalamic (C.01), Somatomotor (C.02), Fronto-Insular-Striatal (C.03), Cingulo-Prefrontal (C.04), Visuo-Auditory (C.05), Medial-Temporal (C.06), Striato-Amygdalo-Hypothalamic (C.07), Cerebello-Brainstem (C.08), Ponto-Midbrain (C.09).
- **Refines canonical RSNs, quantified**: Somatomotor community = 95.4% specificity to canonical somatosensory RSN (effective RSN span ~1.2, essentially 1-to-1). Canonical limbic RSN fragments across 3 inferred communities + hindbrain (effective covering-community count ~5.0). Clean pattern: primary sensory stays cohesive, higher-order association cortex gets structurally refined/split.
- **Multi-modality of the whole landscape**: 4 distinct schemes in the reference sample, tracked across 500 resamples, Friedman chi^2(3)=3.05, p=0.55 -> no significant prevalence difference, genuinely co-dominant.
- **The control that makes this convincing, not a pipeline artifact**: running the identical MCMC + mixture-model pipeline on plain modularity maximization instead collapses to a single mode. Same pipeline, different generative assumption, different outcome -> rules out "the clustering step just likes finding multiple clusters."
- **Robustness check**: leave-one-out across all 9 mice recovers multi-modality in all 9 iterations -> not one outlier animal driving the result.
- **Sophisticated interpretive point**: TDL (MDL-based) actually *penalizes* posterior entropy -> structurally biased *toward* uni-modal models. The winning model came out multi-modal anyway, despite the selection criterion working against that outcome. And modularity maximization's uni-modal landscape isn't a virtue -- it's the worst-fitting model by TDL; uni-modality here signals an inability to express real structure, not a better fit.
- **Mechanism underlying multi-modality**: some communities structurally stable (single pattern, e.g. Cingulo-Prefrontal C.04); some split by hemisphere (e.g. Medial-Temporal C.06); some show coordinated bilateral variation (C.01, C.02, C.03). Concrete examples:
  - Somatomotor (C.02): 2 patterns (omega=53%/46%), differing in whether primary motor regions are strongly (0.9) or weakly (0.5) included -- read as "full sensorimotor integration" vs. "passive sensory monitoring."
  - Olfactory-Prefrontal-Thalamic (C.01): 3 patterns (54%/25%/21%), varying prefrontal-vs-thalamic coupling strength.
  - Fronto-Insular-Striatal (C.03): 3 patterns (36%/32%/31%) -- most ROIs in this community are structurally variable, acting as a "routing hub" coupling alternately with motor or prefrontal circuits.

## Honest limitations (stated in the thesis itself, not glossed over)

- Model selection is relative to the specific hypothesis space tested, not a ground-truth claim about the brain.
- The non-degree-corrected ("rich-club"-like) preference can't be cleanly separated from an artifact of the 20%-threshold graph construction plus BOLD's inherent degree homogeneity -- genuinely unresolved without further sensitivity analysis.
- **Anesthetized mice only** -- anesthesia globally depresses cortical activity; results characterize the anesthetized connectome specifically, not awake/behaving. A real, meaningful scope boundary, not a throwaway caveat.
- BOLD signal's spatial/temporal blurring (vascular architecture) affects exact community boundaries.
- Results depend on the chosen parcellation (Allen CCFv3) even though community *assignment* itself received no spatial prior.

## Real anecdote not in the thesis text, worth having for the interview

Graph-tool had built-in marginal/membership-matrix computation for standard and a-SBM, but **not for h-SBM** (at least at the time). User contacted Tiago Peixoto (graph-tool's author) directly via his Discourse forum; received a single-sentence hint, had to decode it and independently implement code to compute membership marginals across *every level* of the nested hierarchy. Real open-source engagement + independent algorithmic implementation of something genuinely nontrivial (aggregating/decomposing membership probabilities across nested hierarchy levels, not the flat single-level case graph-tool already handled) -- strong "depth and ownership" evidence a written methods section doesn't capture. Not documented anywhere in the thesis itself.

## Terminology: what field does this belong to?

Hierarchy, not competing labels:
- **Broadest correct label**: unsupervised learning (no labels; community structure inferred purely from network topology).
- **Specific formal approach within that**: generative modeling -- specifically a **latent variable model** (community assignment = the unobserved variable) applied to **relational/graph-structured data**, its own subfield (statistical network analysis / relational learning), distinct from ordinary unsupervised learning on i.i.d. feature vectors.
- **"Latent structure discovery"**: not a separate formal category -- it's the task-level description of what a latent variable model is being used for here (generative modeling = mechanism; latent structure discovery = the goal it serves).
- **"Unsupervised discovery"**: informal synonym for the above, not a standard distinct term.
- Murphy calibration: general formalism (latent variable models, Bayesian inference, MCMC, generative models, MDL/Bayesian model selection) very likely covered thoroughly. SBMs specifically are a more specialized network-science topic -- don't overclaim a dedicated section without checking.

## Publication status — important correction, verify before writing any resume bullet

**Chapter 3's Bayesian generative modeling / SBM analysis is UNPUBLISHED, not the 2023 J. Neuroscience paper.** The thesis text itself distinguishes them explicitly: "In our previous work, we characterized the temporal dynamics of these affective states... its network analysis relied on **descriptive clustering**... To build upon this, **the present study** applies a Bayesian generative modeling approach." The 2023 paper (Murty, Song, Surampudi & Pessoa, *J. Neuroscience*, already on the resume's publication list) is that *earlier*, descriptive-clustering analysis of the same threat/reward dataset — a different method, already published. Chapter 3's SBM/generative-modeling analysis is a new, unpublished extension of that dataset. **Only the mouse connectome work (Chapter 2) is submitted (to *Eur. J. Neuroscience*).** Any resume bullet describing Chapter 3's specific SBM findings must not be attributed to the 2023 citation or implied to be published — different analysis, different status.

## Chapter 3: human threat/reward imminence processing — scope question RESOLVED

**Resolved: one resume project entry, not two.** The core methodological contribution (Bayesian generative model comparison + full-posterior multi-modal-landscape characterization, replacing single-consensus assumptions) is genuinely one body of work, demonstrated twice — across species (mouse -> human) *and* across data regimes (resting-state -> task-evoked). That's a stronger claim as one entry ("developed a framework, validated across two species and two data regimes") than as two thinner separate entries. The per-subject/multi-condition extension gets its own bullet within that one entry -- real added technical depth, not just re-running the same script on new data.

**What's identical to Chapter 2**: SBM variants, degree-correction as competing hub hypotheses, `graph-tool`/MCMC/TDL model selection. **The winning architecture replicates**: non-degree-corrected hierarchical SBM wins, purely assortative (modularity-maximization-equivalent) loses -- same result, humans and mice, resting-state and task-evoked. A real cross-species, cross-paradigm consistency finding, not just re-running code.

**What's genuinely different (the "slight changes"), precisely:**
- **Data**: 80 human subjects, active threat-avoidance/reward-seeking task (virtual predator/coin descending on screen, participant controls an avatar), ~66-68% behavioral success rate across conditions (confirms task engagement). 2x2 design: valence (threat/reward) x arousal (high/low) = 4 conditions. Connectivity built from a tightly-defined "imminence window" (-3.75s to +1.25s around trial culmination), not continuous resting-state. 100-region parcellation. 70/80 subjects had complete SBM fits across all conditions -- used for main analyses (280 graphs = 70 subjects x 4 conditions).
- **Model-fitting unit changed structurally**: mouse paper = one group-level graph + bootstrap resampling of subjects for uncertainty. Human paper = SBM fit **independently per individual subject** -- subjects as the natural population-sampling unit directly, not resampled group averages. A genuine methodological upgrade, not cosmetic. Shows up in statistical power: Friedman chi^2(4) ~ 263-273, p<<0.05 across all 4 conditions (vs. the mouse paper's more modest bootstrap-based test) -- real independent subjects give much more power than resampled versions of 9 animals.
- **5 model variants, not 7**: overlapping SBM dropped, explicitly for "excessive computational demands" on human data -- an honest, stated engineering tradeoff.
- **New analytical layer with no Chapter 2 analog**: with 4 conditions instead of 1, this chapter asks a question Chapter 2 structurally couldn't -- is the same repertoire of organizational schemes conserved *across* conditions, and do task demands reweight which scheme dominates? Requires new machinery: a vector-displacement permutation test, cross-condition scheme alignment (TOC pages 136-141) -- not yet read in detail, headline "reweighting" result not yet extracted.

## Still to cover

- [ ] Chapter 3 Results detail: the conserved-repertoire and task-demand-reweighting findings (headline result of this chapter, not yet read with real numbers) -- TOC pages 110-121.
- [ ] Chapter 3 Methods detail: vector displacement permutation test, cross-condition alignment methodology (pages 136-141) -- the genuinely new machinery vs. Chapter 2.
- [ ] Chapter 1 (Introduction, pages 13-44): the general Bayesian/SBM/MCMC/MDL theoretical framework, likely useful for precise formal definitions if asked to derive something from first principles.
- [ ] Supplementary Sections (pages 149-175): the actual mathematical formulations (SBM likelihood, description length derivation, MCMC acceptance criteria) -- read if the interview format seems likely to probe equation-level detail.
- [ ] Resume points for the BGM project (draft once enough material gathered, same process as NeuroLens-RAG -- likely ready now given both chapters' core findings are covered).
