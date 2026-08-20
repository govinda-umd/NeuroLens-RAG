# Precursor Projects — Interview Prep

Two dated "precursor project" resume entries, each motivating one of the two main projects (BGM, NeuroLens-RAG) with a real, specific methodological lineage rather than a generic "prior experience" gesture. Same drafting process as BGM and NeuroLens-RAG: message-first, then bullets, house style (result-verb-first, sparing dashes, no defensive "rather than X" justification clauses, named technical terms where they're standard and accurate).

---

## MKL / tMKL work (2016-2019) — precursor to BGM

**Papers**:
- Surampudi, S. G., Naik, S., Bapi, R. S., Jirsa, V. K., Sharma, A. & Roy, D. (2018). "Multiple Kernel Learning Model for Relating Structural and Functional Connectivity in the Brain." *Scientific Reports*, 8(1), 3265. https://www.nature.com/articles/s41598-018-21456-0
- Surampudi, S. G., Misra, J., Deco, G., Bapi, R. S., Sharma, A. & Roy, D. (2019). "Resting state dynamics meets anatomical structure: Temporal multiple kernel learning (tMKL) model." *NeuroImage*, 184, 609-620.

**Why this motivates BGM**: both papers are generative latent-variable models of brain organization, six years before BGM. The tMKL paper's GMM-based state discovery (K chosen via BIC, not fixed a priori) is structurally the same *kind* of move as BGM's SBM-based community discovery (K chosen via Total Description Length) -- a real, six-year methodological throughline: discover latent structure unsupervised, select the number of latent groups via an information-theoretic criterion, validate reproducibility. Not a clean paradigm alternation -- both MS papers are hybrids (a supervised-fit mechanistic/generative core; tMKL adds a genuinely unsupervised discovery layer on top), and BGM's SBM is actually *more* purely unsupervised than either (no held-out prediction target at all). See `docs/interview-prep-bgm-thesis.md` and the session's earlier discussion for the full precision-checked comparison (GMM<->SBM, BIC<->TDL, train/test replication<->leave-one-out/bootstrap).

### Summary line
*Modeled how the brain's anatomical connectivity graph shapes its functional activity, and how that relationship evolves over time, bridging dynamical-systems theory with graph-based machine learning.*

### Bullets — FINALIZED

1. Reinterpreted an established dynamical-systems model of brain structure shaping function through the lens of graph signal processing, reformulated it as a multiple kernel learning problem, and fit it via LASSO-regularized optimization to learn which combination of kernels, each tuned to its own diffusion scale, best explains a subject's actual functional connectivity. Outperformed existing biophysical models at predicting held-out functional connectivity (71% correlation with empirical data, repeated-cross-validated).
2. Extended the model to capture how functional connectivity evolves over time: quantified the low-dimensional manifold underlying that temporal evolution, clustered it into a compact set of latent brain states, and linked each state back to the anatomical connectivity graph through the same kernel-based framework.
3. Validated the discovered states through repeated train-test splits (state correlations 0.86-0.98 across eleven splits), a perturbation test confirming predictions depended on real anatomical input rather than noise, and a self-consistency check comparing predicted state-specific connectivity against the model's own earlier unsupervised state assignments (87.68% agreement).
4. Validated the model on an independent cohort of 100 Human Connectome Project subjects, about four times larger than the original training set, without retraining, confirming it generalizes beyond the dataset it was built on.

### Precision notes (worth having verbally ready, not in the bullets)
- Bullet 1's chain is four real, distinct steps, not just "used LASSO": reinterpreted via graph signal processing -> reformulated as multiple kernel learning -> fit via sparse, LASSO-regularized optimization. Don't collapse these into one technique if asked to elaborate.
- Bullet 3's 87.68% is a **self-consistency check, not accuracy against external ground truth**: for a test subject, the model generates a predicted FC for each state *k* (the "implicit" label). Separately, that prediction's 25 nearest neighbors among *training* windowed-FC data are looked up, and the mode of *their* state labels (from the original unsupervised GMM clustering) gives an "estimated" label. The 87.68% is how often these two agree -- i.e., does the pattern generated *for* state k actually resemble training examples independently clustered into state k. There is no independently-verified ground truth here; "recall" would overclaim what's measured.
- Terminology is intentionally historically accurate (MKL, graph signal processing, LASSO) rather than dressed up in newer buzzwords -- these are still real, active, respected areas, not obsolete. Modernity lives in the summary line's framing (the *question*, not the technique names).

---

## GRU / PLOS work (2021) — precursor to NeuroLens-RAG

**Paper**: Misra, J.\*, Surampudi, S. G.\*, Venkatesh, M., Limbachia, C., Jaja, J. & Pessoa, L. (2021). "Learning brain dynamics for decoding and predicting individual differences." *PLOS Computational Biology*. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008943 (\*co-first authors)

**Why this motivates NeuroLens-RAG**: this paper already validates a decoder's reliance on plausible brain regions via gradient-based saliency cross-checked against an actual causal intervention (lesioning resting-state networks) -- the direct ancestor of NeuroLens-RAG's CAV/TCAV interpretability stack. The paper's own words: this cross-check "considerably assuaged" the "black box" worry, i.e. a human researcher validating a decoder against domain knowledge by hand. NeuroLens-RAG is what happens when that same instinct is pushed from region-level to *concept*-level validation, and automated against real retrieved literature instead of informal researcher judgment -- with the reliability of that automation itself measured and fixed (the RAG-CAV verification loop's deterministic-verdict design). Also directly relevant: the information-bottleneck-style representation finding here (task-relevant compression, poor raw-signal reconstruction) is the same phenomenon later found in NeuroLens-RAG's Case 2/Case 3 contrastive representations -- a real six-year-plus throughline, not a coincidence specific to one paper.

### Summary line
*Decoded which movie a person was watching purely from their brain activity, then verified computationally that performance depended on the temporal evolution of activity in biologically plausible brain regions, not spurious correlations.*

### Bullets — FINALIZED

1. Built a GRU-based decoder that classified which of 15 movie clips a person was watching from their brain activity alone, reaching about 90% accuracy at the region level. Confirmed the result depended on genuine temporal structure, not just more parameters, using a temporal-shuffling (wavestrapping) test -- scrambling the input's time order while keeping labels fixed dropped accuracy by more than 35 points.
2. Compressed the learned representation to about ten dimensions without meaningfully hurting accuracy, while the same code reconstructed less than ten percent of the original signal's variance -- evidence of an information bottleneck: the representation retained what was relevant to the task and discarded the rest, rather than compressing the raw signal losslessly.
3. Validated the model's gradient-based attribution map with a causal intervention, lesioning each of seven resting-state networks in turn: regions with high attribution caused significantly larger accuracy drops when ablated than regions chosen at random, confirming the model's decisions reflected more than post-hoc correlation.
4. Applied transfer learning by freezing the trained GRU encoder and training a new regressor head to predict individual differences (fluid intelligence, verbal IQ) directly from movie-watching brain activity, reaching statistical significance (p<0.005 for both) at accuracy comparable to established connectome-based prediction methods.

### Precision notes (worth having verbally ready, not in the bullets)
- Bullet 1 uses "temporal-shuffling (wavestrapping)" specifically, not "permutation testing" -- these are two different procedures in the paper. Wavestrapping shuffles the *input's* temporal order (tests whether order matters). Permutation testing (used elsewhere in the paper, not in this bullet) shuffles *labels* to build a chance-level null (8.40% baseline). Don't conflate them if asked to elaborate.
- Bullet 2's "information bottleneck" framing is a precise, standard term for the phenomenon observed, but **the original paper does not use this term itself** -- it's our own retrospective, accurate naming of what was found, not a claim the paper makes about itself. Fine to say if asked "did the paper call it that" -- "no, that's the standard name for the pattern it reports."
- Bullet 4's "transfer learning" is confirmed via the user's own firsthand recollection (frozen GRU encoder, new regressor head trained on top) -- the paper's text alone (as extracted) didn't explicitly confirm weight-freezing, so this relies on direct memory of the work, not just the published text.

---

## Status

Both precursor projects finalized (summary line + 4 bullets each). Combined with BGM and NeuroLens-RAG, this completes the four-project resume structure: MKL/tMKL (2016-2019) -> GRU/PLOS (2021) -> BGM thesis (2024-2026) and NeuroLens-RAG (2026) as two parallel, motivated extensions -- discriminative representation learning lineage (GRU -> NeuroLens-RAG) and generative/latent-variable modeling lineage (MKL/tMKL -> BGM). Remaining work: refine NeuroLens-RAG's own message points and bullets (left in a partially-resolved state -- see `docs/interview-prep-neurolens-rag.md`), then the full-resume pass across all four projects together.
