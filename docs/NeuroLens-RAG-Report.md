# NeuroLens-RAG: Literature-Grounded Mechanistic Validation of Multimodal Neural Decoders

*A representation-learning and interpretability framework for human fMRI decoding, validated against retrieved neuroscience literature.*

---

## Abstract

Decoding cognitive and motor states from human neuroimaging is conventionally evaluated by classification accuracy alone — a criterion that cannot distinguish a model that has learned neurobiologically meaningful structure from one that exploits an incidental statistical shortcut correlated with the label. This gap is particularly acute for representation-learning approaches, whose internal features are not directly interpretable in domain terms. We present NeuroLens-RAG, a framework for decoding movement conditions from human fMRI (HCP Young Adult, MOTOR task, *n* = 100 subjects, 6 conditions) that pairs two complementary representation-learning objectives with a mechanistic-interpretability and literature-grounding pipeline designed to *test*, rather than assume, whether a learned representation aligns with independent domain knowledge. The first objective is supervised multi-task decoding of movement class and hemodynamic response; the second is a contrastive objective that aligns a brain encoder with a text encoder of the six condition descriptions in a shared embedding space, without any classification head. Both were implemented with matched recurrent (GRU) and Transformer backbones and evaluated with population-level statistics — repeated-split resampling and paired non-parametric tests — modeled on established neuroimaging methodology. We then probed what each architecture represents using four complementary attribution methods and Concept Activation Vectors (CAV/TCAV), and closed the loop with a retrieval-augmented generation (RAG) system that retrieves relevant neuroscience literature and tests literature-derived claims directly against each model's internal geometry. The Transformer architecture outperformed the recurrent architecture under both objectives (paired Wilcoxon *p* < 10⁻⁸), with the margin roughly doubling under the contrastive objective. Concept-level analysis showed that effector identity (hand, foot, tongue) is linearly represented with near-ceiling separability in both models, while body-side laterality is represented far more weakly and inconsistently — a result independently reproduced by two unrelated methods for deriving concept directions, one from labeled brain examples and one from unsupervised arithmetic on text embeddings alone. The literature-grounding pipeline surfaced both convergent evidence (an independently retrieved account of contralateral hand representation matching the models' own concept sensitivity) and informative divergences, motivating a verification procedure that separates a deterministic evidentiary judgment from its natural-language narration. We discuss the generality of this concept-verification framework beyond motor decoding and outline planned extensions, including a three-way brain–text–physiological contrastive objective, a generative text-to-brain decoder, and small-model distillation for efficient deployment.

---

## 1. Introduction

### 1.1 The interpretability gap in neural decoding

Deep networks now decode a wide range of cognitive and motor states from human neuroimaging with high accuracy. Accuracy, however, is a weak witness for representational validity. A classifier that separates "left hand" from "right hand" movement at 95% accuracy has demonstrated that *some* function of its input separates the two classes — not that the function it learned corresponds to the neuroanatomical distinction (contralateral motor cortex organization) a domain expert would invoke to explain the same separation. The two claims are logically independent, and the difference matters: it determines whether a decoder generalizes to related conditions never seen during training, and whether its internal representation can be trusted as a proxy for anything about brain organization, rather than as an opaque but effective correlate of the label.

Closing this gap requires evidence beyond held-out accuracy. Two lines of evidence are useful in combination. The first is *mechanistic interpretability*: does the model's representation organize itself along axes a domain expert would recognize (e.g., effector identity, body-side laterality), and is the model's decision demonstrably sensitive to those axes, rather than merely correlated with them? The second is *independent domain grounding*: does that organization agree with what is already established in the literature, obtained without reference to the model at all? Interpretability alone can confirm a model uses a pre-specified concept; it cannot supply concepts a human did not think to ask about. Literature retrieval alone can supply such concepts; it cannot, by itself, verify that a specific model's representation actually depends on them. The present work integrates both lines of evidence into a single pipeline and evaluates each component on real data.

### 1.2 Approach and scope

NeuroLens-RAG decodes six MOTOR-task conditions (baseline, left/right hand, left/right foot, tongue) from Schaefer-300 parcellated, Yeo-7-network-labeled fMRI time series, using two decoding paradigms trained on a common 100-subject cohort with disjoint train/validation/test/hyperparameter-selection splits at the subject level:

- **Supervised multi-task decoding** — a sequence encoder predicts the movement class and, as an auxiliary objective, the instantaneous hemodynamic response, from a windowed input.
- **Contrastive brain–language representation learning** — a brain encoder and a text encoder of the six condition descriptions are trained jointly to align in a shared embedding space, in the manner of CLIP (Radford et al., 2021), adapted to a small closed vocabulary rather than an open caption set.

Both paradigms were implemented with matched GRU and Transformer backbones, enabling a controlled architecture comparison under each objective. On top of the trained models, we built:

- a four-method attribution suite (two gradient-based, two perturbation-based) identifying which resting-state network drives a given decode;
- a Concept Activation Vector (CAV/TCAV; Kim et al., 2018) procedure that tests whether a model's decision is causally sensitive to a human-specified concept direction in its own representation;
- a retrieval-augmented generation system over a curated corpus of neuroscience papers, coupled to a local instruction-tuned language model; and
- a verification loop in which literature-derived claims, extracted automatically from retrieved text, are converted into concept directions and tested against each model's representation — turning retrieval from a citation lookup into a falsifiable audit of the model.

Every comparative claim in this report is accompanied by population-level statistics (repeated-split resampling, paired non-parametric tests, bootstrap confidence intervals), following the methodological standard set by Misra & Pessoa (2025, *eLife*).

![Figure 1: System overview](figures/fig1_system_overview.png)

**Figure 1.** Overview of the NeuroLens-RAG pipeline. Two representation-learning objectives (left) share the same fMRI input; each feeds a mechanistic-interpretability stage, which in turn seeds and is tested against literature retrieved by the grounding stage (right). The verification step combines a deterministic evidentiary judgment with a language-model narration of it (§2.6).

![Figure 1b: Worked example](figures/fig1b_worked_example.png)

**Figure 1b.** Figure 1, elaborated: the same nine-stage pipeline traced through one real decoded window (subject 102311, left-hand movement), with the method used at each stage on the left and the actual intermediate values produced at each stage on the right. Case 2 shares stages 3–9 conceptually but reaches step 7 by its own text-derived route rather than Case 1's labeled-example probe (§2.4).

---

## 2. Methods

### 2.1 Data

Data were drawn from the Human Connectome Project Young Adult release (S1200), MOTOR task, both phase-encoding directions (LR, RL), for 100 subjects. BOLD time series were parcellated into 300 cortical regions of interest (Schaefer et al., 2018) and labeled by their assignment to one of seven canonical resting-state networks (Yeo et al., 2011): Visual, Somatomotor, Dorsal Attention, Salience/Ventral Attention, Limbic, Control, and Default. Time series were motion-regressed, detrended, and z-scored within run. Subjects were partitioned at the subject level (never at the window level, to prevent leakage across overlapping windows) into 65 training, 13 validation, 12 test, and 10 hyperparameter-selection subjects, the last held out entirely from all reported results.

Two supervisory signals were derived per timepoint: a six-way categorical label (baseline plus five movement conditions) and a five-channel continuous regressor obtained by convolving the task's event timeline with the canonical hemodynamic response function.

### 2.2 Supervised multi-task decoding

Causal windows of 32 TRs (stride 2) were mapped to the class label and hemodynamic regressor at the final timepoint of the window. Both a GRU (hidden size 128) and a Transformer (128-dimensional model, 4 attention heads, 2 layers, sinusoidal positional encoding) encoder were trained with a joint objective, $\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda \, \mathcal{L}_{\text{MSE}}(\hat{y}_{\text{HRF}}, y_{\text{HRF}})$ with $\lambda = 0.1$, optimized with AdamW and cosine-annealed learning rate.

### 2.3 Contrastive brain–language representation learning

A second decoding paradigm dispenses with a classification head entirely. A brain encoder (the same GRU/Transformer backbones, with the pooled representation projected to a 64-dimensional embedding) and a text encoder (a frozen pretrained sentence embedding model, MiniLM, with a small trainable linear projection into the same 64-dimensional space) are trained jointly against a temperature-scaled cosine-similarity objective, $\mathcal{L} = \mathcal{L}_{\text{CE}}(\tau \, z_{\text{brain}} z_{\text{text}}^\top, y)$, where $z_{\text{text}}$ is the matrix of the six condition prototypes' embeddings. Because the text side is a small, fixed vocabulary rather than an open per-example caption set, this is a supervised-contrastive, prototype-alignment objective rather than literal in-batch-negative InfoNCE: each brain window is pulled toward its own class's fixed text prototype and pushed away from the other five, with no direct brain-to-brain comparison term.

The trained model was evaluated by (i) prototype-based classification (nearest text prototype by cosine similarity), (ii) cross-modal retrieval (given a condition's text prototype, what fraction of the top-*k* most similar brain windows share that condition), and (iii) a frozen-backbone linear probe forecasting future regional activity from the pooled representation, with a leak-safety constraint requiring the forecast target to remain within the same condition block as the source window.

### 2.4 Mechanistic interpretability

**Attribution.** Four methods identified which of the seven resting-state networks drove a given decode: Saliency and Integrated Gradients (gradient-based, computed on the continuous 300-ROI input and aggregated by network), and exact Shapley values and LIME (perturbation-based, computed directly on a seven-"player" network-ablation abstraction — small enough that all 2⁷ = 128 coalitions can be enumerated exactly for Shapley, without sampling approximation).

**Concept Activation Vectors.** Following Kim et al. (2018), a concept (e.g., *hand*, *left-side*) is defined by a set of positive and negative example inputs. A linear probe fit on the model's pooled hidden representation separating these examples yields a Concept Activation Vector — the direction along which "concept-ness" increases fastest in the model's own representational geometry. The TCAV score is the fraction of held-out examples for which the directional derivative of a target class's output along this direction is positive: an empirical measure of how often nudging the representation toward the concept would increase the model's confidence in that class, computed without altering the model itself.

For the supervised decoder, concept directions were fit by logistic regression on labeled brain examples. For the contrastive model, which has no classification head, we additionally developed a second, independent method: a concept direction can be derived purely from the difference between two text-prototype embeddings in the model's shared embedding space (e.g., $z_{\text{text}}(\text{left}) - z_{\text{text}}(\text{right})$), then pulled back into the brain encoder's hidden space via the transpose of the trained linear projection — an adjoint-map construction requiring no labeled brain examples whatsoever. Because the contrastive model's shared embedding space also accepts arbitrary free text, we extended attribution itself to the same setting: the similarity between a window's brain embedding and the embedding of *any* sentence (not only the six trained conditions) can be attributed to resting-state networks by exactly the same four methods, substituting the similarity score for a classifier logit.

### 2.5 Retrieval-augmented literature grounding

A corpus of eight papers spanning task-fMRI methodology, cortical parcellation, and body-part-specific motor representation (including Ehrsson et al., 2003, and Meier et al., 2008) was chunked, embedded with a sentence-transformer, and indexed for dense retrieval, with an optional cross-encoder reranking stage. A locally hosted, quantized instruction-tuned language model (Llama-3.2-3B) performs three roles per decoded window: labeling each retrieved excerpt's stance toward the decode (supporting, contradicting, or unrelated), extracting a specific testable claim from a relevant excerpt (e.g., "hand movement is contralateral"), and producing a final narrative synthesis.

### 2.6 The concept-verification loop

The central methodological contribution of this work is closing the loop between retrieval and interpretability: a claim extracted from literature is mapped, by keyword match, onto one of the concepts the CAV machinery can test, a concept direction is fit or derived as in §2.4, and the resulting TCAV score is compared against the excerpt's stance. Critically, the *evidentiary verdict* — whether the literature and the model's representation agree, disagree, or bear no relation to each other — is computed deterministically from the excerpt's stance and the measured TCAV score, rather than left to the language model's free judgment. An unrelated excerpt yields no comparison (neither agreement nor disagreement is defined); a supporting excerpt paired with high concept sensitivity yields agreement; a supporting excerpt paired with low concept sensitivity yields a genuine, flagged disagreement. The language model's role is restricted to narrating this already-determined verdict in readable prose, integrating it with the decoded result and the retrieved text — a separation of a falsifiable, reproducible judgment from a fluent but non-authoritative narration of it.

### 2.7 Statistical framework

Every comparative claim is backed by population-level resampling rather than a single train/test split, following Misra & Pessoa (2025, *eLife*): repeated random re-partitioning of the subject pool (20–30 repeats, an empirical stopping rule requiring the running confidence interval to stabilize), paired designs when comparing two architectures on identical resampled splits (enabling a Wilcoxon signed-rank test on per-repeat differences), and post-hoc bootstrap resampling of the test-subject composition for a fixed trained model, reported alongside the repeated-split estimate.

---

## 3. Results

### 3.1 Decoding performance and architecture comparison

Both decoding paradigms reached high performance on the held-out test set: 92.0% test macro-F1 (95% CI [90.9, 93.4], 20 repeated splits) for supervised decoding, and 91.8% (95% CI [88.1, 94.0], 20 repeated splits) for the contrastive paradigm's prototype classification. A controlled comparison of the two backbones under both objectives (**Figure 2**) showed the Transformer architecture significantly outperforming the recurrent architecture in both cases (paired Wilcoxon *p* < 4 × 10⁻⁹ for both), with a substantially larger margin under the contrastive objective (+4.1 percentage points) than under direct supervised decoding (+2.0 percentage points) — the contrastive training signal appears to reward the Transformer's inductive bias more than plain classification does.

![Figure 2: Architecture comparison](figures/fig2_architecture_comparison.png)

**Figure 2.** Test macro-F1 for GRU and Transformer backbones under both decoding objectives, with 95% confidence intervals from 20 repeated subject-level resamples. Asterisks denote a paired Wilcoxon signed-rank test on per-repeat differences (*** *p* < 0.001).

### 3.2 Structure of the contrastive representation

Because retrieval-based evaluation can be perfect even when a joint embedding space merely ranks correctly without genuinely intermixing modalities (a "modality gap," Liang et al., 2022), we evaluated the contrastive model's shared space directly rather than relying on classification accuracy alone. A two-dimensional projection of brain-window embeddings and the six text prototypes (**Figure 3**) shows each text prototype situated within its corresponding class's brain-embedding cluster, rather than in a separate region of the space. This qualitative impression was confirmed quantitatively: the ratio of mean cross-modal to mean within-modality embedding distance was 1.01 (no systematic offset between modalities), and a silhouette score computed on brain embeddings alone, with text entirely excluded, was 0.43 — evidence that the brain encoder organizes windows by condition on its own, rather than merely inheriting apparent structure from proximity to fixed text anchors.

![Figure 3: Joint embedding space](figures/case2_embedding_space_pca.png){width=70%}

**Figure 3.** Principal-component projection of the shared brain–text embedding space. Dots are individual brain-window embeddings (colored by true condition); stars are the six condition text prototypes.

Text-to-brain retrieval quality, summarized by R-precision — the point at which precision, recall, and F1 coincide, avoiding the systematic underestimate of recall that a fixed small *k* produces against a much larger true-class pool — ranged from 0.83 to 0.89 across conditions (**Figure 4**), consistent with the model's classification-level accuracy.

![Figure 4: Retrieval quality](figures/fig4_retrieval_quality.png){width=65%}

**Figure 4.** Text-to-brain retrieval R-precision by condition.

### 3.3 Concept sensitivity of the learned representations

Concept Activation Vector analysis was performed on both decoding paradigms for five concepts: effector identity (*hand*, *foot*, *tongue*) and body-side laterality (*left-side*, *right-side*). For the supervised decoder, concepts were fit from labeled brain examples via logistic regression on the pooled representation (probe accuracy 98.5–99.9%). For the contrastive model, the same five concepts were instead derived purely from differences between text-prototype embeddings, with no labeled brain examples used at any stage (**Figure 5**).

![Figure 5: Concept sensitivity heatmap](figures/fig3_cav_heatmap.png){width=75%}

**Figure 5.** TCAV sensitivity of each decoded class to five concept directions, derived entirely from text-prototype arithmetic (contrastive model). Values are the fraction of held-out examples of the given class for which the model's decision is directionally sensitive to the concept.

Effector identity was represented with near-ceiling linear separability by both methods and both architectures: the *hand* concept showed TCAV sensitivity of 0.98–1.00 for left- and right-hand decodes and 0.11–0.20 for all other classes; *foot* and *tongue* showed an analogous pattern. Body-side laterality was represented far more weakly and less cleanly: TCAV sensitivity for the matching-laterality class ranged only 0.44–0.74, with substantial off-target sensitivity for classes that have no laterality (baseline, tongue) that are not part of either the positive or negative set used to define the concept. This asymmetry — clean, near-binary encoding of *which effector*, diffuse encoding of *which side* — was obtained independently via two methods that share no data or fitting procedure (a supervised probe on brain examples for the multi-task decoder; unsupervised text arithmetic for the contrastive model), which is stronger evidence that it reflects a genuine property of what these models learn from this task and cohort size, rather than an artifact particular to either derivation method.

### 3.4 Literature-grounded concept verification

The verification loop (§2.6) was run on real decoded windows, with retrieved excerpts from the literature corpus converted into concept claims and tested against each model's own representation. In one representative case, a window decoded as left-hand movement showed TCAV sensitivity of 1.0 to both the *hand* and *left-side* concepts, converging with an independently retrieved account of contralateral hand representation (Ehrsson et al., 2003) — two lines of evidence, neither informed by the other, arriving at the same conclusion. In other cases, the loop correctly identified that a retrieved excerpt's claim was unrelated to the concept actually driving a given decode, yielding a null result that the verification procedure records as evidentially uninformative rather than as a disagreement — a distinction that matters because the two have different implications (an uninformative test says nothing about the model; a genuine disagreement flags a specific, checkable discrepancy between the model's representation and a documented claim).

### 3.5 Component-level evaluation of the generation pipeline

Three components of the retrieval-and-generation pipeline were evaluated and, where beneficial, improved. First, retrieval quality was measured on a held-out set of paraphrased queries requiring retrieval of one specific passage from an 880-chunk corpus containing many near-duplicate, topically overlapping chunks — a substantially harder benchmark than paper-level relevance, which saturates at this corpus size. Domain-adaptive contrastive fine-tuning of the retrieval embedding model on a small set of in-domain query–passage pairs improved chunk-retrieval accuracy from 43.9% to 61.0% (top-1) and from 75.6% to 85.4% (top-3) (**Figure 6**). Second, a query-refinement procedure that steers a second retrieval pass toward whichever literature-derived concept the model is measurably most sensitive to increased the retrieved passage's estimated relevance score consistently, though the small (eight-paper) corpus size meant the top-ranked passage itself did not change — an expected ceiling effect at this corpus scale rather than a failure of the technique. Third, supervised fine-tuning of the local generation model on a small set of examples targeting a specific reasoning distinction (whether an unrelated excerpt constitutes a genuine disagreement) improved the model's reliability at producing a required output format, but did not reliably transfer the underlying discrimination itself at this training scale — motivating the deterministic verification design of §2.6, which does not depend on this discrimination being learned by the generation model at all.

![Figure 6: Embedding fine-tuning](figures/fig5_embedding_finetune.png){width=65%}

**Figure 6.** Chunk-level retrieval accuracy before and after domain-adaptive contrastive fine-tuning of the retrieval embedding model.

---

## 4. Discussion

### 4.1 A general framework for validating learned neural representations

The central claim of this work is methodological: accuracy-only evaluation and interpretability-only evaluation are each individually insufficient for establishing that a decoding model's representation is neurobiologically meaningful, and retrieval-augmented literature grounding closes a specific part of the remaining gap — supplying candidate concepts a researcher did not have to enumerate in advance, drawn from the accumulated findings of the field rather than from the model itself. The verification loop developed here (§2.6) demonstrates that this integration is tractable with entirely open, moderately sized models and a modest literature corpus, and that its outputs are falsifiable: a retrieved claim either does or does not survive contact with the model's measured internal sensitivity, and the two possible outcomes are distinguishable from a third, uninformative outcome (claim and model concept simply do not overlap).

### 4.2 The effector/laterality asymmetry as a substantive finding

The consistent, cross-method finding that effector identity is represented far more cleanly than body-side laterality is, we believe, the most substantive empirical result in this report. It was obtained under two decoding objectives, two architectures, and — for the contrastive model — two entirely independent ways of deriving a concept direction (supervised probing versus unsupervised prototype arithmetic), none of which shared a fitting procedure. This convergence argues against the asymmetry being a probing artifact specific to any one method, and raises a genuine question for further investigation: whether it reflects a property of the training data and task design at this subject count, or a more general property of how spatial-laterality information is distributed across the sampled cortical parcellation relative to effector-identity information.

### 4.3 Limitations

The attribution methods used here disagree with each other more often across families (gradient- versus perturbation-based) than within a family, and this disagreement remains unresolved rather than adjudicated. The literature corpus is small (eight papers); both retrieval quality and the diversity of concepts available for verification would be expected to improve with a larger, more systematically curated corpus. The mapping from an extracted literature phrase to a specific testable concept remains keyword-based rather than semantically matched, a simplification that constrains which literature claims can currently be operationalized as tests. Text-conditioned attribution, while functional, did not sharply discriminate between two substantively different query texts evaluated on the same window in our qualitative check, suggesting it is better suited to broad network-level questions than fine discrimination between closely related candidate concepts at present.

### 4.4 Future directions

Three extensions are planned. First, a three-way contrastive objective incorporating the continuous hemodynamic response as a third aligned view alongside brain and text embeddings, motivated by the graded, continuous structure of the hemodynamic signal being potentially more informative for short-horizon forecasting than a discrete condition label; this requires care to avoid leaking information about future event timing into what should be a causal forecasting setup. Second, a generative decoder mapping a point in the shared embedding space — potentially a concept-nudged point, rather than only a condition prototype — back to a synthesized pattern of regional activity, which would allow a concept's predicted neural signature to be inspected directly rather than only summarized by a scalar sensitivity score. Third, knowledge distillation of the supervised decoder into a substantially smaller student model, isolating what a soft-label training signal contributes beyond what the same small architecture achieves on hard labels alone, motivated by eventual real-time or resource-constrained deployment.

---

## 5. Conclusion

NeuroLens-RAG demonstrates that mechanistic interpretability and literature-grounded verification can be combined into a single, reproducible pipeline for auditing what a neural decoding model has actually learned, applied here to two complementary representation-learning objectives on human motor-task fMRI. Beyond establishing that a Transformer architecture significantly outperforms a recurrent one under both objectives, the framework surfaced a specific, cross-validated finding about the asymmetric fidelity of effector versus laterality representation, and showed that literature-derived hypotheses can be tested against a model's internal geometry with a verification procedure whose evidentiary conclusions do not depend on trusting a language model's free-form judgment. We regard this integration — not any single accuracy number — as the primary contribution of the present work.

---

## Appendix: Project summary for a technical audience

| Component | Approach |
|---|---|
| Decoding objectives | Supervised multi-task classification + HRF regression; CLIP-style contrastive brain–text alignment |
| Architectures compared | GRU, Transformer (matched capacity), statistically compared via paired resampling |
| Interpretability | 4-method attribution (Saliency, Integrated Gradients, exact Shapley, LIME); Concept Activation Vectors (CAV/TCAV), including a text-derived, brain-example-free variant unique to the contrastive model |
| Literature grounding | Dense + cross-encoder-reranked retrieval over a curated neuroscience corpus; local instruction-tuned LLM for stance labeling, claim extraction, and narration |
| Verification | Deterministic evidentiary scoring (stance × TCAV) decoupled from LLM narration |
| Statistical framework | Repeated subject-level resampling, paired non-parametric tests, bootstrap CIs, following Misra & Pessoa (2025, *eLife*) |

### Resume-ready summary points

- Designed and validated two complementary representation-learning objectives (supervised multi-task decoding; CLIP-style contrastive brain–language alignment) for 6-class movement decoding from human fMRI across 100 subjects, establishing statistically significant architecture effects (Transformer > GRU, paired Wilcoxon *p* < 10⁻⁸) via population-level, repeated-resampling evaluation rather than single-split accuracy.
- Built a four-method mechanistic interpretability suite (gradient- and perturbation-based attribution; Concept Activation Vectors) and used it to identify a reproducible, cross-architecture, cross-method asymmetry between effector-identity and body-laterality representation — including an interpretability method requiring no labeled examples, derived instead from arithmetic in a learned multimodal embedding space.
- Designed and implemented a retrieval-augmented generation system that converts literature-derived scientific claims into falsifiable tests of a model's internal representation, with a verification procedure that separates a deterministic evidentiary judgment from a language model's narrative explanation of it.
- Applied and extended established neuroimaging statistical methodology (repeated subject-level resampling, paired non-parametric significance testing, bootstrap confidence intervals) to a deep-learning decoding pipeline, matching a peer-reviewed methodological standard.
- Evaluated and improved individual components of a local, fully on-device language-model pipeline (retrieval embeddings, retrieval reranking, generation fine-tuning), quantifying which interventions transferred and which did not.
