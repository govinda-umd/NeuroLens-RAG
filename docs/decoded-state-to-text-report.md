# Decoded State → Text: Design Options & Prior Art

> Research + design report, not implemented. Written to answer: concretely, what would "decoded state → text" look like for NeuroLens-RAG, and has the field done anything like it? Builds on the multi-task Transformer (Experiment 4) from [ml-design-report.md](ml-design-report.md) and the RAG discussion in [project-handoff-summary.md §19](project-handoff-summary.md#19-future-multimodal-and-rag-direction).

## 1. Restating the problem

Experiment 4's output is purely numeric: a predicted class (argmax over 6 logits), a softmax confidence, and a 5-dimensional HRF regression vector. None of that is text, and none of it is connected to your paper index. "Decoded state → text" means building the missing translation step — and it turns out there isn't one obvious way to do it. Below are three levels of sophistication, cheapest first, plus what the field has actually built at each level.

## 2. Three levels of design, from simplest to most ambitious

### Level 0 — deterministic template, no ML, no LLM

Everything needed already exists in the trained model's output: predicted class, confidence, predicted HRF vector, subject/task/run/target-volume metadata. Fill a fixed sentence template:

> "At t=32.4s into the MOTOR run for subject 101309, the model decoded **right_hand movement** with 78% confidence. Predicted HRF amplitude peaked at 0.81 for `right_hand`, consistent with the classification."

This is essentially free to build — the numbers exist, only the string-formatting doesn't. **Limitation**: no anatomical grounding ("why does the model think this?"), no literature connection, and it's stiff, repetitive prose.

### Level 1 — template + spatial attribution ("which brain regions drove this") — **implemented**

The original project vision (handoff §19) wants "model attribution or active networks" as part of the structured description — but the `TransformerDecoder` architecture doesn't give you this for free (its `input_proj = Linear(300 → 128)` mixes all 300 ROIs together at every timestep). [`06_interpretability_rsn.ipynb`](../notebooks/06_interpretability_rsn.ipynb) implements this as a post-hoc step (no retraining) using four established methods — Saliency, Integrated Gradients, exact Shapley values, and LIME — pooled to the 7 Yeo networks via `roi_labels.tsv`, giving exactly the kind of output this section originally sketched:

> "...driven primarily by the Somatomotor network (44% of attribution), consistent with hand-movement decoding."

**One caveat surfaced by actually running this**: the four methods don't fully agree. Gradient-based (Saliency, IG) and perturbation-based (Shapley, LIME) methods cluster internally (~90-95% top-1-network agreement within each pair) but only ~65% across the two families — see [interpretability-methods-notes.md §2](interpretability-methods-notes.md#2-an-empirical-finding-worth-tracking) for the specific disagreement pattern (gradient methods over-weight the Default network relative to perturbation methods). Worth resolving which family to trust — or reporting both — before wiring this into production RAG queries. That doc also surveys methods not yet implemented and Been Kim's concept-based interpretability line as a further alternative to raw network attribution.

### Level 2 — retrieval-grounded generation (RAG, properly)

Use the Level 0/1 text as a **query** into the retrieval index already built in `01_pdf_ingestion.ipynb` (chunking + MiniLM embeddings + cosine similarity), pull back the top-k paper chunks, then have a local LLM write a short paragraph using only those chunks as grounding — e.g. a constrained prompt like *"using only the provided excerpts, explain why Somatomotor-network activation is expected during a decoded right-hand movement; cite excerpt numbers."* This is the actual "RAG" step. The retrieval half already exists; only the prompt-construction and the LLM call are new.

### Level 3 — end-to-end learned brain-to-text (not recommended yet)

Skip templates entirely: train a small adapter ("brain tokenizer") that projects the Transformer's internal features directly into a pretrained LLM's embedding space, so the LLM generates text end-to-end conditioned on brain activity — the pattern several recent field results use (§3b below). This needs far more paired brain/language training data than a 5-subject MOTOR dataset provides. Flagged as a future direction, not proposed now.

## 3. What has the field actually done?

### (a) Reverse inference / meta-analytic text association — closest precedent for Levels 0–1

- **NeuroSynth** (Yarkoni et al. 2011, *Nature Methods*) — automated meta-analysis mining ~14,000+ published fMRI studies' text and reported coordinates, producing maps that link cognitive terms to activation patterns in both directions (forward: term → expected activation; reverse: activation → most-associated terms). This is the closest existing precedent to "activation pattern → associated concept," just computed from the whole published literature rather than a single trained decoder.
- **NeuroQuery** (Dockès et al. 2020, *eLife*) — generalizes NeuroSynth to free-form text queries (not just single terms) via a regression model trained on ~450K activation coordinates and paper text, with an open vocabulary of ~7,500 terms.
- **Text2Brain** (2021) and a follow-up Transformer-based version (2022) — synthesize a brain activation map directly from a free-text query, i.e. a learned, modern-language-model version of the NeuroSynth/NeuroQuery idea.
- **Chat2Brain** (2023) — puts an LLM in front of a NeuroQuery-like backend to handle open-ended natural-language queries.
- **NeuroConText** (2025) — contrastive learning with richer text representations, continuing this line.

These systems mostly run **text → brain-map** or a limited **region → associated-terms**, not full narrative generation — but they're the established precedent for "automated text-based interpretation of brain activity," which is exactly what the retrieval half of Level 2 is doing.

### (b) LLM-centric brain-to-text / captioning — closest precedent for Levels 2–3

- **Tang, LeBel, Jain & Huth (2023, *Nature Neuroscience*)**, "Semantic reconstruction of continuous language from non-invasive brain recordings" — decodes continuous natural language (not a template) from fMRI during perceived speech, imagined speech, or silent video, by combining an fMRI encoding model (built on GPT-derived contextual features) with a language-model decoder that searches for the most probable word sequence given the brain data. The most prominent fMRI+LLM decoding result in the field. Code: [github.com/HuthLab/semantic-decoding](https://github.com/HuthLab/semantic-decoding).
- **Horikawa (2025, *Science Advances*)**, "Mind captioning" (NTT) — generates full descriptive sentences of video content a person watched *or recalled from memory*, purely from fMRI, by aligning brain-decoded semantic features with a deep language model's text-feature space and iteratively optimizing candidate captions. Notably, it works without relying on language-specific brain regions — it translates nonverbal content into language via the LLM, not by reading out inner speech.
- Recent (2025) survey work on "LLM-centric" fMRI decoding describes the emerging general pattern: a small trained **brain tokenizer/adapter** projects brain features into a frozen or fine-tuned pretrained LLM's embedding space, optionally fused with a text prompt, and the LLM generates the output text. This is Level 3 above, and it's where the field is currently pushing — but every version of it needs substantially more paired brain/language data than what you have.

### (c) Adjacent: brain-to-image reconstruction — same pattern, different output modality

- **Takagi & Nishimoto (2023, CVPR)**, "High-resolution image reconstruction with latent diffusion models from human brain activity" — trains a linear encoder mapping fMRI into the latent space of a pretrained Stable Diffusion model, then lets Stable Diffusion generate the image. Structurally identical to Level 3's brain→LLM-embedding-space idea, just targeting a pretrained image generator instead of a pretrained language model.
- Follow-on work (MindEye-style, MindAligner, MindAdapter, 2024–2025) refines this brain→generative-model-latent-space adapter approach, including cross-subject generalization — relevant if NeuroLens-RAG ever scales its brain encoder beyond 5 subjects.

## 4. How this maps back onto NeuroLens-RAG's actual constraints

Every Level-3-style result above was trained on far more data than you have: the Huth lab used many hours of naturalistic story-listening per subject; NTT's Mind Captioning used dedicated video-viewing sessions with caption-matched training data; NeuroSynth/NeuroQuery draw on 10,000+ published studies. None of the field's end-to-end brain→LLM approaches are realistic to train from scratch on a 5-subject MOTOR-task dataset — that path needs an amount of paired brain/language data this project doesn't have.

What **is** realistic and well-precedented at your current scale is the Level 0 → 1 → 2 path:

```
decoded label + confidence (already exists, Level 0)
        ↓
+ network-level attribution via Schaefer/Yeo labels (Level 1, post-hoc, no retraining)
        ↓
used as a retrieval query against 01_pdf_ingestion's existing index
        ↓
local LLM writes a short paragraph grounded only in the retrieved chunks (Level 2)
```

The MOTOR task is actually a good fit for this scaled-down approach: the mapping from decoded label to established neuroanatomy is well characterized in the literature (unilateral hand/foot/tongue movement → contralateral M1/SMA), so a retrieval-grounded explanation has real literature to draw on even with a small dataset. You're not asking the system to discover new neuroscience — just to fluently explain a decode using known literature, which is a much smaller and more tractable ask than what Level 3 systems attempt.

## 5. Recommended concrete next step (if/when you want to build this)

1. ~~Extend the training/eval notebook (or a new one) to save **per-window predictions**~~ — **done**, `engine.py::predict_all` + `results/motor_v1_per_window_predictions.csv`.
2. ~~Add a **gradient- or occlusion-based attribution step** mapped through `roi_labels.tsv` to the 7 Yeo networks~~ — **done**, `06_interpretability_rsn.ipynb` + `results/rsn_attribution_per_window.csv` (Level 1). Cross-method disagreement noted above still needs resolving.
3. Write a small deterministic templating function: `{decoded label, confidence, network attribution} → 1-2 sentences` — pure code, no LLM (Levels 0+1 combined). A minimal version of this exists in `06_interpretability_rsn.ipynb`'s final example; not yet factored into a reusable function.
4. Not yet done: wire the templated text into `01_pdf_ingestion`'s retrieval and add a local LLM call for the final synthesis (Level 2).
5. **Not recommended yet**: Level 3 (learned brain-to-LLM adapter) — revisit only if the subject/data pool grows substantially. Also want to explore direct brain-representation → LLM-embedding integration here per ongoing discussion — see [interpretability-methods-notes.md §5](interpretability-methods-notes.md#5-parking-lot-level-3-brain-representation--llm-embedding-integration).

## 6. Open questions for discussion

- Per-ROI or per-network attribution? Network-level (7 Yeo networks) is coarser but far more useful as a retrieval query and far more literature-friendly than 300 individual ROI names.
- Should text generation run for every prediction, or only above some confidence threshold — given the LLM call is the most expensive step, gating on confidence avoids generating (and half-trusting) explanations for predictions the model itself is unsure about.
- Local LLM choice and how it fits the 16GB memory budget — deferred to a separate, hardware-focused discussion.

## References

- [Large-scale automated synthesis of human functional neuroimaging data (NeuroSynth), Yarkoni et al. 2011](https://pubmed.ncbi.nlm.nih.gov/21706013/)
- [NeuroQuery: comprehensive meta-analysis of human brain mapping, Dockès et al. 2020, eLife](https://elifesciences.org/articles/53385)
- [Text2Brain: Synthesis of Brain Activation Maps from Free-form Text Query](https://arxiv.org/pdf/2109.13814)
- [A Transformer-based Neural Language Model that Synthesizes Brain Activation Maps from Free-Form Text Queries](https://arxiv.org/pdf/2208.00840)
- [Chat2Brain: A Method for Mapping Open-Ended Semantic Queries to Brain Activation Maps](https://arxiv.org/pdf/2309.05021)
- [NeuroConText: Contrastive Learning for Neuroscience Meta-Analysis with Rich Text Representation, 2025](https://www.biorxiv.org/content/10.1101/2025.05.23.655707.full.pdf)
- [Semantic reconstruction of continuous language from non-invasive brain recordings, Tang et al. 2023, Nature Neuroscience](https://www.nature.com/articles/s41593-023-01304-9)
- [HuthLab/semantic-decoding (code)](https://github.com/HuthLab/semantic-decoding)
- [Mind captioning: Evolving descriptive text of mental content from human brain activity, Horikawa 2025, Science Advances](https://www.science.org/doi/10.1126/sciadv.adw1464)
- [MindCaptioning demo code (GitHub)](https://github.com/horikawa-t/MindCaptioning)
- [High-Resolution Image Reconstruction With Latent Diffusion Models From Human Brain Activity, Takagi & Nishimoto, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf)
- [StableDiffusionReconstruction (code)](https://github.com/yu-takagi/StableDiffusionReconstruction)
