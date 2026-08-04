# NeuroLens-RAG

NeuroLens-RAG is a PyTorch-based prototype for learning temporal
representations of fMRI data, retrieving similar brain-dynamics
windows, and answering evidence-grounded questions about experiments
and scientific documents.

## Current status

- Apple Silicon development environment configured
- PyTorch MPS verified
- VS Code and Jupyter kernel configured
- Initial reproducible environment specification created

## Planned prototype

1. Reproduce a temporal GRU baseline in PyTorch
2. Learn embeddings for fMRI windows
3. Retrieve similar neural-dynamics segments
4. Build document retrieval over the accompanying paper
5. Add an evidence-grounded Streamlit interface

## Environment

```bash
conda env create -f environment.yml
conda activate neurolens
```