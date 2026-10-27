# PRISM

PRISM generates a material normal-distribution-function (NDF) shape from a text prompt. It is a research prototype that returns a 64×64, single-channel NDF grid; it does not generate a complete PBR material or guarantee physically calibrated reflectance.

## Current inference models

The exported pipeline has three ONNX graphs, grouped into two release downloads:

1. **E5 text encoder** (`e5_encoder.onnx` + external weights and tokenizer): `intfloat/e5-large-v2`, MIT-licensed. It encodes `query: {prompt}` as a normalized 1024-float vector.
2. **PRISM material model** (`retriever.onnx` and `vae_decoder.onnx`, each with its external `.onnx.data` file): the retriever selects and blends five learned prototypes into a 16-float latent; the Residual V3 Zernike decoder maps that latent to 4096 floats (64×64 NDF). Directional prompt detection and centroid recentering run outside ONNX.

The reference ONNX bundle and export/parity script are generated locally under `out/prism_models/`; model binaries are intentionally excluded from Git. See `docs/handoff/MODEL_INTEGRATION_ANALYSIS.md` for tensor contracts and engine integration guidance. The exported CPU graphs were compared with the PyTorch reference on gold, velvet, and brushed-aluminum prompts; maximum absolute differences were below 4.1e-6 for the final NDF grids. This is a limited parity check, not a broad quality benchmark.

## Training data and provenance

The selected text retriever was fit on the canonical named-material text cache, and the selected VAE was trained/evaluated on NDF grids derived from the canonical material collection. That collection contains 100 MERL BRDF records and 62 RGL materials. MERL BRDFs are converted to normalized NDF-shaped proxies from their retroreflective slice; they are not standalone measured NDFs. RGL provides processed material BRDF data and extracted NDF-related representations. The retriever also uses E5 query embeddings for the training captions and the VAE's 16D encoded material latents as prototypes.

The selected checkpoints do **not** use the experimental DTU/Nielsen, Ngan anisotropic, Cornell, Urban BRDF, or LUH cohorts. Those sources are retained in historical research notes or experimental tooling only. No raw datasets, checkpoints, or training caches are included in this repository or the release archives.

Training and preprocessing details, limitations, source citations, and the selected checkpoint identities are documented in [the model integration analysis](docs/handoff/MODEL_INTEGRATION_ANALYSIS.md), [the text-to-latent handoff](docs/handoff/TEXT_TO_LATENT_HANDOFF.md), and [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).

## Set up the development environment

Requires Python 3.12. Create and activate a virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, call the venv interpreter directly instead: `\.venv\Scripts\python.exe`. The pinned PyTorch build in `requirements.txt` targets CUDA 13.2; for CPU-only systems or a different CUDA stack, install the matching PyTorch build from the official PyTorch selector, then install the remaining requirements. Check the device selected by scripts with `--device auto`, or set `--device cpu` / `--device cuda` where supported.

Downloaded datasets, trained checkpoints, generated caches, results, environments, and exports stay local and are ignored by Git. Do not commit raw data or checkpoints.

## Run the pipeline

From the activated venv, the end-to-end development sequence is:

```powershell
# Download only the canonical MERL and RGL source datasets
python scripts/download_datasets.py --dataset merl
python scripts/download_datasets.py --dataset rgl

# Convert BRDF inputs to 64x64 NDF grids and augment training grids
python -m prism.preprocess
python -m prism.augment

# Train/validate the VAE (the selected released V3 checkpoint is not checked in)
python -m prism.train --basis-type zernike --poly-degree 8 --latent-dim 16 --epochs 120 --batch-size 64 --beta 0.001 --peak-weight 30 --mse-weight 10 --lr 0.0003 --scheduler plateau --seed 42 --residual-decoder --residual-decoder-version 3 --checkpoint-dir checkpoints/zernike_residual_v3
python -m prism.validate --checkpoint checkpoints/zernike_residual_v3/best_model.pt

# Build E5 caption/latent cache, train text predictor, and evaluate
python -m prism.text_dataset --source grid --checkpoint checkpoints/zernike_residual_v3/best_model.pt --embedding-model intfloat/e5-large-v2
python -m prism.train_text --data-cache checkpoints/text_dataset_grid.npz --vae-checkpoint checkpoints/zernike_residual_v3/best_model.pt --checkpoint checkpoints/best_text_model.pt
python -m prism.eval_text --data-cache checkpoints/text_dataset_grid.npz --vae-checkpoint checkpoints/zernike_residual_v3/best_model.pt --text-checkpoint checkpoints/best_text_model.pt

# Generate a material NDF preview
python -m prism.infer_text --prompt "brushed aluminum" --output results/brushed_aluminum.png
```

For released model reproduction, use the exact checkpoint and settings in `docs/handoff/TEXT_TO_LATENT_HANDOFF.md`; the commands above are a basic pipeline template, not a claim that retraining reproduces the selected release bit-for-bit. Model downloads may require Hugging Face access the first time.

## Export ONNX

Install the export-only packages and run the parity exporter:

```powershell
python -m pip install onnx onnxscript onnxruntime
python scripts/export_prism_onnx.py
```

The selected text and VAE checkpoints must exist at the paths configured in the exporter; the E5 model/tokenizer must be cached locally. Outputs go to ignored `out/prism_models/`. This includes ONNX external-weight sidecars, tokenizer files, NDF fixtures, and a Python-vs-ONNX parity report. Keep each `.onnx.data` file beside its matching graph.

## Download/release

Model binaries are distributed as two separately licensed GitHub Release assets: the E5 encoder/tokenizer package and the PRISM retriever/decoder package. Each archive includes a manifest with version, source checkpoint/model revision, tensor contracts, SHA-256 hashes, and applicable license notices. The release procedure is documented in `docs/release.md`.

## License

PRISM source code is licensed under Apache-2.0; see [`LICENSE`](LICENSE). Third-party model and data rights are component-specific and listed in [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES). The PRISM-trained model assets are designated CC BY-SA 4.0 for the model release, based on the confirmed MERL data license; do not apply that license to the separately licensed E5 model files.
