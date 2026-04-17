# modular_i2imt

Modular Image-to-Image Machine Translation (I2IMT) pipeline — a collection of tools
to generate synthetic image/text pairs, run OCR, translate text, and render translated
text back into images. This repository provides modular scripts for dataset
generation, image editing using AnyText models, multiple MT backends, and OCR
backends so you can mix-and-match components for research and experimentation.

**Modules and purpose**
- **Generator**: create synthetic image datasets by drawing source and target text
	on images with configurable fonts, sizes and slopes. See [generator/README.md](generator/README.md) for details.
- **Image generation / editing**: produce edited images containing translated text
	using AnyText-style diffusion editing via [img_gen/predict.py](img_gen/predict.py).
- **Machine Translation (MT)**: multiple translator wrappers to convert YAML-formatted
	text metadata using different models: [MT/nllb.py](MT/nllb.py), [MT/madlad.py](MT/madlad.py), [MT/gemma.py](MT/gemma.py), [MT/seed.py](MT/seed.py).
- **OCR**: extract text and bounding boxes using several OCR approaches: EasyOCR,
	Doctr/TrOCR and DeepSeek at [OCR/easy-ocr.py](OCR/easy-ocr.py), [OCR/trocr.py](OCR/trocr.py), [OCR/deepseek.py](OCR/deepseek.py).
- **Utilities**: small helper scripts for curation and image generation are in [generator/](generator/).

**Quickstart**

1. Install common Python dependencies (or use the environments explained below):

```bash
pip install pillow numpy pyyaml tqdm easyocr jiwer shapely opencv-python transformers torch modelscope doctr vllm
```

2. Generate plain background images (optional):

```bash
python generator/images.py
```

3. Generate a synthetic I2IMT dataset using a config or CLI flags (example):

```bash
python generator/data_generator.py \
	--images generator/images \
	--src_text wmt15/de-en/test.de \
	--tgt_text wmt15/de-en/test.en \
	--src_lang de --tgt_lang en \
	--output data/de-en/test \
	--font_path generator/font/Arial_Unicode.ttf \
	--num_images 100
```

4. (Optional) Curate generated images to fixed size using the curation helper:

```bash
python generator/curate.py data/de-en/test_de
```

5. Run OCR to extract existing text & bboxes from images:

```bash
python OCR/easy-ocr.py path/to/images/* -l en -o ocr_output.yaml
```

6. Translate OCR outputs using one of the MT wrappers (example with NLLB):

```bash
python MT/nllb.py ocr_output.yaml -src en -tgt de --output translated.yaml
```

7. Render translated text back into images with AnyText SD editing:

```bash
python img_gen/predict.py path/to/images/* -bboxes ocr_output.yaml -t translated.yaml --output out_images
```

**Environments**

For each major part of the pipeline we provide a dedicated conda environment YAML so
you can create isolated environments tuned to the dependencies and hardware needs
of that component:

- **MT environment**: [MT/environment.yaml](MT/environment.yaml) — contains
	translation-related packages (PyTorch, Hugging Face `transformers`, `vllm`,
	tokenizers and model/runtime dependencies). Use this environment for running
	scripts in the `MT/` folder.
- **Image editing / AnyText environment**: [img_gen/environment.yaml](img_gen/environment.yaml) —
	includes `modelscope`, image libraries, and GPU-accelerated dependencies used
	by `img_gen/predict.py` and AnyText-style image editing.
- **OCR environment**: [OCR/environment.yaml](OCR/environment.yaml) — contains
	OCR toolkits such as `easyocr`, Doctr-related packages, OpenCV and other
	imaging dependencies used by the scripts in `OCR/`.

Create an environment from one of the YAML files with:

```bash
conda env create -f MT/environment.yaml
conda env create -f img_gen/environment.yaml
conda env create -f OCR/environment.yaml
```

Each YAML includes pinned package versions and channels — adapt the file if you
need different CUDA/tool versions for your hardware.

**Launcher and batch generation**

The generator includes a simple launcher to process a directory of YAML configs:

- [generator/launcher.sh](generator/launcher.sh)

**Repository layout**
- [generator/](generator/): dataset creation, image utilities and configs.
- [img_gen/](img_gen/): AnyText-style image editing pipeline.
- [MT/](MT/): translation wrapper scripts for different models.
- [OCR/](OCR/): OCR extractor scripts and helpers.
- [cascade/](cascade/), [posteval/](posteval/): CSVs and results used for evaluation and experiments.

**Notes & tips**
- Many scripts accept a YAML data file mapping image names to text and bounding boxes.
- GPU is recommended for OCR, MT and AnyText models. Check each script's flags.
- Some models require authentication tokens or large GPU memory; adapt model choices
	to your hardware.

If you want, I can also:
- Add a consolidated requirements file (requirements.txt).
- Create short run examples for a specific language pair and environment.

---
Last updated: consolidated README generated from repository files.
