# Fusing Example-Based and Natural Image Colorization via Shared Key Semantics

Official code repository for the paper:

**Fusing Example-Based and Natural Image Colorization via Shared Key Semantics**<br>
Shaojun Tong, Jingxi Lin, Fakun Chen, and Jincan Liu<br>
2025 International Conference on Virtual Reality and Visualization (ICVRV), pp. 995–1002<br>
[[Paper / DOI](https://doi.org/10.1109/ICVRV67992.2025.00173)] [[IEEE Xplore](https://ieeexplore.ieee.org/document/11410220)]

## Abstract

The field of grayscale image colorization has gained increasing attention in recent years. However, traditional methods face the challenge of failing to effectively prioritize semantically critical features during the coloring process. This paper proposes a Common-Features Guided Multi-Step Colorization Network (CMCN), an advanced framework designed to focus on semantically salient and visually prominent features for high-fidelity colorization. Building upon the Dual Embedding of Content and Color Network (DEN), our approach integrates a pre-trained natural image colorization model with an Attention-Guided Common Features Module (ACFM) to achieve accurate and naturalistic image colorization. The DEN framework enhances the model's flexibility in representing both content and color attributes, while the ACFM employs a shared-object extraction network to adaptively fuse features from diverse colorization results through confidence-based weighting, thereby improving precision and ensuring visually coherent outputs. The proposed CMCN achieves a Fréchet Inception Distance (FID) score of 7.03 on the COCO-Stuff dataset, demonstrating superior performance while maintaining computational efficiency comparable to the baseline model. The effectiveness of our semantic-guided colorization approach is rigorously validated through extensive experiments.

## Repository Structure

```text
.
├── models.py                 # Color encoder, colorization network, and attention modules
├── train2.py                 # Combined multi-step training pipeline
├── test_gray2color.py        # Evaluation/inference script
├── colorization/             # Natural-image colorization integration
├── GCoNet/                   # Shared-object extraction components
├── data/                     # Dataset preparation and loading code
├── vgg_model.py              # VGG feature extractor
└── utils.py                  # Color-space and tensor utilities
```

## Environment

The code is implemented in Python and PyTorch. Its imported dependencies include:

- PyTorch and torchvision
- NumPy, SciPy, and scikit-image
- OpenCV and Pillow
- LMDB
- ModelScope
- fvcore and pytorch-toolbelt
- pytorch-fid, tqdm, and matplotlib

Install a PyTorch build compatible with your CUDA environment first, then install the remaining dependencies as needed. Exact package versions used for the paper are not currently recorded in this repository.

## Data Preparation

Datasets are not included in this repository. Convert an ImageFolder-compatible dataset to LMDB with:

```bash
python data/prepare_data.py \
  --out /path/to/output.lmdb \
  --n_worker 20 \
  --size 256 \
  /path/to/imagefolder
```

## Training

The principal training entry points are `train.py` and `train2.py`. For example:

```bash
python train2.py \
  --datasets /path/to/output.lmdb \
  --batch 16 \
  --size 256 \
  --experiment_name cmcn
```

`train2.py` also accepts `--ckpt1`, `--root_dir`, and `--pred_dir` for the shared-object extraction stage.

## Inference

The current inference scripts contain local dataset, checkpoint, and output paths. Before running inference, update these paths in `test_gray2color.py` (or `test.py`) for your environment:

```bash
python test_gray2color.py
```

## Checkpoints and Reproducibility

Pre-trained weights, VGG weights, datasets, generated images, and experiment outputs are intentionally excluded from the repository because of their size. Several scripts still contain absolute paths from the original experimental environment; replace them with local paths before execution.

## Citation

If this work is useful in your research, please cite:

```bibtex
@inproceedings{tong2025fusing,
  author    = {Shaojun Tong and Jingxi Lin and Fakun Chen and Jincan Liu},
  title     = {Fusing Example-Based and Natural Image Colorization via Shared Key Semantics},
  booktitle = {2025 International Conference on Virtual Reality and Visualization (ICVRV)},
  pages     = {995--1002},
  year      = {2025},
  doi       = {10.1109/ICVRV67992.2025.00173}
}
```

## Acknowledgements

This repository incorporates and adapts components from prior image-colorization and co-saliency research, including [Color2Embed](https://github.com/zhaohengyuan1/Color2Embed) and the GCoNet code included under `GCoNet/`. Please also follow the licenses and citation requirements of the corresponding upstream projects.

## License

See [LICENSE](LICENSE) and the license files in included third-party components.
