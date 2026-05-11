# PhysMamba

Official PyTorch implementation of **PhysMamba: Physics-Inspired Thermal-Mamba for Thermally Consistent Visible-to-Infrared Image Generation**.

PhysMamba is a physics-inspired visible-to-infrared image generation framework designed to improve both structural fidelity and thermal-distribution consistency. The framework combines cross-modal alignment pretraining, cross-level skip feature fusion, and a Thermal-Mamba auxiliary discriminator for long-range thermal-distribution modeling.

## Highlights

- **Two-stage training framework** for visible-to-infrared image generation.
- **Cross-modal alignment pretraining** with masked reconstruction and contrastive learning to learn shared geometric priors between visible and infrared images.
- **CL-Skip generator** for cross-level feature fusion and high-frequency detail recovery.
- **Thermal-Mamba discriminator** that injects global thermal statistics and long-range dependency modeling into the adversarial constraint.
- Evaluation support for common generation metrics, including SSIM, MS-SSIM, LPIPS, L1, PSNR, FID, and inference speed.

## Repository Structure

```text
Physmamba/
|-- train.py                 # Training entry point
|-- test.py                  # Image generation / qualitative testing
|-- evaluate.py              # Quantitative evaluation
|-- eval.py                  # Validation helper used during training
|-- requirements.txt         # Python dependencies recorded for the project
|-- data/                    # Dataset loaders
|-- models/                  # PhysMamba, generator, discriminator, and loss modules
|   |-- physmamba.py         # Main PhysMamba training model
|   |-- networks.py          # Generator and common network components
|   |-- CSAF.py              # Cross-level skip feature fusion module
|   +-- unetgan/
|       |-- mamba_discriminator.py  # Thermal-Mamba discriminator
|       +-- self_perceptual_loss.py # Encoder-based perceptual loss
|-- options/                 # Training and testing options
|-- util/                    # Visualization and utility functions
+-- lpips/                   # LPIPS implementation and weights
```

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n physmamba python=3.8 -y
conda activate physmamba
pip install -r requirements.txt
```

The Thermal-Mamba discriminator also depends on Mamba-related packages. If they are not installed by your environment file, install them separately according to your CUDA and PyTorch versions:

```bash
pip install einops timm mamba-ssm pytorch-fid
```

> Note: `mamba-ssm` may require a CUDA-compatible PyTorch build. Please install the wheel that matches your local CUDA/PyTorch environment.

## Data Preparation

For paired visible-to-infrared training, organize each dataset as follows:

```text
datasets/
+-- M3FD/
    |-- train/
    |   |-- 000001_vis.png
    |   |-- 000001_ir.png
    |   |-- 000002_vis.png
    |   +-- 000002_ir.png
    +-- test/
        |-- 000101_vis.png
        |-- 000101_ir.png
        |-- 000102_vis.png
        +-- 000102_ir.png
```

The visible image is treated as domain `A`, and the infrared image is treated as domain `B`. The current dataloader supports paired samples named with `_vis` and `_ir` suffixes.

Supported dataset modes include:

```text
VEDAI, FMB, M3FD, AVIID_1, KAIST, LLVIP, FLIR
```

For most image-based datasets, use:

```text
<dataroot>/train/*_vis.png
<dataroot>/train/*_ir.png
<dataroot>/test/*_vis.png
<dataroot>/test/*_ir.png
```

For `LLVIP`, the current loader expects `.jpg` files:

```text
*_vis.jpg
*_ir.jpg
```

For `FLIR`, the loader expects preprocessed NumPy arrays:

```text
grayscale_training_data.npy
thermal_training_data.npy
grayscale_test_data.npy
thermal_test_data.npy
```

## Pretrained Encoder

PhysMamba uses a first-stage cross-modal alignment encoder to initialize the generation model and to compute the encoder-based perceptual loss. Before training the generator, prepare a pretrained encoder checkpoint containing:

```text
vis_encoder
```

Example path:

```text
pretrained/encoders_best.pth
```

If pretrained weights are released separately, place them under `pretrained/` or pass their path through `--pretrained_encoder_path`.

## Training

Example training command on M3FD:

```bash
python train.py \
  --dataset_mode M3FD \
  --dataroot ./datasets/M3FD \
  --name physmamba_m3fd \
  --model physmamba \
  --which_model_netG unet_512 \
  --which_model_netD unetdiscriminator \
  --which_direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --lambda_A 100 \
  --norm instance \
  --pool_size 0 \
  --loadSize 512 \
  --fineSize 512 \
  --batchSize 2 \
  --nThreads 4 \
  --gpu_ids 0 \
  --checkpoints_dir ./checkpoints \
  --pretrained_encoder_path ./pretrained/encoders_best.pth
```

Checkpoints and training logs are saved to:

```text
checkpoints/<experiment_name>/
```

To resume training:

```bash
python train.py \
  --dataset_mode M3FD \
  --dataroot ./datasets/M3FD \
  --name physmamba_m3fd \
  --model physmamba \
  --continue_train \
  --which_epoch latest \
  --checkpoints_dir ./checkpoints \
  --pretrained_encoder_path ./pretrained/encoders_best.pth
```

## Testing

Generate infrared images from visible inputs:

```bash
python test.py \
  --dataset_mode M3FD \
  --dataroot ./datasets/M3FD \
  --name physmamba_m3fd \
  --model physmamba \
  --which_epoch 200 \
  --phase test \
  --which_direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --loadSize 512 \
  --fineSize 512 \
  --gpu_ids 0 \
  --checkpoints_dir ./checkpoints \
  --results_dir ./results \
  --how_many 100000
```

The visual results are saved to:

```text
results/<experiment_name>/test_<epoch>/
```

## Evaluation

Run quantitative evaluation:

```bash
python evaluate.py \
  --dataset_mode M3FD \
  --dataroot ./datasets/M3FD \
  --name physmamba_m3fd \
  --model physmamba \
  --which_epoch 200 \
  --phase test \
  --which_direction AtoB \
  --input_nc 3 \
  --output_nc 1 \
  --loadSize 512 \
  --fineSize 512 \
  --gpu_ids 0 \
  --checkpoints_dir ./checkpoints \
  --results_dir ./results
```

The script reports:

```text
SSIM, MS-SSIM, LPIPS, L1, PSNR, FID, FPS
```

## Checkpoint Naming

The training code saves model weights using the following format:

```text
<epoch>_net_G.pth
<epoch>_net_D.pth
<epoch>_net_D_mamba.pth
latest_net_G.pth
latest_net_D.pth
latest_net_D_mamba.pth
```

For example, when testing with `--which_epoch 200`, place the following files under `checkpoints/<experiment_name>/`:

```text
200_net_G.pth
200_net_D.pth
200_net_D_mamba.pth
```

## Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@article{sun2026physmamba,
  title   = {PhysMamba: Physics-Inspired Thermal-Mamba for Thermally Consistent Visible-to-Infrared Image Generation},
  author  = {Sun, Xiaokun and Li, Shuang and Hu, Canbin},
  journal = {Under review},
  year    = {2026}
}
```

The BibTeX entry will be updated after publication.

## Acknowledgements

This codebase is developed based on the PyTorch implementations of pix2pix/CycleGAN and InfraGAN, with additional PhysMamba modules for cross-level feature fusion and Thermal-Mamba thermal-distribution discrimination.

## License

This project is released under the MIT License.
