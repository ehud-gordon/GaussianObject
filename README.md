# Geometric Computer Vision 236861 Final Project
## Improving Gaussian-Splatting in sparse-setting using Diffusion
## by: Eldad Matmon and Ehud Gordon

Look at [report.ipynb](https://github.com/ehud-gordon/GaussianObject/blob/ours/random_poses/report.ipynb) for the project report. 


## Setup

### CUDA

GaussianObject is tested with CUDA 11.7. If you are using a different version, you can choose to install [nvidia/cuda](https://anaconda.org/nvidia/cuda) in a local conda environment or modify the version of [PyTorch](https://pytorch.org/get-started/previous-versions/) in `requirements.txt`.

### Cloning the Repository

The repository contains submodules. Please clone it with

```sh
git clone https://github.com/GaussianObject/GaussianObject.git --recursive
```

or update submodules in `GaussianObject` directory with

```sh
git submodule update --init --recursive
```

### Dataset

You can try GaussianObject with the Mip-NeRF360 dataset and OmniObject3D dataset. The data can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1DUOxFybdsSYJHI5p79O_QH87TIODiJ8h).

The directory structure of the dataset should be

```text
GaussianObject
├── data
│   ├── mip360
│   │   ├── bonsai
│   │   │   ├── images
│   │   │   ├── images_2
│   │   │   ├── images_4
│   │   │   ├── images_8
│   │   │   ├── masks
│   │   │   ├── sparse
│   │   │   ├── zoe_depth
│   │   │   ├── zoe_depth_colored
│   │   │   ├── sparse_4.txt
│   │   │   ├── sparse_6.txt
│   │   │   ├── sparse_9.txt
│   │   │   └── sparse_test.txt
│   │   ├── garden
│   │   └── kitchen
│   └── omni3d
└── ...
```

`images`, `images_2`, `images_4`, `images_8` and `sparse` are from the original dataset. `masks` is the object mask generated with [segment-anything](https://github.com/facebookresearch/segment-anything). `zoe_depth` and `zoe_depth_colored` are the depth maps and colored depth maps. `sparse_4.txt`, `sparse_6.txt` and `sparse_9.txt` are train set image ids and `sparse_test.txt` is the test set.

To test GaussianObject with your own dataset, you can manually prepare the dataset with the same directory structure. The depth maps and colored depth maps are generated with

```sh
python pred_monodepth.py -s <YOUR_DATA_DIR>
```

### Python Environment

GaussianObject is tested with Python 3.10. All the required packages are listed in `requirements.txt`. You can install them with

```sh
# setup pip packages
pip install -r requirements.txt

# setup submodules
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/pytorch3d
pip install -e submodules/minLoRA
pip install -e submodules/CLIP
```

### Pretrained ControlNet Model

Pretrained weights of Stable Diffusion v1.5 and ControlNet Tile need to be put in `models/` following the instruction of [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) with our given script

```sh
python download_hf_models.py
```

## Run the Code

Taking the scene `kitchen` from `mip360` dataset as an example, GaussianObject generate the visual hull of it, train a coarse 3DGS representation, analyze the statistical regularity of the coarse model with leave-one-out strategy, fine-tune the Gaussian Repair Model with LoRA and repair the 3DGS representation step by step.

### Visual Hull

```sh
python visual_hull.py \
    --sparse_id 4 \
    --data_dir data/mip360/kitchen \
    --reso 2 --not_vis
```

The visual hull is saved in `data/mip360/kitchen/visual_hull_4.ply`.

### Coarse 3DGS

```sh
python train_gs.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
```

You can render the coarse model it with

```sh
# render the test set
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --skip_all --skip_train

# render the path
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --render_path
```

The rendering results are saved in `output/gs_init/kitchen/test/ours_10000` and `output/gs_init/kitchen/render/ours_10000`.

### Leave One Out

```sh
python leave_one_out_stage1.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background

python leave_one_out_stage2.py -s data/mip360/kitchen \
    -m output/gs_init/kitchen_loo \
    -r 4 --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --random_background
```

### LoRA Fine-Tuning

```sh
python train_lora.py --exp_name controlnet_finetune/kitchen \
    --prompt xxy5syt00 --sh_degree 2 --resolution 4 --sparse_num 4 \
    --data_dir data/mip360/kitchen \
    --gs_dir output/gs_init/kitchen \
    --loo_dir output/gs_init/kitchen_loo \
    --bg_white --sd_locked --train_lora --use_prompt_list \
    --add_diffusion_lora --add_control_lora --add_clip_lora
```

### Gaussian Repair

```sh
python train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu 0 \
    tag="kitchen" \
    system.init_dreamer="output/gs_init/kitchen" \
    system.exp_name="output/controlnet_finetune/kitchen" \
    system.refresh_size=8 \
    data.data_dir="data/mip360/kitchen" \
    data.resolution=4 \
    data.sparse_num=4 \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2
```

The final 3DGS representation is saved in `output/gaussian_object/kitchen/save/last.ply`. You can render it with

```sh
# render the test set
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --skip_all --skip_train \
    --load_ply output/gaussian_object/kitchen/save/last.ply

# render the path
python render.py \
    -m output/gs_init/kitchen \
    --sparse_view_num 4 --sh_degree 2 \
    --init_pcd_name visual_hull_4 \
    --white_background --render_path \
    --load_ply output/gaussian_object/kitchen/save/last.ply
```

# Acknolwedgements
This project draws inspiration from the GaussianObject project and Gaussian Splatting project by Inria.
