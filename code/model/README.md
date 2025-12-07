# Swin4D



 
### Information

This is a version of Swin4D built on [CogVideoX](https://github.com/THUDM/CogVideo/tree/main). Swin4 is a text to 4D generation pipeline that follows the plucker-conditioned ControlNet architecture originally introduced in [VD3D](https://snap-research.github.io/vd3d/).

### Installation

Install PyTorch first (we used PyTorch 2.4.0 with CUDA 12.4).

```bash
pip install -r requirements.txt
```

### Dataset

Prepare the [RealEstate10K](https://google.github.io/realestate10k/download.html) dataset following the instructions in [CameraCtrl](https://github.com/hehao13/CameraCtrl). The dataset path will be used for video_root_dir in the train and inference scripts. This is the folder structure after pre-processing:

```
- RealEstate10k
  - annotations
    - test.json
    - train.json
  - pose_files
    - 0000cc6d8b108390.txt
    - 00028da87cc5a4c4.txt
    - ...
  - video_clips
    - 0000cc6d8b108390.mp4
    - 00028da87cc5a4c4.mp4
    - ...
```

### Pre-trained ControlNet models

Swin4D-2B: [Checkpoint](https://drive.google.com/file/d/1RmTnF7mJ65s5TSqr4k_cthZXMWesd3nA/view)

Swin4D-5B: [Checkpoint](https://drive.google.com/file/d/1QsfmLmb-_Pv_pSbLrmbqBBehc9Oo6A79/view)

### Inference scripts

Swin4D: CogVideoX-2B
```bash
bash scripts/inference_2b.sh
```

Swin4D: CogVideoX-5B
```bash
bash scripts/inference_5b.sh
```

### Training requirements

The 2B model requires 48 GB memory and the 5B model requires 80 GB memory. Using one node with 8xA100 80 GB should take around 1-2 days for the model to converge.

### Training scripts

These are the fine-tuning scripts to train the ControlNet models on top of a pre-trained base model.

Swin4D-2B
```bash
bash scripts/train_2b.sh
```

Swin4D-5B
```bash
bash scripts/train_5b.sh
```

### Acknowledgements

- This code mainly builds upon [CogVideoX-ControlNet](https://github.com/TheDenk/cogvideox-controlnet)
- This code uses the original CogVideoX model [CogVideoX](https://github.com/THUDM/CogVideo/tree/main)


