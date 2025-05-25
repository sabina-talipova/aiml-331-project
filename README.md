# AIML331 - Computer Vision and Image Processing: Image Classification with CNN, Attention CNN and Vision Transformer (ViT)

This repository contains implementations of several deep learning models for image classification, including:

- Simple CNN  
- Residual CNN  
- CNN with Channel and Spatial Attention  
- Vision Transformer (ViT)  

Additionally, it includes training and evaluation scripts with TensorBoard integration for visualization.

---

## Contents

- `cnn/` - Contains SimpleCNN and ResidualCNN model implementations  
- `cnn_attention/` - Contains AttentionCNN model with channel and spatial attention  
- `vit/` - Contains VisionTransformer implementation  
- `train.py` - Training script with logging and evaluation  
- `main.py` - Entry point script to train selected model from command line  
- `dataset_wrapper.py` - Dataset loading and preprocessing (assumed)  

---

## Requirements

- Python 3.7+  
- PyTorch  
- torchvision  
- TensorBoard  
- numpy  

Install requirements with:

```bash
pip install torch torchvision tensorboard numpy
````

---

## Models Overview

### SimpleCNN

A straightforward convolutional neural network for image classification.

### ResidualCNN

A CNN with residual connections, improving gradient flow and training deeper networks.

### AttentionCNN

A hybrid CNN incorporating channel and spatial attention modules to improve feature representation by focusing on important channels and spatial locations.

### VisionTransformer (ViT)

A Transformer-based model treating images as sequences of patches, leveraging self-attention mechanisms for classification.

---

## Training

### Usage

You can train any of the models using the `main.py` script by specifying the model type and number of epochs:

```bash
python main.py --model cnn --epochs 20
python main.py --model cnn_attention --epochs 20
python main.py --model vit --epochs 20
```

### Example command:

```bash
python main.py --model cnn_attention --epochs 25
```

---

## Configuration

* Default batch size: 32
* Default learning rate: 0.001
* Default image size: 128x128
* Number of classes: 4 (configurable per model)

---

## Training Script (`train.py`)

* Loads datasets (training, validation, test) via `dataset_wrapper`
* Supports GPU if available
* Uses Adam optimizer and CrossEntropyLoss
* Logs training loss and validation accuracy to TensorBoard
* Prints training progress and final test accuracy with inference time

---

## TensorBoard Visualization

Training logs are saved in directories like `runs/exp1`, `runs/attention`, `runs/vit_patch16_heads4`, etc.

To visualize training and validation curves (loss, accuracy), run:

```bash
tensorboard --logdir runs
```

Then open the provided URL (usually `http://localhost:6006`) in your browser.

---

## Directory Structure Example

```
.
├── cnn/
│   ├── model.py            # SimpleCNN and ResidualCNN definitions
├── cnn_attention/
│   ├── model.py            # AttentionCNN definition
├── vit/
│   ├── model.py            # VisionTransformer definition
├── dataset_wrapper.py      # Dataset loading & preprocessing
├── train.py                # Training function with logging
├── main.py                 # CLI entry point to train models
├── README.md               # This file
```

---

## Notes

* Ensure you have your dataset in the correct folder (default: `data/`) or modify `root_path` in `train.py` accordingly.
* The dataset loading code is assumed to be implemented in `dataset_wrapper.py`.
* Model parameters like number of classes, image size, patch size can be modified in the config dictionaries within `main.py`.

