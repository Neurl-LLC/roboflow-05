# Creating Custom Datasets with PyTorch and Roboflow

This repository demonstrates two approaches to working with custom datasets for object detection: one using **PyTorch** and the other using **Roboflow**.

## Custom Dataset with PyTorch

The notebook [`Custom_Dataset_with_pytorch.ipynb`](./Custom_Dataset_with_pytorch.ipynb) shows how to build a custom dataset in PyTorch by subclassing the `Dataset` class and implementing the required methods:

* `__init__` – for initialization and loading annotations
* `__getitem__` – for retrieving samples (images and labels)
* `__len__` – for defining dataset length

We then train a **Faster R-CNN** model from `torchvision` using the dataset and `DataLoader`.

## Custom Dataset with Roboflow

The notebook [`Custom_Dataset_with_roboflow.ipynb`](./Custom_Dataset_with_roboflow.ipynb) demonstrates how to use Roboflow to simplify dataset handling and model training.

### Installation

```bash
pip install -q rfdetr==1.2.1 supervision==0.26.1 roboflow
```

### Downloading a Dataset from Roboflow Universe

```python
from roboflow import download_dataset

dataset = download_dataset(
    "https://universe.roboflow.com/roboflow-jvuqo/poker-cards-fmjio/dataset/4",
    "coco"
)
```

### Training an RF-DETR Model

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall()

model.train(dataset_dir=dataset.location, epochs=10, batch_size=8, grad_accum_steps=2)
```

With this setup, Roboflow automatically handles dataset preparation, letting you focus on training and experimentation.
