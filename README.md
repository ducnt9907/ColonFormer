# ColonFormer: An Efficient Transformer based Method for Colon Polyp Segmentation
This repository contains the official Pytorch implementation of training & evaluation code for ColonFormer.

### Environment
- Creating a virtual environment in terminal: `conda create -n ColonFormer`
- Install `CUDA 11.1` and `pytorch 1.7.1`
- Install other requirements: `pip install -r requirements.txt`

### Dataset
Downloading necessary data:
1. For `Experiment 1` in our paper: 
    - Download testing dataset and move it into `./data/TestDataset/`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view).
    - Download training dataset and move it into `./data/TrainDataset/`, which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view).
2. For `Experiment 2` and `Experiment 3`:
    - All datasets we use in this experiments can be found [here](https://drive.google.com/drive/folders/1KtabJmFvLSvPzyb-MsLuqY7BfWceUmZ8?usp=sharing)
    
### Training
Download MiT's pretrained [weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia) on ImageNet-1K, and put them in a folder `pretrained/`.
Config hyper-parameters and run `train.py` for training. For example:
```
python train.py --backbone b3 --train_path ./data/TrainDataset --train_save ColonFormerB3
```
### Evaluation
For evaluation, specific your backbone version, weight's path and dataset and run `test.py`. For example:
```
python test.py --backbone b3 --weight ./snapshots/ColonFormerB3/last.pth --test_path ./data/TestDataset
```
