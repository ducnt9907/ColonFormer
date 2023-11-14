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
    - All datasets we use in this experiments can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1ExJeVqbcBn6yy-gdGqEYw5phJywHIUXZ/view?usp=sharing)
    
### Training
Download MiT's pretrained `weights` 
(
[google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) | 
[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ)
) on ImageNet-1K, and put them in a folder `pretrained/`.
Config hyper-parameters and run `train.py` for training. For example:
```
python train.py --backbone b3 --train_path ./data/TrainDataset --train_save ColonFormerB3
```
Here is an example in [Google Colab](https://colab.research.google.com/drive/1vUgh7XCiVyboYIAaRBQ2TDVMi8v0CLLK?usp=sharing)
### Evaluation
For evaluation, specific your backbone version, weight's path and dataset and run `test.py`. For example:
```
python test.py --backbone b3 --weight ./snapshots/ColonFormerB3/last.pth --test_path ./data/TestDataset
```
We provide some [pretrained weights](https://drive.google.com/drive/folders/1SVxluPlRVohkN6Q6hG-FpA9L8eapZuxa?usp=sharing) in case you need.

### Citation
If you find this code useful in your research, please consider citing:

``
@article{duc2022colonformer,
  title={Colonformer: An efficient transformer based method for colon polyp segmentation},
  author={Duc, Nguyen Thanh and Oanh, Nguyen Thi and Thuy, Nguyen Thi and Triet, Tran Minh and Dinh, Viet Sang},
  journal={IEEE Access},
  volume={10},
  pages={80575--80586},
  year={2022},
  publisher={IEEE}
}
``
<code>
@inproceedings{ngoc2021neounet,
  title={NeoUNet: Towards accurate colon polyp segmentation and neoplasm detection},
  author={Ngoc Lan, Phan and An, Nguyen Sy and Hang, Dao Viet and Long, Dao Van and Trung, Tran Quang and Thuy, Nguyen Thi and Sang, Dinh Viet},
  booktitle={Advances in Visual Computing: 16th International Symposium, ISVC 2021, Virtual Event, October 4-6, 2021, Proceedings, Part II},
  pages={15--28},
  year={2021},
  organization={Springer}
}
</code>
<code>
@article{thuan2023rabit,
  title={RaBiT: An Efficient Transformer using Bidirectional Feature Pyramid Network with Reverse Attention for Colon Polyp Segmentation},
  author={Thuan, Nguyen Hoang and Oanh, Nguyen Thi and Thuy, Nguyen Thi and Perry, Stuart and Sang, Dinh Viet},
  journal={arXiv preprint arXiv:2307.06420},
  year={2023}
}
</code>



