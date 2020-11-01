# About the repo
This is yet another PyTorch implementation of the Probabilistic UNet. This repo is forked from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch. I have enabled the checkpoint saving and implemented a visualization method.

Probabilistic U-Net paper for segmentation of ambiguous images: https://arxiv.org/abs/1806.05034.
Official code repo: https://github.com/SimonKohl/probabilistic_unet.

ToDo: add some result images here

## Modifications
Please note that in the spirit of understanding the code, the test set is currently being used as the validation set and the model checkpoint is saved based on the best test loss. This is not fair to compare against methods that treat the held-out set as the test set. For fair quantitative evaluation, you probably need to remove the validation loop from `train_model.py`

I have fixed the dataset for train/val purposes. This makes it possible to reliably use the same train/val split for different scripts (training and visualization).

This code was tested with PyTorch 1.6. You probably need to install [pydicom](https://pydicom.github.io/): `pip install pydicom`.
 
### Adding KL divergence for Independent distribution


# Usage
Note: the original repository mentions the need to add the KL divergence loss in PyTorch code. I don't think there is any need to manually add KL divergence in the PyTorch code. [PyTorch distributions](https://pytorch.org/docs/stable/distributions.html) now include the KL divergence for independent Normal distribution. This works fine in PyTorch 1.6, and probably some older versions as well.

## Training
To train the network, use `python3 train_model.py`. This file contains optimization settings and also a directory name (`out_dir`). The trained model and training logs will be saved in this directory: `out_dir`.

To train your own model, you need to prepare a dataset that yields (input patch, segmentation labels) and you should be able to use the pretty much the same code as in `train_model.py`. 

Dealing with NaNs: sometimes, you might encounter NaNs. Decreasing the learning rate might be easiest solution. Of course, modifying the ELBO loss would be more fancy :)

## Visualizing
To save results of segmentation, run `python3 visualize.py`. This will load the trained model, by default it will load the trained model provided with this repo in `trained_model/model_dict.pth`. To use your own trained model, set the `cpk_directory` in `visualize.py`. 


## Train on LIDC Dataset
One of the datasets used in the original paper is the [LIDC dataset](https://wiki.cancerimagingarchive.net). [The original repo](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) preprocessed this data and stored them in a pickle file, which you can [download here](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5?usp=sharing). After downloading the files you should place them in a folder called 'data'. After that, you can train your own Probabilistic UNet on the LIDC dataset using the simple train script provided in `train_model.py`.