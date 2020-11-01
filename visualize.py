# loads a trained model and saves some results on the disk

# The trained model dict is loaded from directory 'cpk_directory' and results are saved in 'out_dir/visual_results'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import pickle
import os

# checkpoint directory
cpk_directory = 'trained_model'     # a trained model is provided in this directory. 
print('Using the trained model from directory: ', cpk_directory)
if not os.path.exists(cpk_directory):
    raise ValueError('Please specify the out_dir in visualize.py which contains the trained model dict')

out_dir = 'outputs/1'  # results will be saved in 'out_dir/visual_results'
save_dir = os.path.join(out_dir, 'visual_results')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print('Folder already exists, overwriting previous results')

batch_size_val = 10    
save_batches_n = 3     # save this many batches
samples_per_example = 4
    
# data
dataset = LIDC_IDRI(dataset_location = 'data/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(0.1 * dataset_size))

train_indices, test_indices = indices[split:], indices[:split]
test_sampler = SubsetRandomSampler(test_indices)
test_loader = DataLoader(dataset, batch_size=batch_size_val, sampler=test_sampler)
print("Number of test patches:", len(test_indices))

# network
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
net.cuda()

# load pretrained model
cpk_name = os.path.join(cpk_directory, 'model_dict.pth')
net.load_state_dict(torch.load(cpk_name))

net.eval()
with torch.no_grad():
    for step, (patch, mask, _) in enumerate(test_loader):
        if step >= save_batches_n:
            break
        patch = patch.cuda()
        mask = mask.cuda()
        mask = torch.unsqueeze(mask,1)
        output_samples = []
        for i in range(samples_per_example):
            net.forward(patch, mask, training=True)
            output_samples.append( torch.sigmoid(net.sample()).detach().cpu().numpy() )

        for k in range(patch.shape[0]):    # for all items in batch
            patch_out = patch[k, 0, :,:].detach().cpu().numpy()
            mask_out = mask[k, 0, :,:].detach().cpu().numpy()
            # pred_out = pred_mask[k, 0, :,:].detach().cpu().numpy()
            plt.figure()
            
            plt.subplot(3,2,1)
            plt.imshow(patch_out)
            plt.title('patch')
            plt.axis('off')
            plt.subplot(3,2,2)
            plt.imshow(mask_out)
            plt.title('GT Mask')
            plt.axis('off')
            
            for j in range(len(output_samples)):  # for all output samples
                plt.subplot(3, 2, j+3)
                plt.imshow(output_samples[j][k, 0, :, :])
                plt.title('prediction #'+str(j+1))
                plt.axis('off')
            
            fname = os.path.join(save_dir, 'result_'+str(step)+'_'+str(k)+'.png')
            plt.savefig(fname, bbox_inches='tight')
            plt.close()


print('Finished saving images')
