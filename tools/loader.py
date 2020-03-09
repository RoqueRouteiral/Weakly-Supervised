import os
import torch
import random
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from scipy import ndimage
import matplotlib.pyplot as plt



def get_loaders(cf, phase='train'):
    patients_train = os.listdir(cf['dataset_path']+'train/')
    patients_val = os.listdir(cf['dataset_path']+'validation/')

    #whether to shuffle or not the images
    if cf['shuffle_data']:
        random.shuffle(patients_train)
        random.shuffle(patients_val) 
    #Creating Data Generator per split
    train_set = Brats_Dataset(patients_train, cf, 'train')
    val_set = Brats_Dataset(patients_val, cf, 'validation')

    train_gen = DataLoader(train_set, batch_size=cf['batch_size'])
    val_gen = DataLoader(val_set, batch_size=cf['batch_size'])
    
    return train_gen, val_gen


class Brats_Dataset(Dataset):
    """Brats Cancer dataset."""

    def __init__(self, indices, cf, phase):
        """
        Args:
            indices : list of the indices for this generator
            cf (Config file): set up info
            phase: train loader or eval loader. Important to apply or not DA.
        """
        self.indices = indices
        self.cf = cf
        self.phase = phase


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx): 
        img_name = os.path.join(self.cf['dataset_path'], self.phase, self.indices[idx])
        label = int(img_name[-5])
        img_np = plt.imread(img_name)
        img_np=np.stack((img_np,)*3, axis=0)
        if self.cf['resize']:
            img_np = resize(img_np,(3,self.cf['size'],self.cf['size']),mode='constant',anti_aliasing=True)    
        
        img = torch.from_numpy(img_np.copy()).float()

        data = (img,label)
        patient_name = self.indices[idx]

        return data, patient_name
        