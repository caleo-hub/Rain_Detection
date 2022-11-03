from torch.utils.data import Dataset
import torch
from torchvision import datasets, io
import glob
import numpy as np
import os

from PIL import Image


class RainDetectionDataset(Dataset):
    def __init__(self, path,transform):
        self.get_images(path)
        self.transform = transform

            
        
        
        
    def get_images(self, path):
        files = glob.glob(path + '*\**\*.npy', recursive=True)
        self.labels = torch.empty((0))
        self.images = torch.empty((0,3,100,100))
        for f in files:
            data, label_tensor = self.load_images(f)
            self.images = torch.cat((self.images,data), dim=0)
            self.labels = torch.cat((self.labels,label_tensor), dim=0)
            
            
    def load_images(self, file):
        data = self.npy_loader(file)
        
        data2 = torch.stack((data,)*3, axis=-1)
        data2 = torch.moveaxis(data2, -1, 1)
        #print(data2.size())
        label = torch.tensor(self.class_dict(file))
        label_tensor = label.repeat(data.size(0))
        
        return data2, label_tensor
    
    def class_dict(self,file):
        
        class_dict = {
            'dataset\\no_rain\\day\\imgdata1.npy': 1, #'no_rain_day'
            'dataset\\no_rain\\day\\imgdata2.npy':1,  #'no_rain_day',
            'dataset\\no_rain\\day\\imgdata3.npy':1, #'no_rain_day',
            'dataset\\no_rain\\night\\imgdata1.npy': 2, #'no_rain_night',
            'dataset\\no_rain\\night\\imgdata2.npy':2, #'no_rain_night',
            'dataset\\rain\\day\\heavy_rain\\imgdata1.npy':3, #'rain_day_heavy',
            'dataset\\rain\\day\\small_rain\\imgdata1.npy':4, #'rain_day_small',
            'dataset\\rain\\day\\small_rain\\imgdata2.npy':4, #'rain_day_small',
            'dataset\\rain\\night\\heavy_rain\\imgdata1.npy':5, #'rain_night_heavy',
            'dataset\\rain\\night\\small_rain\\imgdata1.npy':6, #'rain_night_small',
            'dataset\\rain\\night\\small_rain\\imgdata2.npy':6, #'rain_night_small',
        }
        
        return class_dict[file]
            
    def __getitem__(self, index):
        
        
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image.numpy())
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def __len__(self):
        return self.labels.size(0)
        
        
    def npy_loader(self,path):
        sample = torch.from_numpy(np.load(path))
        return sample

class ImageRainDetectionDataset(Dataset):
    def __init__(self, path,transform):
        self.path = path
        self.transform = transform 
        self.get_files()
        
    def get_files(self):
        self.files = glob.glob(self.path + '*\**\*.jpg', recursive=True)
            
    def load_images(self, index):
        image_file = self.files[index]
        data = Image.open(image_file)
        label = torch.tensor(self.class_dict(os.path.dirname(image_file)))
        return data, label
    
    def class_dict(self,file):
        
        class_dict = {
            'image_rain_dataset\\no_rain\\day': 0, #'no_rain_day'
            'image_rain_dataset\\no_rain\\night': 1, #'no_rain_night',
            'image_rain_dataset\\rain\\day\\heavy_rain':2, #'rain_day_heavy',
            'image_rain_dataset\\rain\\day\\small_rain':3, #'rain_day_small',
            'image_rain_dataset\\rain\\night\\heavy_rain':4, #'rain_night_heavy',
            'image_rain_dataset\\rain\\night\\small_rain':5, #'rain_night_small',
        }
        
        return class_dict[file]
    
    def __getitem__(self, index):
        
        
        image, label = self.load_images(index)
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def __len__(self):
        return len(self.files)