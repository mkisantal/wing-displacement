import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class WingDataset(Dataset):
    def __init__(self, root_dir='/scratch-b/mate/wing2', trailing_edge=True, test=False, test_ids={5, 10, 15},
                transforms=None):
        
        self.root_dir = os.path.join(root_dir, 'CAM_TE' if trailing_edge else 'CAM_LE')
        print('Loading dataset containing {} images.'.format(len(os.listdir(self.root_dir))))
        
        # loading annotations
        df_full = self.load_annotations()
        df = None
        if test:
            for run_id in test_ids:
                df = df_full.loc[df_full['run_id'] == run_id] if df is None else df.append(df_full.loc[df_full['run_id']  == run_id])
        else:
            for run_id in test_ids:
                df = df_full.loc[df_full['run_id'] != run_id] if df is None else df.loc[df['run_id']  != run_id] 
        self.df = df
        
        self.transforms = transforms
        
        return
    
    def load_annotations(self):
        ann_file = sorted(os.listdir(self.root_dir))[0]
        with open (os.path.join(self.root_dir, ann_file), 'r') as f:
            df = pd.read_csv(f, sep=',', header=None)
            df.columns = ['filename', 'abs_dy', 'run_id', 'run_counter', 't', 'experiment', 'measurement', 'file_identifier', 'relative_dy']
            return df
        
    def load_image(self, index):
        # loading and resizing image
        img_name = self.df.iloc[index, 0]
        img_path = os.path.join(self.root_dir, img_name + '.jpg')
        pil_image = Image.open(img_path).crop([640, 0, 1088, 448]).resize((224, 224))
        return pil_image
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pil_img = self.load_image(idx)
        if self.transforms is not None:
            torch_image = self.transform(pil_img)
        else:
            torch_image = pil_img
        y = self.df.iloc[idx]['abs_dy']
        return torch_image, y
    
    
if __name__ == "__main__":
    w = WingDataset(trailing_edge=True, test=True)
    img, y = w[13]
    print('Sample loaded. ', 'y = {}'.format(y), img.size)
    
    