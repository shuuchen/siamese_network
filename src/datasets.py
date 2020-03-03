from options import Config
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import numpy as np
import random
import os


class MadoriDataset(Dataset):
    
    def _prepare(self):
        if self.test:
            data_file = Config.test_file
        else:
            data_file = Config.train_file if self.train else Config.val_file
        with open(data_file, 'r') as f:
            lines = f.readlines()
        f.close()
        self.unit_paths = [os.path.join(Config.data_dir, line.strip()) for line in lines]
        
    def __init__(self, img_size=(256, 256), train=True, test=False):
        self.train = train
        self.test = test
        self.img_size = (256, 256)
        self._prepare()
        
    def __len__(self):
        return len(self.unit_paths)
    
    def _resize(self, img):
        w, h = img.size
        if w < h:
            a = 256.0 / h
            b = int(w * a)
            img = img.resize((b, 256), Image.BILINEAR)
        else:
            a = 256.0 / w
            b = int(h * a)
            img = img.resize((256, b), Image.BILINEAR)
        return img
    
    def _pad2(self, img):
        w, h = img.size
        img = TF.pad(img, (0,0,256-w,0), padding_mode='edge') if h == 256 else \
               TF.pad(img, (0,0,0,256-h), padding_mode='edge')
        return img
    
    def _pad(self, img):
        w, h = img.size
        return TF.pad(img, (0,0,256-w,0), fill=255) if h == 256 else \
               TF.pad(img, (0,0,0,256-h), fill=255)
    
    def _transform(self, img):
        return self._pad(self._resize(img))
    
    def _aug_img(self, image):
        if random.random() > 0.5:
            image = TF.rotate(image, random.choice([90, 180, 270]))
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        return image
    
    def _select_img_path(self, unit_path, not_equal_to=None):
        img_path = os.path.join(unit_path, random.choice(os.listdir(unit_path)))
        if not_equal_to:
            while img_path == not_equal_to:
                img_path = os.path.join(unit_path, random.choice(os.listdir(unit_path)))
        return img_path
    
    def __getitem__(self, idx):
        
        unit_path1 = self.unit_paths[idx]
        
        img1_path = self._select_img_path(unit_path1)
        img1 = self._transform(Image.open(img1_path).convert('L'))
        
        label = random.randint(0, 1)
        if label:
            # choose different floorplan
            unit_path2 = unit_path1
            while unit_path2 == unit_path1:
                unit_path2 = random.choice(self.unit_paths)            
            img2 = self._transform(Image.open(self._select_img_path(unit_path2)).convert('L'))
        else:
            # choose similar floorplan
            img2_path = self._select_img_path(unit_path1, not_equal_to=img1_path)
            img2 = self._transform(Image.open(img2_path).convert('L'))
            
        img1, img2 = TF.to_tensor(self._aug_img(img1)), TF.to_tensor(self._aug_img(img2))
        return img1, img2, torch.from_numpy(np.array([label],dtype=np.float32))
    
    
class TriMadoriDataset(MadoriDataset):
    
    def __init__(self, train=True, test=False):
        super().__init__(train=train, test=test)
       
    # choose triplet (anchor, neg, pos) randomly
    def __getitem__(self, idx):
        
        # anchor unit
        unit_path1 = self.unit_paths[idx]
        img1_path = self._select_img_path(unit_path1)
        img1 = self._transform(Image.open(img1_path).convert('L'))
        
        # choose different floorplan
        unit_path2 = unit_path1
        while unit_path2 == unit_path1:
            unit_path2 = random.choice(self.unit_paths)            
        img2 = self._transform(Image.open(self._select_img_path(unit_path2)).convert('L'))
    
        # choose similar floorplan
        img3_path = self._select_img_path(unit_path1, not_equal_to=img1_path)
        img3 = self._transform(Image.open(img3_path).convert('L'))

        # anchor, neg, pos
        return TF.to_tensor(self._aug_img(img1)), TF.to_tensor(self._aug_img(img2)), TF.to_tensor(self._aug_img(img3))