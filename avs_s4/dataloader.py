import os
from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from config import cfg
import pdb

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel


class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train'):
        super(S4Dataset, self).__init__()
        self.split = split
        self.mask_num = 1 if self.split == 'train' else 5
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index): # return_video_name=False
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]

        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, self.split, category, video_name)
        audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, category, video_name)
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        #####################################
        img_base_path_for_cropped = []
        cropped_txt_path = []
        #####################################
        imgs, masks = [], []

        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
            
            #### WHAT I ADDED 0523 ####
            img_base_path_for_cropped.append(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)))
            cropped_txt_path.append(os.path.join(img_base_path, "%s_%d.txt"%(video_name, img_id)))
        if len(img_base_path_for_cropped) != 5:
            print(img_base_path_for_cropped)
            assert False
        # for i in img_base_path_for_cropped:
        #     print("one time each       ", i)
        
        # notes: cropped_txt_path는 리스트라 1~5까지 txt path 저장, img_path는 하나의 path 담는 변수라 5 png path만 저장.
        # print("<<<<<<<<<<one pair>>>>>>>>>")
        # print(img_path)
        # print(cropped_txt_path)

        # cropped_image = self.get_cropped_image(img_base_path, cropped_txt_path)

        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)
        #####################################################
        
        imgs_dict = {}
        cropped_imgs = []
        # label = 0
        # for i in crop_path:
        #     if "classes.txt" in i:
        #         crop_path.remove(i)
        
        # for i in img_base_path_for_cropped:
        #     print(i)

        # assert False 

        for i in range(5):

            # img_temp = img_base_path + "_" + str(i+1) + ".png"

            c = img_base_path_for_cropped[i] # txt_temp라 하려다가 임시로 c

            if "ambulance_siren" in c :
                label = 0
            elif "baby_laughter" in c :
                label = 1
            elif "cap_gun_shooting" in c :
                label = 2 
            elif "cat_meowing" in c :
                label = 3
            elif "chainsawing_trees" in c:
                label = 4
            elif "coyote_howling" in c:
                label = 5
            elif "dog_barking" in c:
                label = 6
            elif "driving_buses" in c:
                label = 7
            elif "female_singing" in c:
                label = 8
            elif "helicopter" in c:
                label = 9
            elif "horse_clip-clop" in c:
                label = 10
            elif "lawn_mowing" in c:
                label = 11
            elif "lions_roaring" in c:
                label = 12
            elif "male_speech" in c:
                label = 13
            elif "mynah_bird_singing" in c:
                label = 14
            elif "playing_acoustic_guitar" in c:
                label = 15
            elif "playing_glockenspiel" in c:
                label = 16
            elif "playing_piano" in c:
                label = 17
            elif "playing_tabla" in c:
                label = 18
            elif "playing_ukulele" in c:
                label = 19
            elif "playing_violin" in c:
                label = 20
            elif "race_car" in c:
                label = 21
            elif "typing_on_computer_keyboard" in c:
                label = 22

            cropped_imgs.append(self.get_cropped_image(c, cropped_txt_path[i]))
    
        while(len(cropped_imgs) != 5):
            cropped_imgs.append(cropped_imgs[-1])
        cropped_imgs = torch.stack(cropped_imgs, dim=0)
        
        imgs_dict = {"image": cropped_imgs, "label": label}

        #####################################################
        if self.split == 'train':
            return imgs_tensor, audio_log_mel, masks_tensor, imgs_dict#, (c, cropped_txt_path[i], label)
        else:
            return imgs_tensor, audio_log_mel, masks_tensor, category, video_name


    def __len__(self):
        return len(self.df_split)

    ############### FROM OUR DATA_LOADER.PY CODE ###############
    def crop_image(self, img, crop_size):
        (left, top, right, bottom) = crop_size
        # Crop and resize the image using the bounding box coordinates
        crop_img = TF.crop(img, top=top, left=left, height=bottom-top, width=right-left)
        resized_img = TF.resize(crop_img, size=(224, 224))

        # Tensor 값 출력하기
        tensor_img = TF.to_tensor(resized_img)

        return tensor_img

    def get_cropped_image(self, image_path, crop_path):
        img = Image.open(image_path).convert("RGB")
        # print(image_path)
        # assert False
        if os.path.exists(crop_path):
            with open(crop_path, "r") as f:
                try:
                    x, y, w, h = [float(x) for x in f.readline().strip().split()[1:]]
                except:
                    print("except!!!")
                    x, y, w, h = 0,0,0,0
        else:
            x, y, w, h = 0, 0, 0, 0

        # Convert the coordinates to pixel values
        width, height = img.size
        left = int((x - w/2) * width)
        top = int((y - h/2) * height)
        right = int((x + w/2) * width)
        bottom = int((y + h/2) * height)

        img = self.crop_image(img, (left, top, right, bottom))

        return img
    ############### END OF OUR DATA_LOADER.PY CODE ###############



if __name__ == "__main__":
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask, cropped_imgs = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        # imgs, audio, mask, category, video_name = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()
