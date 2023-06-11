import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import glob
import numpy as np

class AVSDataset(Dataset):
    def __init__(self, mode, img_files, audio_files, img_path, audio_path, audio_transform=None):
        super().__init__()
        self.mode = mode
        self.audio_path = audio_path
        self.img_path = open(img_path, 'r').readlines()
        
        for i in range(len(self.img_path)):
            self.img_path[i] = self.img_path[i].strip()

        self.audio_files = audio_files
        self.img_files = img_files

        self.audio_transform = audio_transform

        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        # mask
        if self.mode == 'train':
            self.img_transform = transforms.Compose([transforms.Resize((224,224))])
        elif self.mode == 'eval':
            self.img_transform = transforms.Compose([transforms.Resize((224,224))])
        else: # 나중에 train 말고 test쓸 때 고쳐야 할 부분
            self.img_transform = transforms.Compose([transforms.Resize(())])

    def __getitem__(self, idx):
        # Image
        images = glob.glob(self.img_path[idx] + '/*.png')

        crop_path = glob.glob(self.img_path[idx] + '/*.txt')
        sorted(images)
        sorted(crop_path)
        
        # getitem 할 때 마다 5장씩만 idx로 가져오는데 매번 리스트랑 딕셔너리가 리셋되는게 문제 -> main에서 for loop로 해결
        imgs_dict = {}
        imgs = []
        label = 0
        for i in crop_path:
            if "classes.txt" in i:
                crop_path.remove(i)
        
        #  Class 23개
        for (i, c) in zip(images, crop_path):

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
            imgs.append(self.get_image(i, i[:-4] + ".txt"))

        while(len(imgs) != 5):
            imgs.append(imgs[-1])
        imgs = torch.stack(imgs, dim=0)

        imgs_dict = {"image": imgs, "label": label}

        # try:
        #     imgs = torch.stack(imgs, dim=0)
        #     if imgs.shape[-4] != 5:
        #         torch.cat(imgs, imgs[imgs.shape[-4]-1], dim=0)
        #     imgs_dict = {"image": imgs, "label": label}
        # except:
        #     print("No bounding box coordinate file found. Setting coordinates to zero.")

        # 이미지 5개씩, 라벨은 1개 들은 딕셔너리 반환
        return imgs_dict
    
    def __len__(self):
        return len(self.img_path)
    
    def get_image(self, image_path, crop_path):
        img = Image.open(image_path).convert("RGB")
        
        if os.path.exists(crop_path):
            with open(crop_path, "r") as f:
                try:
                    x, y, w, h = [float(x) for x in f.readline().strip().split()[1:]]
                except:
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

    def crop_image(self, img, crop_size):
        (left, top, right, bottom) = crop_size
        # Crop and resize the image using the bounding box coordinates
        crop_img = TF.crop(img, top=top, left=left, height=bottom-top, width=right-left)
        resized_img = TF.resize(crop_img, size=(224, 224))

        # Tensor 값 출력하기
        tensor_img = TF.to_tensor(resized_img)

        return tensor_img



# if __name__ == "__main__":
#     img_path = '/shared_dataset/avsbench_data/tsne_info.txt'
#     dataset = AVSDataset('train','','',img_path,'')
#     # print(dataset.img_path)
#     # img_files = os.listdir(img_path)
#     # dataset = AVSDataset('train', "", "", "", "")
    
#     total_dicts = [] # 일단 임의로 getitem으로부터 가져온 dict를 list에 추가. list안에 dict안에 tensor와 int여서 구조 오바면 리스트 말고 다른걸로 가져와도 좋아요. 

#     for idxs in range(80): # 총 이미지파일개수/5 로 해야하는데 임의로 80으로 해둠
#         total_dicts.append(dataset.__getitem__(idxs))

#     print(total_dicts) # 출력 하다보면 잘되다가 갑자기 멈춤 ... "ValueError: not enough values to unpack (expected 4, got 0)" 중간에 coordinates에 들어갈 때 이상한 값이 있나봐요 ...
