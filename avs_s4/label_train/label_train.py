import torch
from torch.utils.data import TensorDataset
from label_train import data_loader
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.nn.functional import normalize as normalize
import torchvision.models as models
import torchvision.transforms as transforms
from label_train import avsclass
import torch.nn.functional as F


class train_label():

    
    def __init__(self):
        self.min_loss = 9999999999
        self.train_model = avsclass.AVSModelClass()

        self.img_path = '/shared_dataset/avsbench_data/tsne_info.txt'
        self.train_data = DataLoader(data_loader.AVSDataset('train', '', '', self.img_path, ''), batch_size=4, shuffle=True)
        self.eval_data = DataLoader(data_loader.AVSDataset('eval', '', '', self.img_path, ''), batch_size=4, shuffle=True)

        # train_labels = torch.tensor([0, 1, 2, 3, 4, 5]) # 앰뷸, 베이비, 캡건, 캣미야오
        # num_classes = 23
        self.device = torch.device('cuda:0')


        self.lr = 0.0001             # learning rate
        self.epoch = 5          # total number of epochs
        self.optimizer = optim.Adam(self.train_model.resnet.parameters(), lr=self.lr)   # optimizer
        self.loss_function =  torch.nn.PairwiseDistance(p=2).to('cuda:0')    # loss function
        self.ce_loss = torch.nn.CrossEntropyLoss().to('cuda:0')

        self.params = {
            'epoch':self.epoch,
            'optimizer':self.optimizer,
            'lr':self.lr,
            'loss_function':self.loss_function,
            'cross_entropy':self.ce_loss,
            'train_dataloader':self.train_data,
            'eval_dataloader':self.eval_data,
            'device': self.device
        }


    def get_minloss(self):
        return self.min_loss

    def train(self, data_set):
        loss_function=self.params["loss_function"]
        train_dataloader = data_set
        eval_dataloader=self.params["eval_dataloader"]
        # test_dataloader=params["test_dataloader"]
        device=self.params["device"]
        epochs = self.params["epoch"]
        optimizer = self.params["optimizer"]
        ce_loss = self.params["cross_entropy"]
        newmodel = avsclass.AVSModelClass()


        for epoch in range(0, epochs):
            # 모델 train으로 전환
            newmodel.train()

            print("-"*10, "Epoch {}/{}".format(epoch+1, epochs), "-"*10)
            

            for i, batch_data in enumerate(train_dataloader):
                # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
                imgs, audio, mask, data= batch_data
                try:
                    inputs = data['image'].to(device).permute(1,0,2,3,4).contiguous()
                    
                    labels = torch.tensor(data['label']).to(device)
                    labels = F.one_hot(labels, num_classes=23).to(device)
                    
                except:
                    print("Error: ", i, '\t', labels)

                # 이전 batch에서 계산된 가중치를 초기화
                optimizer.zero_grad() 
                
                train_loss = 0
                labels = labels.float()
                cross_entropy = 0

                output1 = newmodel.forward(inputs[0])
                output2 = newmodel.forward(inputs[1])
                output3 = newmodel.forward(inputs[2])
                output4 = newmodel.forward(inputs[3])
                output5 = newmodel.forward(inputs[4])
                
                train_loss = (loss_function(output1, output2) + loss_function(output2, output3) + loss_function(output3, output4) + loss_function(output4, output5)).mean()
                cross_entropy = (ce_loss(output1, labels) + ce_loss(output2, labels) + ce_loss(output3, labels) + ce_loss(output4, labels) + ce_loss(output5, labels)).mean()
                
                total_loss = train_loss + cross_entropy
                
                
                cross_entropy.backward()
                
                optimizer.step()

                print("train_loss: ", train_loss.data.cpu().numpy(), ", cross entropy: ", cross_entropy.data.cpu().numpy(), ", total_loss: ", total_loss.data.cpu().numpy())

            print("-"*10, " Change to eval mode. ", "-"*10)
            # 모델 eval로 전환
            newmodel.eval()

            eval_loss = 0
            eval_cross_entropy = 0

            for i, batch_data in enumerate(train_dataloader):
                imgs, audio, mask, data= batch_data
                # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
                try:
                    inputs = data['image'].to(device).permute(1,0,2,3,4).contiguous()
                    
                    labels = torch.tensor(data['label']).to(device)
                except:
                    print("Error: ", i, '\t', labels)

                # 이전 batch에서 계산된 가중치를 초기화
                #optimizer.zero_grad() 
                
                output1 = newmodel.forward(inputs[0])
                output2 = newmodel.forward(inputs[1])
                output3 = newmodel.forward(inputs[2])
                output4 = newmodel.forward(inputs[3])
                output5 = newmodel.forward(inputs[4])


                eval_loss = (loss_function(output1, output2) + loss_function(output2, output3) + loss_function(output3, output4) + loss_function(output4, output5)).mean()
                eval_cross_entropy = (ce_loss(output1, labels) + ce_loss(output2, labels) + ce_loss(output3, labels) + ce_loss(output4, labels) + ce_loss(output5, labels)).mean()
                
                eval_total_loss = eval_loss + eval_cross_entropy
                
            if eval_total_loss < self.min_loss:
                self.min_loss = eval_total_loss
                # torch.save
                save_path = '/shared_dataset/avsbench_data/weights_jisu/'
                try: 
                    torch.save(newmodel, save_path + 'model.pt') # 전체 모델 저장
                    torch.save(newmodel.state_dict(), save_path + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
                    torch.save(optimizer.state_dict(), save_path + 'optimizer_state_dict.pt') # optimizer 객체의 state_dict 저장
                    torch.save({
                        'loss': min_loss,
                        'epoch': epoch,
                        'model': newmodel.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'device': device,
                        'loss_function': loss_function,
                    }, save_path + 'all.tar') # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능
                    print("-"*10, "Epoch {}/{} is the best epoch".format(epoch+1, epochs), "-"*10)
                    print("The best loss is ", min_loss)
                    print("<<<<<<<<<<<torch saving completed>>>>>>>>>>>>")
                except: 
                    print("<<<<<<<<<<<pt file saving NOT completed>>>>>>>>>>>>")

        print("The best loss is: ", self.min_loss.data.cpu().numpy())
        

