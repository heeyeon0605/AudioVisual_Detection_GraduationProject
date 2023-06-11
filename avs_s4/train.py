import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
from torch import optim

from config import cfg
from dataloader import S4Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou
from utils.system import setup_logging
import pdb

from label_train import avsclass

import torch.nn.functional as F
import torch.nn as nn

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

############################################################################################
def Cosine_Similarity(img, audio):
    audio = audio.unsqueeze(0)
    audio = audio.unsqueeze(0)
    
    audio = F.adaptive_avg_pool2d(audio, output_size=(4, 23))
    audio = torch.squeeze(audio)
    
    cs = nn.CosineSimilarity(dim=1, eps=1e-6)

    cs_out = cs(img, audio)
    
    return cs_out

def label_train(data_set, params, newmodel, audio):

    # 1 sample => 5  224 x 224 x 3
    # batch => B x 5 x 224 x 224 x 3
    loss_function=params["loss_function"]
    # eval_dataloader=params["eval_dataloader"]
    # test_dataloader=params["test_dataloader"]
    epochs = params["epoch"]
    optimizer = params["optimizer"]
    ce_loss = params["cross_entropy"]


    print("-"*10, "Epoch {}/{}".format(epoch+1, epochs), "-"*10)
    
    # for i, batch_data in enumerate(train_dataloader):
    #     # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
    #     imgs, audio, mask, data= batch_data

    try:
        inputs = data_set['image'].permute(1,0,2,3,4).contiguous()

        labels = torch.tensor(data_set['label'])
        labels = F.one_hot(labels, num_classes=23).cuda()
        
    except:
        print("Error: ", '\t', labels)

    # 이전 batch에서 계산된 가중치를 초기화
    # optimizer.zero_grad() 
    
    train_loss = 0
    labels = labels.float()
    output1 = newmodel.forward(inputs[0]) # B x 23
    output2 = newmodel.forward(inputs[1])
    output3 = newmodel.forward(inputs[2])
    output4 = newmodel.forward(inputs[3])
    output5 = newmodel.forward(inputs[4])

    
    cs_output1 = Cosine_Similarity(output1, audio)
    cs_output2 = Cosine_Similarity(output2, audio)
    cs_output3 = Cosine_Similarity(output3, audio)
    cs_output4 = Cosine_Similarity(output4, audio)
    cs_output5 = Cosine_Similarity(output5, audio)


    def kl_divergence(aud, img):
            # aud, img is B x 49
            B = aud.shape[0]

            if B == 1:
                return 0
            loss = 10 * \
                torch.sum(aud * torch.log((aud+1e-10) / (img + 1e-10)), 1).mean()

            return loss

    #loss_function =  torch.nn.PairwiseDistance(p=2).to('cuda:0')    # loss function
    
    cross_entropy = (ce_loss(output1, labels) + ce_loss(output2, labels) + ce_loss(output3, labels) + ce_loss(output4, labels) + ce_loss(output5, labels)).mean()
    train_loss = (loss_function(cs_output1, cs_output2) + loss_function(cs_output2, cs_output3) + loss_function(cs_output3, cs_output4) + loss_function(cs_output4, cs_output5)).mean()
    # KL_loss = 

    total_loss = train_loss + cross_entropy

    # print(output1, labels)
    
    print("train_loss: ", train_loss.data.cpu().numpy(), ", cross entropy: ", cross_entropy.data.cpu().numpy(), ", total_loss: ", total_loss.data.cpu().numpy())

    return total_loss

    # print("-"*10, " Change to eval mode. ", "-"*10)
    # # 모델 eval로 전환
    # newmodel.eval()

    # eval_loss = 0
    # eval_cross_entropy = 0

    # for i, batch_data in enumerate(train_dataloader):
    #     imgs, audio, mask, data= batch_data
    #     # train dataloader 로 불러온 데이터에서 이미지와 라벨을 분리
    #     try:
    #         inputs = data['image'].to(device).permute(1,0,2,3,4).contiguous()
            
    #         labels = torch.tensor(data['label']).to(device)
    #     except:
    #         print("Error: ", i, '\t', labels)

    #     # 이전 batch에서 계산된 가중치를 초기화
    #     #optimizer.zero_grad() 
        
    #     output1 = newmodel.forward(inputs[0])
    #     output2 = newmodel.forward(inputs[1])
    #     output3 = newmodel.forward(inputs[2])
    #     output4 = newmodel.forward(inputs[3])
    #     output5 = newmodel.forward(inputs[4])


    #     eval_loss = (loss_function(output1, output2) + loss_function(output2, output3) + loss_function(output3, output4) + loss_function(output4, output5)).mean()
    #     eval_cross_entropy = (ce_loss(output1, labels) + ce_loss(output2, labels) + ce_loss(output3, labels) + ce_loss(output4, labels) + ce_loss(output5, labels)).mean()
        
    #     eval_total_loss = eval_loss + eval_cross_entropy
        
    # if eval_total_loss < min_loss:
    #     min_loss = eval_total_loss

    # return min_loss
############################################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=10, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)


    parser.add_argument('--sa_loss_flag', action='store_true', default=False, help='additional loss for last four frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--sa_loss_stages", default=[], nargs='+', type=int, help='compute sa loss in which stages: [0, 1, 2, 3')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')


    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)

    args = parser.parse_args()


     #   # ######## our code #########

    # device = torch.device('cuda:0')

    # newmodel = avsclass.AVSModelClass()

    lr_ = 0.00002             # learning rate
    loss_function = torch.nn.PairwiseDistance(p=2)
    ce_loss = torch.nn.CrossEntropyLoss()#.to('cuda:0')

    # ###########################

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")


    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    

    # train our code
    train_model = avsclass.AVSModelClass()
    train_model = torch.nn.DataParallel(train_model).cuda()
    train_model.train()
    optimizer = optim.Adam(train_model.parameters(), lr=lr_)   # optimizer

    params = {
        'epoch': args.max_epoches,
        'optimizer':optimizer,
        'lr':lr_,
        'loss_function':loss_function,
        'cross_entropy':ce_loss
    }
    ####################

    # for k, v in model.named_parameters():
    #         print(k, v.requires_grad)

    # video backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_backbone = audio_extractor(cfg, device)
    audio_backbone.cuda()
    audio_backbone.eval()

    # Data
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = S4Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

    # Optimizer
    model_params = model.parameters()
    optimizer_avs = torch.optim.Adam(model_params, args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')


    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    min_loss = 999999999
    for epoch in range(args.max_epoches):

        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask, cropped_imgs= batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            cropped_imgs['image'] = cropped_imgs['image'].cuda()
         
            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B, H, W)
            audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4]) # [B*T, 1, 96, 64]
            with torch.no_grad():
                audio_feature = audio_backbone(audio) # [B*T, 128]

            # our code for label_train
            min_loss = label_train(cropped_imgs, params, train_model, audio_feature)


            output, visual_map_list, a_fea_list = model(imgs, audio_feature) # [bs*5, 1, 224, 224]
            loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1).unsqueeze(1), \
                                                a_fea_list, visual_map_list, \
                                                lambda_1=args.lambda_1, \
                                                count_stages=args.sa_loss_stages, \
                                                sa_loss_flag=args.sa_loss_flag, \
                                                mask_pooling_type=args.mask_pooling_type)

            loss = loss + min_loss 
            print(loss)

            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

            optimizer.zero_grad()
            optimizer_avs.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer_avs.step()

            global_step += 1

            if (global_step-1) % 50 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lambda_1:%.4f, lr: %.4f'%(
                            global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), args.lambda_1, optimizer.param_groups[0]['lr'])
                # train_log = ['Iter:%5d/%5d' % (global_step - 1, max_step),
                #         'Total_Loss:%.4f' % (avg_meter_loss.pop('total_loss')),
                #         'iou_loss:%.4f' % (avg_meter_iou_loss.pop('iou_loss')),
                #         'sa_loss:%.4f' % (avg_meter_sa_loss.pop('sa_loss')),
                #         'lambda_1:%.4f' % (args.lambda_1),
                #         'lr: %.4f' % (optimizer.param_groups[0]['lr'])]
                # print(train_log, flush=True)
                logger.info(train_log)


        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                imgs, audio, mask, _, _ = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B*frame, C, H, W)
                mask = mask.view(B*frame, H, W)
                audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
                with torch.no_grad():
                    audio_feature = audio_backbone(audio)

                output, _, _ = model(imgs, audio_feature) # [bs*5, 1, 224, 224]


                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(checkpoint_dir, '%s_best.pth'%(args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s'%model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            # print(val_log)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))
