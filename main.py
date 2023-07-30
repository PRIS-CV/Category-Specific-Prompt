import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, all_gather, get_map, get_animal, compute_F1
from utils.visualize import visualize
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending, one_hot
from utils.config import get_config
from models import xclip
import math
import clip
import requests
import coop
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score

from torch.utils.tensorboard import SummaryWriter

from torch.utils import checkpoint

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

writer = SummaryWriter()

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def get_para_num(model):
    lst = []
    for para in model.parameters():
        lst.append(para.nelement())
    print(f"total paras number: {sum(lst)}")
    
    
def get_trainable_para_num(model):
    lst = []
    for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")

def main(config): 
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=True)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)
    
    # print(train_data.classes)
    text_labels = generate_text(train_data.classes) #[140,77]
    animal_labels = generate_text(train_data.animal_classes)
    
    if config.TEST.ONLY_TEST:
        map  = validate(val_loader, val_data, text_labels, animal_labels, model, config, vis=False)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {map:.4f}")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, animal_labels, config, mixup_fn, train_data)

        map = validate(val_loader, val_data, text_labels, animal_labels, model, config, vis=False)
        
        writer.add_scalar('map', map, epoch)
        
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {map:.1f}")
        is_best = map > max_accuracy
        max_accuracy = max(max_accuracy, map)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    map = validate(val_loader, val_data, text_labels, animal_labels, model, config, vis=False)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {map:.4f}")


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, animal_labels, config, mixup_fn, train_data):
    get_para_num(model)
    get_trainable_para_num(model)
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    coop_model = coop.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,)
    state_dict = torch.load('/mnt/sdb/data/jingyinuo/results/animal_kingdom/animal_coop/best.pth')
    coop_model.load_state_dict(state_dict['model'])
    coop_model.to(device)
    coop_model.eval()
        
    texts = text_labels.cuda(non_blocking=True)
    animals = animal_labels.cuda(non_blocking=True)
    
    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        animal_pred = batch_data["animal"].cuda(non_blocking=True)
        mid_frame = batch_data["mid_frame"].cuda(non_blocking=True)
        # label_id = label_id.reshape(-1)
        images = images.view((-1,config.DATA.NUM_FRAMES,3)+images.size()[-2:])
        
        animal_classes = train_data.animal_classes
        animal_pred = get_animal(coop_model, mid_frame, animal_classes, device)
        
        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        
        output = model(images, texts, animal_labels, animal_pred)
        # output = checkpoint.checkpoint(model, images, texts, animal_labels, animal_pred)
        total_loss = criterion(output, label_id)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.requires_grad_(True)
                scaled_loss.backward()
                
        else:
            total_loss.backward()
            
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    writer.add_scalar('loss', tot_loss_meter.avg, epoch)

@torch.no_grad()
def validate(val_loader, val_data, text_labels, animal_labels, model, config, vis=False):
    model.eval()
    
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    map_meter = AverageMeter()
    ani_map_meter = AverageMeter()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load('ViT-B/16', device)
    # resnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
    coop_model = coop.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,)
    state_dict = torch.load('/mnt/sdb/data/jingyinuo/results/animal-best.pth')
    coop_model.load_state_dict(state_dict['model'])
    coop_model.to(device)
    coop_model.eval()
    
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            animal_gt = batch_data["animal"]
            mid_frame = batch_data['mid_frame']
            label_id = label_id.reshape(-1)
            
            animal_classes = val_data.animal_classes
            
            animal_pred = get_animal(coop_model, mid_frame, animal_classes, device)
            # animal_pred = get_animal(clip_model, mid_frame, animal_classes, device)
            
            ani_map_meter.update_predictions(animal_pred, animal_gt)
            # animal = animal.int().unsqueeze(1)
            # animal = torch.mm(similarity, animal_labels).int().unsqueeze(1)
            
            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                label_id = label_id.cuda(non_blocking=True)
                # label_id  = one_hot(label_id, config.DATA.NUM_CLASSES)
                image_input = image.cuda(non_blocking=True)
                animal_labels = animal_labels.cuda(non_blocking=True)
                # print(animal_input1)
                
                animal_pred = animal_pred.cuda(non_blocking=True) 
                # print(animal_input)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                # 从这里开始改 预测动物在训练时也需要改
                output = model(image_input, text_inputs, animal_labels, animal_pred) # + output
                # output = model(image_input, text_inputs, imagenet_labels, animal_pred) # + output
                
                similarity = output.view(b, -1).softmax(dim=-1)
                tot_similarity += similarity
            
            values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            values_5, indices_5 = tot_similarity.topk(5, dim=-1)
            
            ####### 以下为针对多标签进行的修改 #######
            
            label_id = label_id.reshape((b,-1))
            # print(animal_gt.shape)
            
            label_real = []
            bb = []
            
            animal_label_real = []
            bbb = []
            
            label = torch.nonzero(label_id)
            animal_label = torch.nonzero(animal_gt)
            
            for i in range(b):
                for line in label:
                    if line[0] == i:
                        bb.append(line[1])
                label_real.append(bb)
                bb = []
            
            for i in range(b):
                for line in animal_label:
                    if line[0] == i:
                        bbb.append(line[1])
                animal_label_real.append(bbb)
                bbb = []

            acc1, acc5 = 0, 0

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)
            
            
            if vis == True:
                print("######## vis start ########")
                filename = batch_data['filename']
                classes = val_data.classes
                for i in range(len(filename)):
                    label = ''
                    video = '/mnt/sdb/data/jingyinuo/results/animal_kingdom/vis/animal/' + filename[i][45:]
                    print(video)
                    video_read = filename[i]
                    for lab in label_real[i]:
                        label = label + '  ' + classes[int(lab)][1]
                    
                    animal_label = ''
                    for lab in animal_label_real[i]:
                        animal_label = animal_label + '  ' + animal_classes[int(lab)][1]
                    
                    values_5, indices_5 = tot_similarity[i].topk(5, dim=-1)
                    animal_values_5, animal_indices_5 = animal_pred[i].topk(5, dim=-1)
                    pred = ''
                    animal_predd = ''
                    for j in range(len(indices_5)):
                        pred = pred + '  ' + classes[indices_5[j]][1]
                    visualize(video, video_read, pred, label)
                    
                print("######## vis end ########")
            
            
            tot_similarity, label_id = all_gather([tot_similarity, label_id])
            map_meter.update_predictions(tot_similarity, label_id)
            
            if idx % config.PRINT_FREQ == 0:
                logger.info(
                    f'Test: [{idx}/{len(val_loader)}]\t'
                    # f'Acc@1: {acc1_meter.avg:.3f}\t'
                    # f'Map: {map_meter.avg:.3f}\t'
                )
        
    acc1_meter.sync()
    acc5_meter.sync()
    
    map = get_map(torch.cat(map_meter.all_preds).cpu().numpy(), torch.cat(map_meter.all_labels).cpu().numpy())
    ani_map = get_map(torch.cat(ani_map_meter.all_preds).cpu().numpy(), torch.cat(ani_map_meter.all_labels).cpu().numpy())
    print(f'ani_map {ani_map:.4f}')
    
    pred = torch.cat(map_meter.all_preds).cpu().numpy()
    label = torch.cat(map_meter.all_labels).cpu().numpy()
    f13, p3, r3 = compute_F1(3, pred, label, 'overall')
    f15, p5, r5 = compute_F1(3, pred, label, '')

    logger.info(f' * f1@3 {f13:.4f} p3 {p3:.4f} r3 {r3:.4f} f1@5 {f15:.4f} p5 {p5:.4f} r5 {r5:.4f} map {map:.4f}')
    
    return map


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True) 
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)