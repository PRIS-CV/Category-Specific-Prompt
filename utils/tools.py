import numpy as np
import torch.distributed as dist
import torch
from torchvision.models.resnet import resnet50
from torchvision import transforms
import clip
import os
from sklearn.metrics import average_precision_score
from PIL import Image
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score

def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor

def reduce_tensor(tensor, n=None):
    if n is None:
        n = dist.get_world_size()
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / n
    return rt
   

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.all_preds = []
        self.all_labels = []
        
    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def sync(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        val = torch.tensor(self.val).cuda()
        sum_v = torch.tensor(self.sum).cuda()
        count = torch.tensor(self.count).cuda()
        self.val = reduce_tensor(val, world_size).item()
        self.sum = reduce_tensor(sum_v, 1).item()
        self.count = reduce_tensor(count, 1).item()
        self.avg = self.sum / self.count


def epoch_saving(config, epoch, model,  max_accuracy, optimizer, lr_scheduler, logger, working_dir, is_best):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(working_dir, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if is_best:
        best_path = os.path.join(working_dir, f'best.pth')
        torch.save(save_state, best_path)
        logger.info(f"{best_path} saved !!!")


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    if os.path.isfile(config.MODEL.RESUME): 
        logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        load_state_dict = checkpoint['model']

        msg = model.load_state_dict(load_state_dict, strict=False)
        logger.info(f"resume model: {msg}")

        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            start_epoch = checkpoint['epoch'] + 1
            max_accuracy = checkpoint['max_accuracy']

            logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
            
            del checkpoint
            torch.cuda.empty_cache()

            return start_epoch, max_accuracy
        except:
            del checkpoint
            torch.cuda.empty_cache()
            return 0, 0.

    else:
        logger.info(("=> no checkpoint found at '{}'".format(config.MODEL.RESUME)))
        return 0, 0


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def generate_text(data):
    text_aug = f"{{}}"
    classes = torch.cat([clip.tokenize(text_aug.format(c), context_length=77) for i, c in data])

    return classes

def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    https://github.com/facebookresearch/SlowFast/blob/2090f2918ac1ce890fdacd8fda2e590a46d5c734/slowfast/utils/meters.py#L231
    """
    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    # labels = labels.astype('int')
    aps = [0]
    print(preds.shape)
    print(labels.shape)
    # try:
    aps = average_precision_score(labels, preds, average=None)
    # except ValueError:
    #     print(
    #         "Average precision requires a sufficient number of samples \
    #         in a batch which are missing in this sample."
    #     )
    print(aps)
    mean_ap = np.mean(aps)
    return mean_ap

# def get_animal(image, ):
#     ### 抽取中间一帧 采用CLIP预训练模型得到动物预测结果 ###
    
def get_animal(model, image_input, label, device):
    # Load the model
    # dirpath = image_route + image[idx]
    # image_input = np.transpose(image_input, (0, 3, 1, 2))
    image_input = image_input.to(device)
    # text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in label]).to(device)

    # Calculate features
    with torch.no_grad():
    #     image_features = model.encode_image(image_input)
    #     text_features = model.encode_text(text_inputs)

    # # Pick the top most similar labels for the image
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
        output = model(image_input)
        similarity = output.softmax(dim=-1)
    # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # print(similarity.shape)
    
    values, indices = similarity[0].topk(len(label))

    # Re-arrange the prediction tensor
    new_values = torch.zeros(len(label))
    new_indices = indices.tolist()
    for i in range(len(label)):
        new_values[i] = values[new_indices.index(i)]
    # get_map_preds[idx,:] = new_values

    # Print the result
    # for value, index in zip(values, indices):
        # print(f"{label[index]}: {100 * value.item():.2f}%")
        
    return similarity

    # map = get_map(get_map_preds.numpy(), get_map_labels.numpy())
    # print(map)
    
# def get_animal(model, image_input, label):
#     inp = Image.fromarray(inp.astype('uint8'), 'RGB')
#     inp = transforms.ToTensor()(inp).unsqueeze(0)
#     with torch.no_grad():
#         prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
#     return {labels[i]: float(prediction[i]) for i in range(1000)}

def compute_F1(k,predictions,labels,mode_F1):
    idx = np.argsort(predictions,axis = 1)
    for i in range(predictions.shape[0]):
        predictions[i][idx[i][-k:]]=1
        predictions[i][idx[i][:-k]]=0
        
    if mode_F1 == 'overall':
        print('evaluation overall!! cannot decompose into classes F1 score')
        mask = predictions == 1
        TP = np.sum(labels[mask]==1)
        p = TP/np.sum(mask)
        r = TP/np.sum(labels==1)
        f1 = 2*p*r/(p+r)
        
#        p_2,r_2,f1_2=compute_F1_fast0tag(predictions,labels)
    else:
        num_class = predictions.shape[1]
        print('evaluation per classes')
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r  = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:,idx_cls])
            label = np.squeeze(labels[:,idx_cls])
            if np.sum(label>0)==0:
                continue
            binary_label=np.clip(label,0,1)
            f1[idx_cls] = f1_score(binary_label,prediction)#AP(prediction,label,names)
            p[idx_cls] = precision_score(binary_label,prediction)
            r[idx_cls] = recall_score(binary_label,prediction)
        f1 = np.mean(f1)
        p = np.mean(p)
        r = np.mean(r)
    return f1,p,r