from logging import Logger
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial
import random

import io
import os
import os.path as osp
import shutil
import warnings
from collections.abc import Mapping, Sequence
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import Dataset
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
import os.path as osp
import mmcv
import numpy as np
import torch
import tarfile
from .pipeline import *
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
import pandas as pd
from .tools import splitstr
import clip
from utils.tools import generate_text

PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 ann_file,
                 pipeline,
                 labels_file,
                 animal_labels_file=None,
                 num_animal_classes=None,
                 repeat = 1,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False,):
        super().__init__()
        self.use_tar_format = True if ".tar" in data_prefix else False
        data_prefix = data_prefix.replace(".tar", "")
        self.ann_file = ann_file
        self.labels_file = labels_file
        self.animal_labels_file = animal_labels_file
        self.num_animal_classes = num_animal_classes
        self.repeat = repeat
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    # @abstractmethod
    # def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        # if self.multi_class and isinstance(results['label'], list):
        
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
            
            # token = torch.zeros(1, animal_labels.shape[-1])
            # for i in range(len(results['animal'])):
            #     token = token + animal_labels[results['animal'][i]]
            # token = token/(len(results['animal']))
            
            # print(token.shape)
            
            animal_onehot = torch.zeros(self.num_animal_classes)
            animal_onehot[results['animal']] = 1.
            num = len(results['animal'])
            results['animal'] = animal_onehot
            # results['animal'] = results['animal'] / num
                                                                   
        aug1 = self.pipeline(results)
        if self.repeat > 1:
            aug2 = self.pipeline(results)
            ret = {"imgs": torch.cat((aug1['imgs'], aug2['imgs']), 0),
                    "label": aug1['label'].repeat(2),
            }
            return ret
        else:
            return aug1

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        # if self.multi_class and isinstance(results['label'], list):
        if self.multi_class:
            # onehot = torch.zeros(self.num_classes)
            # onehot[results['label']] = 1.
            # results['label'] = onehot
            
            # onehot = torch.zeros(self.num_animal_classes)
            # onehot[results['animal']] = 1.
            # results['animal'] = onehot
            
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
            
            # token = torch.zeros(1, animal_labels.shape[-1])
            # for i in range(len(results['animal'])):
            #     token = token + animal_labels[results['animal'][i]]
            # token = token/(len(results['animal']))
            
            # print(token.shape)
            
            animal_onehot = torch.zeros(self.num_animal_classes)
            animal_onehot[results['animal']] = 1.
            num = len(results['animal'])
            results['animal'] = animal_onehot
            # results['animal'] = results['animal'] / num
            
        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)

class VideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, multi_class, num_classes, animal_labels_file, num_animal_classes, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, labels_file, animal_labels_file, num_animal_classes, start_index=start_index, **kwargs)
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.num_animal_classes = num_animal_classes
        self.animal_labels_file = animal_labels_file
        # print(self.typee)

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    @property
    def animal_classes(self):
        animal_classes_all = pd.read_csv(self.animal_labels_file)
        return animal_classes_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        # if self.ann_file.endswith('.json'):
        #     return self.load_json_annotations()

        # video_infos = []
        # with open(self.ann_file, 'r') as fin:
        #     for line in fin:
        #         line_split = line.strip().split()
        #         if self.multi_class:
        #             assert self.num_classes is not None
        #             filename, label = line_split[0], line_split[1:]
        #             label = list(map(int, label))
        #         else:
        #             filename, label = line_split
        #             label = label.split(',')
        #             for i in range(len(label)):
        #                 label[i] = int(label[i])
        #         if self.data_prefix is not None:
        #             filename = osp.join(self.data_prefix, filename)
        #         video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))
        # return video_infos
    
    
    # three input: video, label, animal
        print('########### loading data ##########')
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        
        all_classes = pd.read_csv(self.labels_file).values.tolist()
        all_animal_classes = pd.read_csv(self.animal_labels_file).values.tolist()
        a = []
        for i in range(len(all_animal_classes)):
            a.append(all_animal_classes[i][1])
        all_animal_classes = a
        
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                # print(line.strip())
                line_split = line.strip().split("	",1)
                filename, animal_label = line_split[0], splitstr(line_split[1:][0], all_classes)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                
                labels = []
                animals = []
                for i in range(len(animal_label)):
                    labels.append(int(animal_label[i][1]))
                    # print(all_animal_classes)
                    animals.append(all_animal_classes.index(animal_label[i][0]))
                video_infos.append(dict(filename=filename, label=labels, animal = animals, tar=self.use_tar_format))
                                
        return video_infos
        


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def mmcv_collate(batch, samples_per_gpu=1): 
    # if not isinstance(batch, Sequence):
    #     raise TypeError(f'{batch.dtype} is not supported.')
    # if isinstance(batch[0], Sequence):
    #     transposed = zip(*batch)
    #     return [collate(samples, samples_per_gpu) for samples in transposed]
    # elif isinstance(batch[0], Mapping):
    #     return {
    #         key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
    #         for key in batch[0]
    #     }
    # else:
    return default_collate(batch)


def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'animal', 'mid_frame'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]
        
    
    train_data = VideoDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              labels_file=config.DATA.LABEL_LIST, pipeline=train_pipeline, 
                              multi_class=config.DATA.MULTI_CLASSES, num_classes=config.DATA.NUM_CLASSES,
                              num_animal_classes = config.DATA.NUM_ANIMAL_CLASSES, animal_labels_file = config.DATA.ANIMAL_LABEL_LIST)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data, sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=16,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )
    
    print('########### train data loaded ############')
    
    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label', 'animal', 'filename', 'mid_frame'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)
    
    val_data = VideoDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, 
                            labels_file=config.DATA.LABEL_LIST, pipeline=val_pipeline, 
                            multi_class=config.DATA.MULTI_CLASSES, num_classes=config.DATA.NUM_CLASSES,
                            num_animal_classes = config.DATA.NUM_ANIMAL_CLASSES, animal_labels_file = config.DATA.ANIMAL_LABEL_LIST)
    
    print('########### val data loaded ############')
    
    indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=2,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=2),
        #collate_fn=lambda x:x,
    )

    return train_data, val_data, train_loader, val_loader