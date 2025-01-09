# Updated SOLOv2 Configuration for Custom Dataset

_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'C:/Users/boaz-/Documents/TUE/00 - Afstuderen/14 - Traindata/'

backend_args = None

classes = ('Bubble') 

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train_annotations_reduced.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=4),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val_annotations_reduced.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=4),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val_annotations_reduced.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator



# Pretrained weights
load_from = "C:/Users/boaz-/Documents/TUE/00 - Afstuderen/14 - Traindata/pretrained_models/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth"
resume = False

# Model settings
model = dict(
    type='SOLOv2',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),
    mask_head=dict(
        type='SOLOV2Head',
        num_classes=1,  # Replace with the number of object classes in your datasett 
        init_cfg=None,  # Prevent loading COCO weights
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_mask=dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0)),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.1,
        mask_thr=0.5,
        filter_thr=0.05,
        kernel='gaussian',
        sigma=2.0,
        max_per_img=100))


# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate schedule
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[27, 33], gamma=0.1)
]


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),  # Log every 50 iterations
    checkpoint=dict(type='CheckpointHook', interval=1)  # Save every epoch
)

log_level = 'INFO'

# Training settings
max_epochs = 50
train_cfg = dict(max_epochs=max_epochs)

