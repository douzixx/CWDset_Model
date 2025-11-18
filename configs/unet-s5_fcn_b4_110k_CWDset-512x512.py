
custom_imports = dict(imports=['CWDset_code.data_pipelines', 'CWDset_code.CWD_dataset'], allow_failed_imports=False)

_base_ = [
    '_base_/datasets/CWDset_base.py',
    '_base_/default_runtime.py'
]
default_scope = 'mmseg'


num_classes = 2
crop_size = (512, 512)
train_dataloader = dict(
    batch_size=4, num_workers=4, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=_base_.dataset_type, data_root=_base_.data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='labels/train'),
        pipeline=_base_.train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=_base_.dataset_type, data_root=_base_.data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='labels/val'),
        pipeline=_base_.val_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice','mFscore'], prefix='val')
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice','mFscore'], prefix='test')


data_preprocessor = dict(
    type='SegDataPreProcessor', size=crop_size,
    mean=[0.0118, 0.0148, 0.0153, 0.0321],
    std=[0.00621, 0.00612, 0.00750, 0.0125],
    bgr_to_rgb=False, pad_val=0, seg_pad_val=255)

norm_cfg = dict(type='SyncBN', requires_grad=True)


model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=4,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[1,15])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,class_weight=[1,15])),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=110000, val_interval=2500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=110000, by_epoch=False)
]


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10000, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=10000,
        save_best='val/mIoU', rule='greater', max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))