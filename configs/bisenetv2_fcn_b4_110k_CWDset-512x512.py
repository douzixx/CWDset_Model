
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
        type='BiSeNetV2',
        in_channels=4,
        detail_channels=(64, 64, 128),
        semantic_channels=(16, 32, 64, 128),
        semantic_expansion_ratio=6,
        bga_channels=128,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=None,
        align_corners=False),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=1024,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead', in_channels=16, in_index=1, channels=128,
            num_convs=1, concat_input=False, dropout_ratio=0.1,
            num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead', in_channels=32, in_index=2, channels=128,
            num_convs=1, concat_input=False, dropout_ratio=0.1,
            num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead', in_channels=64, in_index=3, channels=128,
            num_convs=1, concat_input=False, dropout_ratio=0.1,
            num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead', in_channels=128, in_index=4, channels=128,
            num_convs=1, concat_input=False, dropout_ratio=0.1,            num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))


optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=5e-4)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=110000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='PolyLR', eta_min=1e-4, power=0.9, begin=0, end=110000, by_epoch=False)
]


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=2500, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=-1,
        save_best='val/mIoU', rule='greater', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))