
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
        type='MixVisionTransformer',
        in_channels=4,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=110000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=110000, by_epoch=False)
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