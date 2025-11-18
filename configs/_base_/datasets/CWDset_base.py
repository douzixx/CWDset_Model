dataset_type = 'CWDset'
data_root = 'data/CWDset/' # 假设你的数据在 'pythonProject/data/CWDset/'

# 你的 TIF 训练管道
train_pipeline = [
    dict(type='LoadTiffImageFromFile', to_float32=True),
    dict(type='LoadTiffAnnotations', reduce_zero_label=False),
    dict(type='RandomResizeTiff',
         scale=(1.0, 1.5),
         interpolation='bilinear'),
    dict(type='RandomCropTiff',
         crop_size=(512, 512)),
    dict(type='RandomFlipTiff',
         prob=0.5,
         direction='horizontal'),
    dict(type='RandomFlipTiff',
         prob=0.5,
         direction='vertical'),
    dict(type='PackSegInputs')
]

# 你的 TIF 验证/测试管道
val_pipeline = [
    dict(type='LoadTiffImageFromFile', to_float32=True),
    dict(type='LoadTiffAnnotations', reduce_zero_label=False), # 必须加载标签才能评估
    dict(type='PackSegInputs')
]

test_pipeline = val_pipeline