dataset_type = 'CWDset'
data_root = 'data/CWDset/' 


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


val_pipeline = [
    dict(type='LoadTiffImageFromFile', to_float32=True),
    dict(type='LoadTiffAnnotations', reduce_zero_label=False), 
    dict(type='PackSegInputs')
]

test_pipeline = val_pipeline
