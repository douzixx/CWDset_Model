# 文件: pythonProject/configs/_base_/default_runtime.py
#
# 这是一个“纯Python” (non-lazy) 版本的
# default_runtime.py，用来解决你的 RuntimeError

# 默认钩子 (Hooks)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    # 你的主配置会用 save_best='mIoU' 来覆盖这个
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 默认环境配置
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# 可视化后端
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 日志处理器
log_processor = dict(by_epoch=False)
# 日志级别
log_level = 'INFO'

# 加载和恢复
load_from = None
resume = False