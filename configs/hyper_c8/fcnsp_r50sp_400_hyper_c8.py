_base_ = [
    '../_base_/models/fcnsp_r50sp.py', '../_base_/datasets/hyper_c8.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_400.py'
]
norm_cfg = dict(type='BN', track_running_stats=True, requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg,in_channels=8),
    decode_head=dict(num_classes=2,norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2,norm_cfg=norm_cfg))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)