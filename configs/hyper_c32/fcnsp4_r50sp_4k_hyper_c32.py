_base_ = [
    '../_base_/models/fcnsp_r50sp.py', '../_base_/datasets/hyper_c32.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]
norm_cfg = dict(type='BN', track_running_stats=True, requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg,in_channels=3),
    decode_head=dict(num_classes=2,norm_cfg=norm_cfg,num_sp=4,sp_s=8),
    auxiliary_head=dict(num_classes=2,norm_cfg=norm_cfg,num_sp=2,sp_s=8))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)