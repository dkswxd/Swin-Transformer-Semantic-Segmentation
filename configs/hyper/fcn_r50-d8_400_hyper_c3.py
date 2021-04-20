_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/hyper_c3.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_400.py'
]
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=2,norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2,norm_cfg=norm_cfg))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,)