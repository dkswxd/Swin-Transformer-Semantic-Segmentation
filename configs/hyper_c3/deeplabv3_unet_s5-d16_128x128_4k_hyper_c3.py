_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/hyper_c3.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg,with_cp=True),
    decode_head=dict(num_classes=2,norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2,norm_cfg=norm_cfg),
    test_cfg=dict(crop_size=(128, 128), stride=(85, 85)))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
evaluation = dict(metric='mIoU')
