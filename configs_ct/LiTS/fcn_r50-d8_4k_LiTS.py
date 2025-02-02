_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/LiTS.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]
# norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN3d', requires_grad=True)
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
conv_cfg = dict(type='Conv3d')
model = dict(
    backbone=dict(
        in_channels=1,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg
    ),
    decode_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.1, 1])
    ),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.1, 1])
    ),
    test_cfg=dict(
        mode='slide',
        stride=(160,160,40),
        crop_size=(320, 320, 80)
    ))


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,)
