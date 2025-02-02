_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/pascal_context.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(norm_cfg=norm_cfg,),
    decode_head=dict(num_classes=60,norm_cfg=norm_cfg,),
    auxiliary_head=dict(num_classes=60,norm_cfg=norm_cfg,),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
