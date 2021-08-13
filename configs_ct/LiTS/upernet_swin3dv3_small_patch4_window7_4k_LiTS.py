_base_ = [
    '../_base_/models/upernet_swin3dv3.py', '../_base_/datasets/LiTS.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]
norm_cfg = dict(type='GN', num_groups=128, requires_grad=True)
conv_cfg = dict(type='Conv3d')
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=((4, 4, 4), (4, 4, 4)),
        shift=((0, 0, 0), (2, 2, 2)),
        dilate=((1, 1, 1), (1, 1, 1)),
        patch_size=(4, 4, 4),
        down_sample_size=(2, 2, 2),
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        use_spectral_aggregation=False,
        in_chans=1
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        channels=256,
        num_classes=3,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg
    ),
    auxiliary_head=dict(
        in_channels=384,
        channels=128,
        num_classes=3,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=0.00001, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,)
