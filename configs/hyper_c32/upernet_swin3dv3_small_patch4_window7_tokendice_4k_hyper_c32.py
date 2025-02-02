_base_ = [
    '../_base_/models/upernet_swin3dv3.py', '../_base_/datasets/hyper_c32.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_4k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=((1, 7, 7), (1, 7, 7), (9, 1, 1)),
        shift=((0, 0, 0), (0, 3, 3), (0, 0, 0)),
        dilate=((1, 1, 1), (1, 1, 1), (1, 1, 1)),
        patch_size=(4, 4, 4),
        down_sample_size=(1, 2, 2),
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        in_chans=1,
        use_spectral_aggregation='token'
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='DC_and_CE_loss', loss_weight=1)
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2,
        norm_cfg=norm_cfg,
        loss_decode=dict(type='DC_and_CE_loss', loss_weight=0.4)
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
    samples_per_gpu=2,
    workers_per_gpu=8,)
