# dataset settings
dataset_type = 'hyper'
data_root = '../data/hyper'
img_norm_cfg = dict(
    mean=[128]*32, std=[16]*32, to_rgb=False)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadENVIHyperSpectralImageFromFile',channel_select=range(5,37)),#[5, 10, 15, 20, 25, 30, 35, 40]
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadENVIHyperSpectralImageFromFile',channel_select=range(5,37)),#[5, 10, 15, 20, 25, 30, 35, 40]
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='hdr_dir',
        ann_dir='ann_dir',
        split='split_dir/split_hsy_train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='hdr_dir',
        ann_dir='ann_dir',
        split='split_dir/split_hsy_val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='hdr_dir',
        ann_dir='ann_dir',
        split='split_dir/split_hsy_test.txt',
        pipeline=test_pipeline))
