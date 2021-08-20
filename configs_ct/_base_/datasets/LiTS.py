# dataset settings
dataset_type = 'LiTS'
data_root = '../data/LiTS'

crop_size = (320, 320, 160)
train_pipeline = [
    dict(type='NiiLoadImageFromFile'),
    dict(type='NiiLoadAnnotationsFromFile',transpose='dhw2hwd',lits_remove_turmor=True),
    dict(type='NiiClipImageValue',min_value=-200,max_value=200),
    dict(type='NiiRemoveSlice',expand=20),
    dict(type='NiiResize', ratio_range=(0.5, 1.5), target_spacing=(1, 1, 1)),
    dict(type='NiiRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='NiiNormalizeImage'),
    dict(type='NiiRandomFlip', prob=0.5),
    dict(type='NiiPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='NiiDefaultFormatBundle'),
    dict(type='NiiCollect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='NiiLoadImageFromFile'),
    dict(type='NiiClipImageValue',min_value=-200,max_value=200),
    dict(
        type='NiiMultiScaleFlipAug',
        img_scale=None,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='NiiResize'),
            dict(type='NiiNormalizeImage'),
            dict(type='NiiRandomFlip'),
            dict(type='NiiImageToTensor', keys=['img']),
            dict(type='NiiCollect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Training',
        ann_dir='Training',
        split='train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Training',
        ann_dir='Training',
        split='val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Training',
        split='val.txt',
        pipeline=test_pipeline))