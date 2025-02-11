# model settings
model = dict(
    type='ReDet',
    backbone=dict(
        type='ReResNet',
        pretrained='work_dirs/ReResNet_pretrain/re_resnet50_c8_batch256-25b16846.pkl',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='jittor'),
    neck=dict(
        type='ReFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='FasterrcnnHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLossForRcnn', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=2, version=1),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=16,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=True,
        with_module=False,
        loss_cls=dict(
            type='CrossEntropyLossForRcnn', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    rbbox_roi_extractor=dict(
        type='RboxSingleRoIExtractor',
        roi_layer=dict(type='RiRoIAlign', output_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    rbbox_head = dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=16,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLossForRcnn', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D_v1')),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D_v1')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssignerRbbox',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                sampler=dict(
                    type='RandomSamplerRotated',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg = dict(
        rpn=dict(
            # TODO: test nms 2000
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr = 0.05, nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img = 2000)
    ),
    )
# model training and testing settings
# dataset settings
dataset_type = 'DOTADataset'
dataset = dict(
    train=dict(
        type=dataset_type,
        images_dir='/mnt/disk/flowey/dataset/DOTA_1024/trainval_split/images/',
        annotations_file='/mnt/disk/flowey/dataset/DOTA_1024/trainval_split/trainval1024.pkl',
        version='1',
        filter_min_size=32,
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "RotatedRandomFlip",
                prob = 0.5,
                direction="horizontal",
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=2,
        shuffle=True,
        ),
    val=dict(
        type=dataset_type,
        images_dir='/mnt/disk/flowey/dataset/DOTA_1024/trainval_split/images/',
        annotations_file='/mnt/disk/flowey/dataset/DOTA_1024/trainval_split/trainval1024.pkl',

        version='1',
        filter_min_size=32,
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        ),
    test=dict(
        type="ImageDataset",        
        images_dir='/mnt/disk/flowey/dataset/DOTA_1024/test_split/images/',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=1,
        shuffle=False,
    )
)
# optimizer
optimizer = dict(
    type='SGD', 
    lr=0.01, 
    momentum=0.9, 
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=35, 
        norm_type=2)
)
# learning policy
scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[8, 11])

logger = dict(
    type="RunLogger")
max_epoch = 12
eval_interval = 1
checkpoint_interval = 1
log_interval = 20
# pretrained_weights = '/mnt/disk2/flowey/JRS/work_dirs/ReDet_re50_refpn_1x_dota1/epoch_12.pkl'

# map: 0.7623183203127312
# classaps: plane:0.8912137088763317, baseball-diamond:0.8227656853636833, bridge:0.5438046929395232,
# ground-track-field:0.741438878333126, small-vehicle:0.7792042787717315, large-vehicle:0.8293264252845306,
# ship:0.8816971776117367, tennis-court:0.9082554204483129, basketball-court:0.8708274706937482,
# storage-tank:0.8588086316992405, soccer-ball-field:0.5960834417486613, roundabout:0.6243493262883512,
# harbor:0.7679488008566184, swimming-pool:0.6758004726927681, helicopter:0.6432503930826027


