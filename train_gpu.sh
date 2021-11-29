


# for pedestrian attribute recognition

CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/peta.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/peta_zs.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/rapv1.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/rapv2.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/rap_zs.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --cfg ./configs/pedes_baseline/pa100k.yaml

# for swin transformer, change cfg.TRAIN.BATCH_SIZE: 32
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1233 train.py --cfg ./configs/pedes_baseline/pa100k.yaml




# for multi-label classification
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1233 train.py --cfg ./configs/multilabel_baseline/coco.yaml


