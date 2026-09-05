CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=8 train.py --batch 8 --experiment_name Color2Embed_1 --datasets --datasets E:\Python_Project\Color2Embed-main\data\train_data\ImageNet_train_lmdb

## you can reload the model for continuing training.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --batch 8 --experiment_name Color2Embed_1 --ckpt experiments/Color2Embed_1/005000.pt --datasets ./train_datasets/ImageNet_train_lmdb
