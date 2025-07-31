export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_tiny_f32_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/data/data_dhjung/coin/COIN_video'
DATA_PATH='/home/oem/users/dhjung/VideoMamba/videomamba/video_sm/data_list/coin' 

# PARTITION='video5'
# GPUS=8
# GPUS_PER_NODE=8
# CPUS_PER_TASK=16

torchrun --nproc_per_node=4 \
        run_class_finetuning.py \
        --model videomamba_tiny \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 180 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 32 \
        --orig_t_size 32 \
        --num_workers 16 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 40 \
        --lr 2e-4 \
        --drop_path 0.1 \
        --aa rand-m5-n2-mstd0.25-inc1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --test_num_segment 10 \
        --test_num_crop 3 \
        --dist_eval \
        --bf16 \
        --finetune './exp/coin/videomamba_tiny/checkpoint/videomamba_t16_coin_f32_res224.pth' \
        --resume './exp/coin/videomamba_tiny/checkpoint/videomamba_t16_coin_f32_res224.pth' \
        --eval
