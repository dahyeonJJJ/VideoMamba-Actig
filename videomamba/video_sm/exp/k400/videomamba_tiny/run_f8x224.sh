export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_tiny_f8_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/ephemeral/kinetics_400/videos_320'
DATA_PATH='/home/ubuntu/jdh/VideoMamba/videomamba/video_sm/data_list/k400'

# PARTITION='video5'
# GPUS=8
# GPUS_PER_NODE=8
# CPUS_PER_TASK=16

torchrun --nproc_per_node=2 \
        run_class_finetuning.py \
        --model videomamba_tiny \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 64 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 8 \
        --num_workers 12 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 70 \
        --lr 2e-4 \
        --drop_path 0.1 \
        --aa rand-m5-n2-mstd0.25-inc1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --bf16 \
        --finetune './exp/k400/videomamba_tiny/checkpoint/videomamba_t16_k400_f8_res224.pth' \
        --resume './exp/k400/videomamba_tiny/checkpoint/videomamba_t16_k400_f8_res224.pth' \
        --eval