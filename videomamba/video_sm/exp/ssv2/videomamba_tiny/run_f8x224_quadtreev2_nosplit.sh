export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='quadtreev2_nosplit_p14_d4'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/data/data_dhjung/20bn-something-something-v2'
DATA_PATH='/home/oem/users/dhjung/VideoMamba/videomamba/video_sm/data_list/sthv2'

# PARTITION='video5'
# GPUS=8
# GPUS_PER_NODE=8
# CPUS_PER_TASK=16

torchrun --nproc_per_node=4 \
        run_class_finetuning.py \
        --model videomamba_tiny_quadtreev2_nosplit_p14_d4 \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'SSV2' \
        --nb_classes 174 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 24 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 8 \
        --num_workers 12 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 35 \
        --lr 4e-4 \
        --drop_path 0.1 \
        --aa rand-m5-n2-mstd0.25-inc1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --test_num_segment 2 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16 \