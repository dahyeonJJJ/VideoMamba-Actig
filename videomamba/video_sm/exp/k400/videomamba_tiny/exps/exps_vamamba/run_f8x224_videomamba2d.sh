export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba2d_tiny_f8_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/home/oem/users/dhjung/kinetics400/kinetics_400/videos_320'
DATA_PATH='/home/oem/users/dhjung/new_k400/label'

# PARTITION='video5'
# GPUS=8
# GPUS_PER_NODE=8
# CPUS_PER_TASK=16

torchrun --nproc_per_node=4 \
        run_class_finetuning.py \
        --model videomamba2d_tiny \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 32 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 8 \
        --num_workers 12 \
        --warmup_epochs 0 \
        --tubelet_size 1 \
        --epochs 20 \
        --lr 0.5e-4 \
        --drop_path 0.1 \
        --aa rand-m5-n2-mstd0.25-inc1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --bf16 \
        --finetune './exp/k400/videomamba_tiny/checkpoint/videomamba_t16_k400_f8_res224.pth' \