export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='videomamba_small_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
# PARTITION='video5'
# NNODE=1
# NUM_GPUS=8
# NUM_CPU=128

torchrun --nproc_per_node=2 \
        --master_port 25901 \
        main.py \
        --root_dir_train /ephemeral/IMNET/ILSVRC2012_img_train/ \
        --meta_file_train /ephemeral/IMNET/ImageNet_train.txt \
        --root_dir_val /ephemeral/IMNET/ILSVRC2012_img_val/ \
        --meta_file_val /ephemeral/IMNET/ImageNet_val.txt \
        --model videomamba_small \
        --batch-size 512 \
        --num_workers 16 \
        --lr 5e-4 \
        --weight-decay 0.05 \
        --drop-path 0.15 \
        --no-model-ema \
        --output_dir ${OUTPUT_DIR} \
        --bf16 \
        --dist-eval \
        --resume /ephemeral/jdh/VideoMamba/videomamba/image_sm/exp/videomamba_small/checkpoint/videomamba_s16_in1k_res224.pth \
        --eval