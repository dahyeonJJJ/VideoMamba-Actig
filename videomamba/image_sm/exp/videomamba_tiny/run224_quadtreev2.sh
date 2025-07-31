export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME='videomamba_tiny_quadtreev2_p14_d4_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"

# PARTITION='video5'
# NNODE=1
# NUM_GPUS=8
# NUM_CPU=128

torchrun --nproc_per_node=2 \
        main.py \
        --root_dir_train /ephemeral/IMNET/ILSVRC2012_img_train/ \
        --meta_file_train /ephemeral/IMNET/ImageNet_train.txt \
        --root_dir_val /ephemeral/IMNET/ILSVRC2012_img_val/ \
        --meta_file_val /ephemeral/IMNET/ImageNet_val.txt \
        --model videomamba_tiny_quadtreev2_p14_d4 \
        --batch-size 384 \
        --num_workers 16 \
        --lr 5e-4 \
        --clip-grad 5.0 \
        --weight-decay 0.1 \
        --drop-path 0 \
        --no-repeated-aug \
        --aa v0 \
        --no-model-ema \
        --output_dir ${OUTPUT_DIR} \
        --bf16 \
        --dist-eval \
        