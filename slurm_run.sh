#!/bin/bash
#SBATCH --job-name=trellis2
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=hpc-mid
#SBATCH --output=output/slurm_logs/trellis2_%j.out
#SBATCH --error=output/slurm_logs/trellis2_%j.err


# Always run from the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p output/slurm_logs

# --- Conda setup (robust) ---
source /mnt/home/jaehoon/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

# --- CUDA env ---
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# # --- Python path (append, don't overwrite) ---
# export PYTHONPATH="/fs/gamma-projects/ARL3D/jaehoon/code/outdoor_relighting/2d-gaussian-splatting:${PYTHONPATH:-}"
# export TORCH_CUDA_ARCH_LIST="9.0"

IMAGE_PATH=/mnt/home/jaehoon/code/gs-3dgs/data/IMG_0477/masked_input/00117.png
OUTPUT_PATH=/mnt/home/jaehoon/code/gs-3dgs/data/IMG_0477/masked_input/00117_trellis2
HDRI_PATH=/mnt/home/jaehoon/code/gs-3dgs/data/IMG_0477/masked_input/00117_hdri.exr
OUTPUT_NAME=00117_trellis2

# export TORCH_CUDA_ARCH_LIST="9.0"
python example.py --image_path $IMAGE_PATH \
 --output_path $OUTPUT_PATH \
 --hdri_path $HDRI_PATH \
 --output_name $OUTPUT_NAME