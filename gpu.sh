srun -t 22:30:00 --gres=gpu:1 --pty /bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/nvidia-390/"
export CUDA_VISIBLE_DEVICES=0

