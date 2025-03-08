
export CUDA_LAUNCH_BLOCKING=1
./train_net.py \
  --resume \
  --config-file /path/to/yours/yaml \
  --eval-only MODEL.WEIGHTS /path/to/your/pth \
  OUTPUT_DIR /path/to/your/outputs

