export CUDA_VISIBLE_DEVICES=0,1,2,3

export CUDA_LAUNCH_BLOCKING=1
./train_net.py --num-gpus 4 \
  --resume \
  --config-file /path/to/yours/yaml \
  OUTPUT_DIR /path/to/your/outputs \
  MODEL.WEIGHTS /path/to/your/pth