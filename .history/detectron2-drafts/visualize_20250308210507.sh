# export CUDA_VISIBLE_DEVICES=3,4
export CUDA_LAUNCH_BLOCKING=1
export visual_path=/data2/wuxinrui/Projects/ICCV/table3/cascade_detectron/visual/cascade/zero

python visualize.py \
  --config-file /path/to/yours/yaml \
  --eval-only MODEL.WEIGHTS /path/to/your/pth \