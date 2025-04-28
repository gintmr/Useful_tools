# python F:\ML\Projects\Ocean\SA-1B_xinrui\SA-1B_trans_TO_COCO\merge_SA1B.py --input_dir F:\ML\Projects\Ocean\SA-1B_xinrui\subset1 --output F:\ML\Projects\Ocean\SA-1B_xinrui\merge_subset1.json

#!/bin/bash

# 定义路径参数
ANNO_PATH="F:/ML/Projects/Ocean/SA-1B_xinrui/merged_subset1.json"
IMAGE_FOLDER_PATH="F:/ML/Projects/Ocean/SA-1B_xinrui/subset1"
OUTPUT_IMAGE_FOLDER="F:/ML/Projects/Ocean/SA-1B_xinrui/subset1_1024"
OUTPUT_ANNO_PATH="F:/ML/Projects/Ocean/SA-1B_xinrui/merged_subset1_1024.json"

# 调用 Python 脚本
python process_images_and_annotations.py \
    --anno_path "$ANNO_PATH" \
    --image_folder_path "$IMAGE_FOLDER_PATH" \
    --output_image_folder "$OUTPUT_IMAGE_FOLDER" \
    --output_anno_path "$OUTPUT_ANNO_PATH" \
    --max_size 1024 \
    --seg_type rle