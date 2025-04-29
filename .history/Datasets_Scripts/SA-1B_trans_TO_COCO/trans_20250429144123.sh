#!/bin/bash

#===> modified by XinruiWu


#===> Step 1: unzip the SA-1B dataset
tar -xvf F:/ML/Projects/Ocean/SA-1B_xinrui/subset3.tar -C F:/ML/Projects/Ocean/SA-1B_xinrui/subset3


#===> Step 2: merge the SA-1B dataset into one json file
python F:/ML/Projects/Ocean/SA-1B_xinrui/SA-1B_trans_TO_COCO/merge_SA1B.py --input_dir F:/ML/Projects/Ocean/SA-1B_xinrui/subset3 --output F:/ML/Projects/Ocean/SA-1B_xinrui/merge_subset3.json


#===> Step 3: resize the images and annotations to 1024x1024
ANNO_PATH="F:/ML/Projects/Ocean/SA-1B_xinrui/merge_subset3.json"
IMAGE_FOLDER_PATH="F:/ML/Projects/Ocean/SA-1B_xinrui/subset3"
OUTPUT_IMAGE_FOLDER="F:/ML/Projects/Ocean/SA-1B_xinrui/subset3_1024"
OUTPUT_ANNO_PATH="F:/ML/Projects/Ocean/SA-1B_xinrui/merge_subset3_1024.json"

python F:/ML/Projects/Ocean/SA-1B_xinrui/SA-1B_trans_TO_COCO/resize_COCO.py \
    --anno_path "$ANNO_PATH" \
    --image_folder_path "$IMAGE_FOLDER_PATH" \
    --output_image_folder "$OUTPUT_IMAGE_FOLDER" \
    --output_anno_path "$OUTPUT_ANNO_PATH" \
    --max_size 1024 \
    --seg_type rle