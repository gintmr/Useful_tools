import os
import glob
import shutil
import numpy as np
import glob
import os.path
import pycocotools.mask as mask_util
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
import glob
import os
import argparse
from PIL import Image
import shutil
import pycocotools.mask as mask
from segment_anything import sam_model_registry, SamPredictor

import pycocotools.mask as mask_utils
from tqdm import tqdm
import logging

log_filename = "inaturalist.log"

# 配置日志
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='vit_h', help="model type", choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", type=str, default="/data2/wuxinrui/Projects/ICCV/evaluate_fm/checkpoint/sam_vit_h_4b8939.pth", help="path to the checkpoint")
    parser.add_argument("--test_img_path", type=str, default="/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/test_list", help="the test image path")
    parser.add_argument("--label_path", type=str, default="/data2/wuxinrui/Projects/ICCV/jsons_for_salient_instance_segmentation/test_1_prompts.json", help="the test json path")
    parser.add_argument("--output_dir", type=str, default="/data2/wuxinrui/Projects/ICCV/evaluate_fm/datasets/outputs", help="path to save the model")
    parser.add_argument("--prompt", type=str, default="point", help="which kind of prompt",choices=['point','bbox'])
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    # parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--image_size", type=int, default=1024, help="image size")
    device = "cuda"
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
    sam_checkpoint = args.checkpoint_path
    model_type = args.model_type
    prompt=args.prompt
    logging.info("start testing !!!")
    logging.info("############################")
    logging.info(f"json_path = {args.label_path}")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device) 
    predictor = SamPredictor(sam)

    with open(args.label_path, 'r') as f:
        json_data = json.load(f)
        
        annotations = json_data['annotations']

    img_files = [f for f in os.listdir(args.test_img_path)]
    # if f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]

    ious = []
    pas = []
    for img_file in tqdm(img_files):
        

        now_img_anno = [annotation for annotation in annotations if annotation['image_name'] == img_file]

        img_path = os.path.join(args.test_img_path, img_file)
        image = cv2.imread(img_path, 2)
        height, width = image.shape[:2]
        # print(height, width)
        
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.uint8(image)
            height,width=np.shape(image)[0],np.shape(image)[1]
        except:
            # 如果图像被截断，则打印错误信息并跳过
            print("Image truncated: " + img_file)
            continue
        # print(img_path)
        logging.info(img_path)
        # 如果输出目录下已经存在对应的json文件，则跳过
        # if os.path.exists(os.path.join(args.output_dir,name.replace(".jpg",".json"))):
        #     continue
        # is_crowd = []
        pred = []
        gt = []
        size = [height,width]
        ori_masks = []
        pred_masks = []
        for item_anno in now_img_anno:
            # print(item_anno)
            # is_crowd.append(item_anno['iscrowd'])
            
            
            img_mask = np.zeros([height, width])
            #ddd
            ori_segmentation = item_anno["segmentation"]
            gt.append(ori_segmentation)
            ori_mask = np.array(mask_utils.decode(ori_segmentation), dtype=np.float32)
            ori_masks.append(ori_mask)
            # print(f"ori_mask :{ori_mask}")
            #ddd
            predictor.set_image(image)
            # 根据prompt参数选择不同的预测方式
            if prompt=="bbox":
                # 使用bbox进行预测
                input_box = np.array([item_anno['bbox'][0], item_anno['bbox'][1], item_anno['bbox'][0]+item_anno['bbox'][2], item_anno['bbox'][1]+item_anno['bbox'][3]])
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
            elif prompt=="point":
                # 使用point进行预测
                points,labels=[],[]
                coordinates=item_anno['coords']
                for item_coords in coordinates:
                    points.append([item_coords[0], item_coords[1]])
                    labels.append(1)
                input_point = np.array(points)
                input_label = np.array(labels)
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
                ## mask 为True、False的矩阵，shape：(1, 427, 640)
                ## logits 为低分辨率逻辑值，可用于后续输入使用。shape： (1, 256, 256)
                pred_masks.append(masks[0])
            else:
                print("not implemented")
            # 将masks[0]中的像素值设为255
            # print(f"masks: {masks}, scores: {scores}, logits: {logits}")
            # print(f"masks shape: {masks.shape}")
            # print(f"logits shape: {logits.shape}")

            img_mask[masks[0]] = 255
            # 将masks[0]转换为Fortran风格的数组
            fortran_ground_truth_binary_mask = np.asfortranarray(masks[0])
            # 将Fortran风格的数组转换为RLE编码
            compressed_rle = mask_util.encode(fortran_ground_truth_binary_mask)
            # 将RLE编码的counts字段转换为字符串
            # compressed_rle['counts'] = str(compressed_rle['counts'], encoding="utf-8")
            # 将RLE编码添加到item_anno字典中
            # 将masks[0]的面积添加到item_anno字典中
            pred.append(compressed_rle)

        
        # pic_iou = conculate_iou(pred, gt, is_crowd) ## shape:[len(pred), len(pred)]
        # pic_pa = conculate_pa(pred_masks, ori_masks)
        pa_list, iou_list = conculate_pa_iou(pred_masks, ori_masks)
        
        
        ious.append(iou_list[i] for i in range(len(iou_list)))
        pas.append(pa_list[i] for i in range(len(pa_list)))
        
    avg_iou = np.mean(ious)
    avg_pa = np.mean(pas)
    
    print(f"Average IoU: {avg_iou}")
    print(f"Average PA: {avg_pa}")
    
    
    logging.info(f"json_path = {args.label_path}")
    logging.info(f"Average IoU: {avg_iou}")
    logging.info(f"Average PA: {avg_pa}")

def conculate_iou(pred, gt, iscrowd):
    
    # 计算IoU
    iou = mask_util.iou(pred, gt, iscrowd)
    ious = [iou[i][i] for i in range(len(pred))]
    return ious
def conculate_pa(pred_masks, ori_masks):
    pa_list = []
    # 计算PA
    for i in range(len(pred_masks)):
        pred_mask = pred_masks[i]
        ori_mask = ori_masks[i]
 # 计算TP, FP, TN, FN
        TP = np.sum((pred_mask == True) & (ori_mask == True))
        FP = np.sum((pred_mask == True) & (ori_mask == False))
        TN = np.sum((pred_mask == False) & (ori_mask == False))
        FN = np.sum((pred_mask == False) & (ori_mask == True))

        pa = (TP + TN) / (TP + TN + FP + FN)
        pa_list.append(pa)
        
    return pa_list

def conculate_pa_iou(pred_masks, ori_masks):
    pa_list = []
    iou_list = []
    # 计算PA
    for i in range(len(pred_masks)):
        pred_mask = pred_masks[i]
        ori_mask = ori_masks[i]
 # 计算TP, FP, TN, FN
        TP = np.sum((pred_mask == True) & (ori_mask == True))
        FP = np.sum((pred_mask == True) & (ori_mask == False))
        TN = np.sum((pred_mask == False) & (ori_mask == False))
        FN = np.sum((pred_mask == False) & (ori_mask == True))

        pa = (TP + TN) / (TP + TN + FP + FN)
        pa_list.append(pa)
        
        iou = TP / (TP + FP + FN)
        iou_list.append(iou)
        
    return pa_list, iou_list

if __name__ == "__main__":
    main()
