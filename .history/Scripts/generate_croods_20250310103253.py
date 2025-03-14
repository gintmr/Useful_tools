import numpy as np
import random
import os
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import json

def process_json_files(annotations_directory, coco_annotation_path, num_random_points):
    # 初始化 COCO API
    coco = COCO(coco_annotation_path)

    # 遍历目录中的所有JSON文件
    for filename in os.listdir(annotations_directory):
        coords = []
        if filename.endswith('.json'):
            file_path = os.path.join(annotations_directory, filename)
            
            # 读取JSON文件
            with open(file_path, 'r') as f:
                annotations = json.load(f)
            
            # 遍历annotations列表
            h = annotations[-1]["height"]
            w = annotations[-1]["width"]
            gt = np.zeros((h, w), dtype=np.uint8)
            masks = []
            for annotation in annotations[:-1]:
                if not annotation["iscrowd"]:
                    segs = annotation['segmentation']
                    for seg in segs:
                        seg = np.array(seg).reshape(-1, 2)
                        cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], int(annotation['category_id']))
                        # print(f"gt = {gt}")
                        print(f"gt_shape = {gt.shape}")
                        masks.append(gt)
                elif annotation["iscrowd"]:
                    has_crowd_flag = 1
                    rle = annotation['segmentation']['counts']
                    # ##DDD以下是对乱码形式的rle解析
                    # mask = np.array(mask_utils.decode(rle), dtype=np.float32)

                    #ddd以下是对整数形式的rle解析
                    # 断言rle的和等于annotation['segmentation']['size'][0] * annotation['segmentation']['size'][1]
                    assert sum(rle) == annotation['segmentation']['size'][0] * annotation['segmentation']['size'][1]
                    # 将annotation转换为mask
                    mask = coco.annToMask(annotation)

                    # 获取mask中唯一的标签
                    unique_label = list(np.unique(mask))
                    # 断言mask中只有两个标签，且1和0都在其中
                    assert len(unique_label) == 2 and 1 in unique_label and 0 in unique_label
                    # 将gt中mask为0的部分置为0，mask为1的部分置为255
                    gt = gt * (1 - mask) + mask * 255 
                    masks.append(mask)
                visualize_masks_as_binary(masks, annotations_directory + "/" + filename[:-5]+"_binary.png")
                
                # 移除背景点（值为0的点）
                non_background_points = np.argwhere(gt > 0)
                
                # 如果存在非背景点，则随机选择 num_random_points 个点
                if non_background_points.size > 0:
                    print(f"size = {non_background_points.size}")
                    print(f"lenof = {len(non_background_points)}")
                    for _ in range(num_random_points):
                        random_point_idx = random.randint(0, non_background_points.size // 2 - 1)
                        random_point = non_background_points[random_point_idx]
                        print("Random non-background point:", random_point)
                        coords.append(random_point.tolist())  # 将点转换为列表并添加到 coords 中
                else:
                    print("No non-background points found.")
            
            # 将生成的随机点位添加到 annotations 中
            print(f"coords = {coords}")
            annotations[-1]["coords"] = coords
            
            # 将更新后的annotations写回JSON文件
            with open(file_path, 'w') as f:
                json.dump(annotations, f, indent=4)

    print('所有annotations文件处理完成。')


def visualize_masks_as_binary(masks, output_path):
    """
    将mask矩阵可视化为黑白图像
    :param masks: mask矩阵列表，每个mask是一个二维NumPy数组
    :param output_path: 输出图像的路径
    """
    # 假设所有mask的形状相同，取第一个mask的形状
    if not masks:
        raise ValueError("No masks provided")
    
    height, width = masks[0].shape
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    for mask in masks:
        combined_mask[mask > 0] = 255  # 将mask区域设置为白色（255）
    
    # 保存为黑白图像
    cv2.imwrite(output_path, combined_mask)
    print(f"Binary mask image saved to {output_path}")



# 使用示例
annotations_directory = '/User/Projects/ICCV/evaluate_fm/datasets/json'  # 替换为实际目录
coco_annotation_path = "/User/Datasets/COCO/annotations/instances_val2017.json"  # COCO 标注文件路径
num_random_points = 5  # 控制生成的随机点位数量 



process_json_files(annotations_directory, coco_annotation_path, num_random_points)