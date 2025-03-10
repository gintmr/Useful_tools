import os  
import json
import numpy as np
import cv2
from pycocotools import mask as maskUtils
from tqdm import tqdm
import shutil


image_folder_list = ['/data2/wuxinrui/Datasets/UIIS/UDW/train',
                     '/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/train_list'
                     ]

json_folder_list = ['/data2/wuxinrui/Datasets/UIIS/UDW/annotations/train.json',
                    '/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/train-taxonomic_cleaned.json'
                  ]

save_image_folder = '/data2/wuxinrui/Projects/ICCV/table3/cascade_detectron/merge_datasets/images'
save_json_folder = '/data2/wuxinrui/Projects/ICCV/table3/cascade_detectron/merge_datasets/annotations'
os.makedirs(save_image_folder,exist_ok=True)
os.makedirs(save_json_folder,exist_ok=True)

json_file_name = 'merge_datasets.json'


merged_json = {}
merged_json['images'] = []
merged_json['annotations'] = []
merged_json['categories'] = []

for i in tqdm(range(len(json_folder_list)), desc="Merging Datasets"):
    
    image_id_dict = {} ## 以image_id 为键值的dit，方便读取对应信息
    category_id_dict = {} ## 以category_id 为键值的dit，方便读取对应信息
    
    
    with open(json_folder_list[i],'r') as f:
        json_data = json.load(f)
        for image in tqdm(json_data['images'], desc=f"Merging Images in {json_folder_list[i]}"):
            image_id = image['id']
            
            merged_json['images'].append(image)
            image_new_id = len(merged_json['images']) - 1
            merged_json['images'][image_new_id]['id'] = image_new_id ## 此处确定的id为图片id,id序号是数字类型
            
            image_id_dict[str(image_id)] = {
                'file_name': image['file_name'],
                'new_id': image_new_id,
                'height': image['height'],
                'width': image['width']
            }
            
            
        for category in tqdm(json_data['categories'], desc=f"Merging Categories in {json_folder_list[i]}"):
            
            category_id = category['id']
            
            merged_json['categories'].append(category)
            category_new_id = len(merged_json['categories']) ## 此处确定的id为类别id，需要从 1 开始，序号0留给Foreground
            merged_json['categories'][category_new_id-1]['id'] = category_new_id
           
            if category_id is not None:
                category_id_dict[str(category_id)] = {
                    'name': category['name'],
                    'new_id': category_new_id
                }
            
            
            
        for annotation in tqdm(json_data['annotations'], desc=f"Merging Annotations in {json_folder_list[i]}"):
            merged_json['annotations'].append(annotation)
            now_id = len(merged_json['annotations']) - 1
            merged_json['annotations'][now_id]['id'] = now_id ## 当前标注自身的序号
            
            ori_image_id = annotation['image_id'] ## 得到合并前的对应图像id
            new_image_id = image_id_dict[str(ori_image_id)]['new_id'] ## 得到合并后的对应图像id
            image_file_name = image_id_dict[str(ori_image_id)]['file_name'] ## 得到合并后的对应图像文件名
            
            
            image_file_path = os.path.join(image_folder_list[i], image_file_name)
            shutil.copy(image_file_path, save_image_folder) ## 复制图片
            
            
            image_height = image_id_dict[str(ori_image_id)]['height'] ## 得到合并后的对应图像高度
            merged_json['annotations'][now_id]['image_height'] = image_height 

            image_width = image_id_dict[str(ori_image_id)]['width'] ## 得到合并后的对应图像宽度
            merged_json['annotations'][now_id]['image_width'] = image_width
             
            
            merged_json['annotations'][now_id]['image_id'] = new_image_id ## 更新图像id
            merged_json['annotations'][now_id]['image_name'] = image_file_name ## 更新图像文件名
            
            ori_category_id = annotation['category_id'] ## 得到合并前的对应类别id
            if ori_category_id is not None:
                new_category_id = category_id_dict[str(ori_category_id)]['new_id'] ## 得到合并后的对应类别id
                merged_json['annotations'][now_id]['category_id'] = new_category_id
            
            if annotation['segmentation'] is not None:
                ### 如果是rle格式的segmentation
                segmentation = annotation['segmentation']
                if "size" not in segmentation:
                    mask = np.zeros((image_height, image_width), dtype=np.uint8)
                    polygon = segmentation[0]
                   # 将一维数组转换为 (N, 2) 的形状
                    polygon_pairs = np.array(polygon).reshape(-1, 2)

                    # 转换为整数类型
                    polygon_pairs = polygon_pairs.astype(np.int32)

                    # 调整形状为 (N, 1, 2)
                    polygon_pairs = polygon_pairs.reshape(-1, 1, 2)
                    cv2.fillPoly(mask, polygon_pairs, 1)
                    
                    rle = maskUtils.encode(np.asfortranarray(mask))
                    
                    annotation['segmentation'] = {
                        "size": [image_height, image_width],
                        "counts":  rle["counts"].decode("utf-8")
                    }
            
            if "iscrowd" not in annotation or annotation['iscrowd'] is None:
                merged_json['annotations'][now_id]['iscrowd'] = 0
                
                
                
                
save_json_path = os.path.join(save_json_folder, json_file_name)

with open(save_json_path, 'w') as f:
    json.dump(merged_json, f, indent=4)
    print(f"Saved merged json to {save_json_path}")
    
save_category_path = save_json_path.replace('.json', '_categories.json')

with open(save_category_path, 'w') as f:
    json.dump(merged_json['categories'], f, indent=4)
    print(f"Saved category json to {save_category_path}")