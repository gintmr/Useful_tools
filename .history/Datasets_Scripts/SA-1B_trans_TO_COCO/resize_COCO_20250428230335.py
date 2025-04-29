import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def resize_image(image, max_size=1024):
    """
    等比例缩放图像，最长边不超过 max_size
    """
    h, w = image.shape[:2]
    scale = min(max_size / max(h, w), 1.0)  # 计算缩放比例
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale


def resize_annotations(coco, annotations, scale, seg_type):
    """
    等比例缩放标注中的分割掩码、标注框和关键点
    """
    for ann in annotations:
        if 'bbox' in ann:
            ann['bbox'] = [int(x * scale) for x in ann['bbox']]
        if 'segmentation' in ann:
            if seg_type == 'poly':
                ann['segmentation'] = [[int(x * scale) for x in seg] for seg in ann['segmentation']]
            elif seg_type == 'rle':
                mask = coco.annToMask(ann=ann)
                resized_mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
                rle = maskUtils.encode(np.asfortranarray(resized_mask))
                ann['segmentation'] = {
                    'size': rle['size'],
                    'counts': rle['counts'].decode('utf-8')
                }
        if 'keypoints' in ann:
            ann['keypoints'] = [int(x * scale) for x in ann['keypoints']]
    return annotations


def process_images_and_annotations(cocoed_anno, image_folder_path, output_image_folder, output_anno_path, max_size=1024, seg_type="rle"):
    """
    处理所有图像和标注，保存缩放后的图像和更新后的标注
    """
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    else:
        for img in os.listdir(output_image_folder):
            os.remove(os.path.join(output_image_folder, img))
    if os.path.exists(output_anno_path):
        os.remove(output_anno_path)

    coco = cocoed_anno
    imgIds = coco.getImgIds()[:]
    new_images = []
    new_annotations = []

    for imgId in tqdm(imgIds):
        img_info = coco.loadImgs(imgId)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(image_folder_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: {img_path} not read.")
            continue

        resized_image, scale = resize_image(image, max_size)

        new_img_name = os.path.splitext(img_name)[0] + ".jpg"
        new_img_path = os.path.join(output_image_folder, new_img_name)
        cv2.imwrite(new_img_path, resized_image)

        img_info['file_name'] = new_img_name
        img_info['height'], img_info['width'] = resized_image.shape[:2]
        new_images.append(img_info)

        annIds = coco.getAnnIds(imgIds=imgId)
        annotations = coco.loadAnns(annIds)
        resized_annotations = resize_annotations(coco, annotations, scale, seg_type)
        new_annotations.extend(resized_annotations)

    new_anno = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.loadCats(coco.getCatIds())
    }
    with open(output_anno_path, 'w') as f:
        json.dump(new_anno, f, indent=4)

    print(f"Resized images saved to {output_image_folder}")
    print(f"Updated annotations saved to {output_anno_path}")


def main():
    parser = argparse.ArgumentParser(description="Process images and annotations for COCO format.")
    parser.add_argument("--anno_path", type=str, required=True, help="Path to the COCO annotation JSON file.")
    parser.add_argument("--image_folder_path", type=str, required=True, help="Path to the image folder.")
    parser.add_argument("--output_image_folder", type=str, required=True, help="Path to the output image folder.")
    parser.add_argument("--output_anno_path", type=str, required=True, help="Path to the output annotation JSON file.")
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum size for resizing images.")
    parser.add_argument("--seg_type", type=str, default="rle", choices=["poly", "rle"], help="Segmentation type: 'poly' or 'rle'.")

    args = parser.parse_args()

    cocoed_anno = COCO(args.anno_path)
    process_images_and_annotations(
        cocoed_anno,
        args.image_folder_path,
        args.output_image_folder,
        args.output_anno_path,
        max_size=args.max_size,
        seg_type=args.seg_type
    )


if __name__ == '__main__':
    main()