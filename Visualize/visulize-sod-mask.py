import cv2
import os
import numpy as np

def overlay_mask_on_image(image_path, mask_path, output_path, mask_color=(255, 0, 0), alpha=0.5):
    """
    将掩码叠加到原图上并保存。
    :param image_path: 原图路径
    :param mask_path: 掩码路径
    :param output_path: 输出路径
    :param mask_color: 掩码颜色 (B, G, R)
    :param alpha: 掩码透明度 (0 到 1)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to read mask at {mask_path}")
        return

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = mask > 0

    mask_image = np.zeros_like(image)
    mask_image[mask] = mask_color

    # 将掩码叠加到原图上
    overlay = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)

    # 保存结果
    cv2.imwrite(output_path, overlay)
    print(f"Saved: {output_path}")

image_folder = "/data2/MIMC_FINAL/visual-results/images/"  # 原图文件夹
mask_folder = "/data2/SGL-KRN/visual-results"              # 掩码文件夹
output_folder = "/data2/SGL-KRN/visual-results-overlay"    # 输出文件夹

os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(mask_folder):
    basename = os.path.splitext(filename)[0]
    mask_path = os.path.join(mask_folder, filename)
    for image_name in os.listdir(image_folder):
        if basename in image_name:
            image_path = os.path.join(image_folder, image_name)
            
    output_path = os.path.join(output_folder, filename)

    if not os.path.exists(image_path):
        print(f"Warning: No corresponding image found for mask {filename}")
        continue

    overlay_mask_on_image(image_path, mask_path, output_path, mask_color=(178, 102, 255), alpha=0.3)

print("All images processed and saved.")