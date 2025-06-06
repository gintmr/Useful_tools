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
    # 读取原图
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image at {image_path}")
        return

    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to read mask at {mask_path}")
        return

    # 调整掩码尺寸以匹配原图
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 将掩码转换为布尔值
    mask = mask > 0

    # 创建掩码图像
    mask_image = np.zeros_like(image)
    mask_image[mask] = mask_color

    # 将掩码叠加到原图上
    overlay = cv2.addWeighted(image, 1 - alpha, mask_image, alpha, 0)

    # 保存结果
    cv2.imwrite(output_path, overlay)
    print(f"Saved: {output_path}")

# 输入文件夹路径
image_folder = "/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/visual-results/images/"  # 原图文件夹
mask_folder = "/data2/wuxinrui/Projects/ICCV/SGL-KRN/visual-results"  # 掩码文件夹
output_folder = "/data2/wuxinrui/Projects/ICCV/SGL-KRN/visual-results-overlay"  # 输出文件夹

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有图片
for filename in os.listdir(mask_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        mask_path = os.path.join(mask_folder, filename)
        image_path = os.path.join(image_folder, filename.replace('.png', '.jpg'))
        output_path = os.path.join(output_folder, filename)

        # 跳过不存在的原图
        if not os.path.exists(image_path):
            print(f"Warning: No corresponding image found for mask {filename}")
            continue

        # 将掩码叠加到原图上并保存
        overlay_mask_on_image(image_path, mask_path, output_path, mask_color=(178, 102, 255), alpha=0.3)

print("All images processed and saved.")