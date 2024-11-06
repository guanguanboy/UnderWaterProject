"""
from PIL import Image
import numpy as np

def convert_depth_tif_to_png(input_tif, output_png):
    # 读取32位深度图（tif格式）
    depth_image = Image.open(input_tif)
    
    # 将图像转换为 numpy 数组
    depth_array = np.array(depth_image)

    # 确保深度图是32位浮点数或整数类型
    if depth_array.dtype != np.uint32 and depth_array.dtype != np.float32:
        raise ValueError("Input depth image must be of type uint32 or float32")

    # 归一化到0-65535范围，适合16位深度图
    depth_array = np.clip(depth_array, 0, np.max(depth_array))  # 限制深度值范围
    depth_array = (depth_array / np.max(depth_array) * 65535).astype(np.uint16)  # 归一化到16位

    # 转换为16位 PNG并保存
    output_image = Image.fromarray(depth_array)
    output_image.save(output_png, format="PNG")

    print(f"Conversion successful! Saved as {output_png}")

# 使用示例
input_tif = 'Datasets/D3_depth_gt/depthT_S04856.tif'  # 输入32位tif深度图路径
output_png = 'Datasets/D3_depth_png_gt/depthT_S04856.png'  # 输出16位png图像路径

convert_depth_tif_to_png(input_tif, output_png)
"""

import os
from PIL import Image
import numpy as np

def convert_depth_tif_to_png(input_tif, output_png):
    # 读取32位深度图（tif格式）
    depth_image = Image.open(input_tif)
    
    # 将图像转换为 numpy 数组
    depth_array = np.array(depth_image)

    # 确保深度图是32位浮点数或整数类型
    if depth_array.dtype != np.uint32 and depth_array.dtype != np.float32:
        raise ValueError("Input depth image must be of type uint32 or float32")

    # 归一化到0-65535范围，适合16位深度图
    depth_array = np.clip(depth_array, 0, np.max(depth_array))  # 限制深度值范围
    depth_array = (depth_array / np.max(depth_array) * 65535).astype(np.uint16)  # 归一化到16位
    #depth_array = np.clip(depth_array, 0, 65535).astype(np.uint16)  # 限制深度值范围

    # 转换为16位 PNG并保存
    output_image = Image.fromarray(depth_array)
    output_image.save(output_png, format="PNG")

    print(f"Conversion successful! Saved as {output_png}")

def convert_all_tif_in_folder(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 处理.tif文件
        if filename.endswith('.tif'):
            input_tif = os.path.join(input_folder, filename)
            output_png = os.path.join(output_folder, filename.replace('.tif', '.png'))

            # 调用转换函数
            convert_depth_tif_to_png(input_tif, output_png)

# 使用示例
input_folder = 'Datasets/D3_depth_gt/'  # 输入文件夹路径，包含32位tif深度图
output_folder = 'Datasets/D3_depth_png_gt/'  # 输出文件夹路径，保存16位png深度图

convert_all_tif_in_folder(input_folder, output_folder)

