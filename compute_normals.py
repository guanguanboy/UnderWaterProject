import cv2
import numpy as np
import mxnet as mx
import os

def depth2normal(depth):
    w, h = depth.shape
    dx = -(depth[2:h, 1:h-1] - depth[0:h-2, 1:h-1]) * 0.5
    dy = -(depth[1:h-1, 2:h] - depth[1:h-1, 0:h-2]) * 0.5
    dz = mx.nd.ones((w-2, h-2))
    dl = mx.nd.sqrt(mx.nd.elemwise_mul(dx, dx) + mx.nd.elemwise_mul(dy, dy) + mx.nd.elemwise_mul(dz, dz))
    dx = mx.nd.elemwise_div(dx, dl) * 0.5 + 0.5
    dy = mx.nd.elemwise_div(dy, dl) * 0.5 + 0.5
    dz = mx.nd.elemwise_div(dz, dl) * 0.5 + 0.5
    return np.concatenate([dy.asnumpy()[np.newaxis, :, :], dx.asnumpy()[np.newaxis, :, :], dz.asnumpy()[np.newaxis, :, :]], axis=0)

def resize_to_square(image):
    h, w = image.shape
    size = max(h, w)
    padded_image = np.zeros((size, size), dtype=image.dtype)
    padded_image[:h, :w] = image
    return padded_image, h, w  # 返回填充后的图像和原始尺寸

def process_depth_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            # 读取深度图
            output_path = os.path.join(output_dir, f"normal_{filename}")
            if os.path.exists(output_path):
                print(f"File {output_path} already exists. Skipping...")
                continue  # 跳过该文件

            depth_path = os.path.join(input_dir, filename)
            depth = cv2.imread(depth_path, 0)
            padded_depth, orig_h, orig_w = resize_to_square(depth)
            
            # 计算法线并恢复原始尺寸
            normal = np.array(depth2normal(mx.nd.array(padded_depth)) * 255)
            normal = cv2.cvtColor(np.transpose(normal, [1, 2, 0]), cv2.COLOR_BGR2RGB)
            normal = normal[:orig_h, :orig_w]  # 去除填充区域

            cv2.imwrite(output_path, normal.astype(np.uint8))
            print(f"Saved normal map to {output_path}")

if __name__ == '__main__':
    input_directory = '/data/gl/Codes/UnderWater3D/Datasets/D3_depth_png_gt/'
    output_directory = '/data/gl/Codes/UnderWater3D/Datasets/D3_normal_png_gt/'
    process_depth_images(input_directory, output_directory)

"""
import os
import numpy as np
from PIL import Image

def load_normal_map(path):
    # 加载法线图并归一化
    normal_map = np.array(Image.open(path)).astype(np.float32)
    normal_map = (normal_map / 255.0) * 2 - 1  # 将范围从 [0, 255] 映射到 [-1, 1]
    
    # 归一化为单位向量
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= norm
    return normal_map

def compute_average_angle_difference(normal_map1, normal_map2):
    # 计算法线向量的点积
    dot_product = np.sum(normal_map1 * normal_map2, axis=2)
    
    # 将点积限制在 [-1, 1] 范围内，防止数值误差导致超出范围
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算夹角并转换为角度
    angles = np.arccos(dot_product)  # 弧度
    angles_degrees = np.degrees(angles)  # 转换为角度
    
    # 计算平均角度差
    average_angle_difference = np.mean(angles_degrees)
    return average_angle_difference

def calculate_directory_angle_difference(dir1, dir2):
    total_angle_diff = 0.0
    count = 0

    for filename in sorted(os.listdir(dir1)):
        if filename.endswith('.png'):
            normal_map1_path = os.path.join(dir1, filename)
            normal_map2_path = os.path.join(dir2, filename)

            # 检查对应法线图是否存在
            if not os.path.exists(normal_map2_path):
                print(f"File {normal_map2_path} does not exist. Skipping...")
                continue

            # 加载法线图
            normal_map1 = load_normal_map(normal_map1_path)
            normal_map2 = load_normal_map(normal_map2_path)

            # 计算平均角度差
            angle_diff = compute_average_angle_difference(normal_map1, normal_map2)
            total_angle_diff += angle_diff
            count += 1
            print(f"Angle difference for {filename}: {angle_diff:.2f} degrees")

    # 计算所有法线图对的总平均角度差
    if count > 0:
        overall_average_angle_diff = total_angle_diff / count
        print(f"\nOverall Average Angle Difference: {overall_average_angle_diff:.2f} degrees")
    else:
        print("No matching files found.")

if __name__ == '__main__':
    dir1 = '/mnt/petrelfs/tangyiwen/water-splatting/gt_surface_normals_visualization'
    dir2 = '/mnt/petrelfs/tangyiwen/water-splatting/surface_normals_visualization'
    calculate_directory_angle_difference(dir1, dir2)

"""