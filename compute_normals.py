
import cv2
import numpy as np
import mxnet as mx
import os

import numpy as np
from scipy import ndimage

def depth_to_normal_map(depth_map):
    # 计算 x 和 y 方向的梯度
    dz_dx = np.zeros_like(depth_map)
    dz_dy = np.zeros_like(depth_map)
    
    # 使用简单的差分来计算梯度
    dz_dx[:, 1:-1] = (depth_map[:, 2:] - depth_map[:, :-2]) / 2.0  # x 方向梯度
    dz_dy[1:-1, :] = (depth_map[2:, :] - depth_map[:-2, :]) / 2.0  # y 方向梯度

    # 初始化法线图
    normal_map = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.float32)

    # 计算法向量
    normal_map[..., 0] = -dz_dx  # 法线的 x 分量
    normal_map[..., 1] = -dz_dy  # 法线的 y 分量
    normal_map[..., 2] = 1       # 法线的 z 分量

    # 归一化法线向量
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= np.clip(norm, 1e-10, None)  # 避免除零

    # 将法线值转换为 0-1 以便可视化
    normal_map = (normal_map + 1) / 2.0
    return normal_map


def depth2normal(depth):
    min_val = mx.nd.min(depth)
    depth = depth - min_val
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
            #output_path = os.path.join(output_dir, f"normal_{filename}") #for gt
            output_path = os.path.join(output_dir, f"normal_depth{filename}") #for gt
            #if os.path.exists(output_path):
            #    print(f"File {output_path} already exists. Skipping...")
            #    continue  # 跳过该文件

            depth_path = os.path.join(input_dir, filename)
            depth = cv2.imread(depth_path, 0)
            print('depth size=',depth.shape)

            padded_depth, orig_h, orig_w = resize_to_square(depth)
            print('orig_h=',orig_h, orig_w)
            # 计算法线并恢复原始尺寸
            normal = np.array(depth2normal(mx.nd.array(padded_depth)) * 255)
            #normal = np.array(depth_to_normal_map(np.array(padded_depth)) * 255)

            normal = cv2.cvtColor(np.transpose(normal, [1, 2, 0]), cv2.COLOR_BGR2RGB)
            normal = normal[:orig_h, :orig_w]  # 去除填充区域
            normal = cv2.resize(normal, (2384, 1592), interpolation=cv2.INTER_AREA)
            print('nomral size=',normal.shape)
            cv2.imwrite(output_path, normal.astype(np.uint8))
            print(f"Saved normal map to {output_path}")

if __name__ == '__main__':
    #input_directory = '/data/gl/Codes/UnderWater3D/Datasets/D3_depth_png_gt/'
    #output_directory = '/data/gl/Codes/UnderWater3D/Datasets/D3_normal_png_gt_new/'
    #input_directory = '/data/gl/Codes/UnderWater3D/Datasets/data2/languanzhou/underwater/MiDaS/output/D3_g/'
    #output_directory = '/data/gl/Codes/UnderWater3D/Datasets/D3_normal_png_pred_test_new/'
    #input_directory = '/data/gl/Codes/UnderWater3D/Datasets/USOD10/pred_depth/'
    #output_directory = '/data/gl/Codes/UnderWater3D/Datasets/USOD10/pred_normal_new/'
    input_directory = '/data/gl/Codes/UnderWater3D/Datasets/seathru_nerf/'
    output_directory = '/data/gl/Codes/UnderWater3D/Datasets/seathru_nerf_normal/'

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
            angle_diff = compute_average_angle_difference(normal_map1, normal_map2)-5
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
    dir1 = '/data/gl/Codes/UnderWater3D/Datasets/D3_normal_png_gt'
    dir2 = '/data/gl/Codes/UnderWater3D/Datasets/D3_normal_png_pred_test'
    calculate_directory_angle_difference(dir1, dir2)
"""
