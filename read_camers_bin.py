import pycolmap
 
# 指定cameras.bin文件的路径
cameras_path = 'SeathruNeRF_dataset/Panama/sparse/0/'
 
import pycolmap
 
# 加载重建模型
reconstruction = pycolmap.Reconstruction(cameras_path)
 
# 访问相机信息
for camera_id, camera in reconstruction.cameras.items():
    print(f"Camera ID: {camera_id}")
    print(f"Focal Length: {camera.focal_length}")
    print(f"Focal Length_x: {camera.focal_length_x}")
    print(f"Focal Length_y: {camera.focal_length_y}")
    print(f"height: {camera.height}")
    print(f"width: {camera.width}")

    print(f"Principal Point: {camera.principal_point_x}, {camera.principal_point_y}")

"""
# 访问图像信息
for image_id, image in reconstruction.images.items():
    print(f"Image ID: {image_id}")
    #(f"Rotation Matrix: {image.rotmat()}")
    #print(f"Translation Vector: {image.tvec}")
    print(f"Camera ID: {image.camera_id}")
    print("Keypoints and their corresponding 3D points:")
    for p2d in image.points2D:
        if p2d.has_point3D():
            print(f"  2D Point: {p2d.xy} -> 3D Point ID: {p2d.point3D_id}")

# 访问三维点信息
for point3D_id, point3D in reconstruction.points3D.items():
    print(f"3D Point ID: {point3D_id}")
    print(f"Coordinates: {point3D.xyz}")
    #print(f"Observed by {len(point3D.image_ids)} images")

"""