import numpy as np
import struct

def read_cameras(file_path):
    cameras = {}
    
    with open(file_path, 'rb') as f:
        # Read the number of cameras
        num_cameras = struct.unpack('I', f.read(4))[0]

        for _ in range(num_cameras):
            # Read camera ID
            camera_id = struct.unpack('I', f.read(4))[0]
            
            # Read model type
            model_type = struct.unpack('I', f.read(4))[0]
            
            # Read width and height
            width = struct.unpack('I', f.read(4))[0]
            height = struct.unpack('I', f.read(4))[0]
            
            # Read parameters, the number of parameters depends on the model
            if model_type == 0:  # PINHOLE model
                params = struct.unpack('ffff', f.read(16))  # fx, fy, cx, cy
            elif model_type == 1:  # SIMPLE_RADIAL model
                params = struct.unpack('fff', f.read(12))  # fx, fy, cx, k1
            # You can add more models as needed
            
            cameras[camera_id] = {
                'model_type': model_type,
                'width': width,
                'height': height,
                'parameters': params
            }
    
    return cameras

# Usage example
cameras = read_cameras('SeathruNeRF_dataset/Curasao/sparse/0/cameras.bin')
for camera_id, camera_info in cameras.items():
    print(f'Camera ID: {camera_id}, Info: {camera_info}')
