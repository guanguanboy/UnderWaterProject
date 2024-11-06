import numpy as np

# Load the poses_bounds.npy file
poses_bounds = np.load('poses_bounds.npy')

# Print the shape of the array
print(f'Shape of poses_bounds: {poses_bounds.shape}')

# Print the first few poses and bounds
for i in range(min(5, poses_bounds.shape[0])):  # Print first 5 entries
    print(f'Pose {i}: {poses_bounds[i][:12]}')  # Rotation and translation
    print(f'Bounds {i}: {poses_bounds[i][12:]}')  # Bounds
