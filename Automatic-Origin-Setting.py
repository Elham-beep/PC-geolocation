import open3d as o3d
import laspy
import numpy as np
import json
import sys

'''
Automatic Origin Setting by Finding the Point with the Lowest Coordinates

In this approach:

1. Identify the voxel with the lowest X, Y, and Z indices.
2. Set this voxel's center as the local origin.
3. Translate the entire point cloud relative to this new origin.
4. Shift the point cloud to have all coordinates >=0
5. (Optional) Normalize the point cloud.
6. Store all transformation parameters in a JSON file.
7. Save the transformed point cloud to a new .las file.
'''

def load_las_point_cloud(file_path):
    """
    Load a .las point cloud using laspy and convert it to Open3D format.

    Args:
        file_path (str): Path to the .las file.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    try:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        print(f"Loaded {len(pcd.points)} points from {file_path}")
        return pcd
    except Exception as e:
        print(f"Error loading LAS file: {e}")
        sys.exit(1)

def calculate_voxel_center(pcd, voxel_size=0.5):
    """
    Calculate the center of the voxel containing the point with the lowest X, then Y, then Z indices.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The size of each voxel.

    Returns:
        np.ndarray: The (Xg, Yg, Zg) coordinates of the voxel center.
        tuple: The voxel indices (XI, YI, ZI).
    """
    points = np.asarray(pcd.points)
    min_bound = pcd.get_min_bound()

    # Compute voxel indices for each point
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)

    # Step 1: Find the minimum X index
    min_x_idx = np.min(voxel_indices[:, 0])
    candidates_x = points[voxel_indices[:, 0] == min_x_idx]
    voxel_indices_x = voxel_indices[voxel_indices[:, 0] == min_x_idx]

    # Step 2: Within the minimum X indices, find the minimum Y index
    min_y_idx = np.min(voxel_indices_x[:, 1])
    candidates_xy = candidates_x[voxel_indices_x[:, 1] == min_y_idx]
    voxel_indices_xy = voxel_indices_x[voxel_indices_x[:, 1] == min_y_idx]

    # Step 3: Within the minimum X and Y indices, find the minimum Z index
    min_z_idx = np.min(voxel_indices_xy[:, 2])
    target_voxel_indices = voxel_indices_xy[:, 2] == min_z_idx
    target_voxel = np.where(target_voxel_indices)[0]

    if len(target_voxel) == 0:
        raise ValueError("No points found in the target voxel.")

    # Select the first point in the target voxel
    selected_point = candidates_xy[target_voxel_indices][0]

    # Compute the voxel center
    voxel_center_x = min_x_idx * voxel_size + voxel_size / 2 + min_bound[0]
    voxel_center_y = min_y_idx * voxel_size + voxel_size / 2 + min_bound[1]
    voxel_center_z = min_z_idx * voxel_size + voxel_size / 2 + min_bound[2]
    voxel_center = np.array([voxel_center_x, voxel_center_y, voxel_center_z])
    voxel_indices_tuple = (min_x_idx, min_y_idx, min_z_idx)

    print(f"Calculated voxel center at: {voxel_center}")
    print(f"Voxel indices: XI:{min_x_idx}, YI:{min_y_idx}, ZI:{min_z_idx}")

    return voxel_center, voxel_indices_tuple

def translate_to_local_origin(pcd, voxel_center):
    """
    Translate the point cloud so that the voxel center aligns with the global origin.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_center (np.ndarray): The (Xg, Yg, Zg) coordinates of the voxel center.

    Returns:
        o3d.geometry.PointCloud: The translated point cloud.
        dict: Transformation parameters.
    """
    # Translate the point cloud in place
    pcd.translate(-voxel_center, relative=True)

    transform_params = {
        "translation_vector": (-voxel_center).tolist(),
        "voxel_center": voxel_center.tolist()
    }

    print(f"Translated point cloud by vector: {transform_params['translation_vector']}")

    # Optional: Print new bounds to verify translation
    new_min_bound = pcd.get_min_bound()
    new_max_bound = pcd.get_max_bound()
    print(f"New Min Bound after translation: {new_min_bound}")
    print(f"New Max Bound after translation: {new_max_bound}")

    return pcd, transform_params

def shift_point_cloud_to_positive(pcd):
    """
    Shift the point cloud so that all coordinates are positive.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.

    Returns:
        o3d.geometry.PointCloud: The shifted point cloud.
        np.ndarray: The shift vector applied.
    """
    min_bound = pcd.get_min_bound()
    # Calculate the necessary shift to make all coordinates positive
    shift = np.maximum(0, -min_bound) + 1.0  # Add 1.0 as buffer to ensure positivity
    pcd.translate(shift, relative=True)
    print(f"Shifted point cloud by vector: {shift}")
    return pcd, shift

def normalize_point_cloud(pcd):
    """
    Normalize the point cloud by scaling it to fit within a unit cube.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.

    Returns:
        o3d.geometry.PointCloud: The normalized point cloud.
        dict: Transformation parameters.
    """
    if pcd.is_empty():
        print("Point cloud is empty. Cannot normalize.")
        return pcd, {}

    bounds = pcd.get_max_bound() - pcd.get_min_bound()
    max_dim = np.max(bounds)

    if max_dim > 0:
        scale = 1.0 / max_dim
        pcd_normalized = pcd.scale(scale, center=(0, 0, 0))
    else:
        print("Max dimension is zero or undefined. Setting scale to 1.")
        scale = 1.0
        pcd_normalized = pcd

    transform_params = {
        "scale": scale
    }

    print(f"Normalized point cloud with scale factor: {scale}")

    return pcd_normalized, transform_params

def save_transformation_to_json(transform_dict, filename):
    """
    Save transformation parameters to a JSON file.

    Args:
        transform_dict (dict): The transformation parameters.
        filename (str): The JSON file name.
    """
    # Define a recursive function to convert all numpy types to native types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(item) for item in obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert(obj.tolist())
        else:
            return obj

    transformed_dict = convert(transform_dict)

    try:
        with open(filename, 'w') as f:
            json.dump(transformed_dict, f, indent=4)
        print(f"Transformation parameters saved to {filename}")
    except Exception as e:
        print(f"Error saving transformation parameters to JSON: {e}")
        sys.exit(1)

def save_point_cloud_to_las(pcd, original_las_path, output_las_path):
    """
    Save the transformed Open3D point cloud back to a .las file, preserving original LAS header.

    Args:
        pcd (o3d.geometry.PointCloud): The transformed point cloud.
        original_las_path (str): Path to the original .las file.
        output_las_path (str): Path to save the transformed .las file.
    """
    try:
        las = laspy.read(original_las_path)
    except Exception as e:
        print(f"Error reading LAS file: {e}")
        sys.exit(1)

    points = np.asarray(pcd.points)

    # Check if there are points to process
    if len(points) == 0:
        print("No points to save. Exiting.")
        sys.exit(1)

    # Find the new min and max bounds
    new_min_bound = pcd.get_min_bound()
    new_max_bound = pcd.get_max_bound()

    print(f"New Min Bound: {new_min_bound}")
    print(f"New Max Bound: {new_max_bound}")

    # Retrieve the original scale factors from the header
    scale_x, scale_y, scale_z = las.header.scale
    print(f"Original LAS scale factors: X={scale_x}, Y={scale_y}, Z={scale_z}")

    # Update the LAS header's offset to [0, 0, 0] first
    las.header.offset = [0, 0, 0]
    print(f"Updated LAS header offset to: {las.header.offset}")

    # Update LAS header's min bounds to the new minimum coordinates
    las.header.min = [new_min_bound[0], new_min_bound[1], new_min_bound[2]]
    print(f"Updated LAS header min to: {las.header.min}")

    # Now, calculate scaled coordinates
    x_scaled = np.round(points[:, 0] / scale_x).astype(np.int32)
    y_scaled = np.round(points[:, 1] / scale_y).astype(np.int32)
    z_scaled = np.round(points[:, 2] / scale_z).astype(np.int32)

    # Debugging: Print min and max of scaled coordinates
    print(f"Scaled X coordinates: min={x_scaled.min()}, max={x_scaled.max()}")
    print(f"Scaled Y coordinates: min={y_scaled.min()}, max={y_scaled.max()}")
    print(f"Scaled Z coordinates: min={z_scaled.min()}, max={z_scaled.max()}")

    # Check for overflow
    if (x_scaled < 0).any() or (y_scaled < 0).any() or (z_scaled < 0).any():
        raise OverflowError("Negative coordinates detected after scaling.")

    max_las_int = 2**31 - 1
    if (x_scaled > max_las_int).any() or (y_scaled > max_las_int).any() or (z_scaled > max_las_int).any():
        raise OverflowError("Scaled coordinates exceed LAS integer limits.")

    # Assign the scaled integer coordinates to the LAS file
    try:
        las.x = x_scaled
        las.y = y_scaled
        las.z = z_scaled
    except OverflowError as e:
        print(f"Error setting LAS coordinates: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error setting LAS coordinates: {e}")
        sys.exit(1)

    # Let laspy automatically update header.min and header.max based on new coordinates
    # No need to set las.header.max manually

    # Save the LAS file
    try:
        las.write(output_las_path)
        print(f"Transformed point cloud saved to {output_las_path}")
    except Exception as e:
        print(f"Error writing LAS file: {e}")
        sys.exit(1)





def main_automatic_origin():
    """
    Main function for Approach 2: Automatic Origin Setting.
    """
    # Paths
    input_las = "C:/Users/bimax/Desktop/Scan2Bim-new-workflow/down-left-part-subsample-20m.las"  # Replace with your input .las file path
    output_las = "C:/Users/bimax/Desktop/Scan2Bim-new-workflow/translated_point_cloud_auto.las"
    json_file = "C:/Users/bimax/Desktop/Scan2Bim-new-workflow/transformation_params_auto.json"
    # rotation_json = "rotation_matrix.json"  # JSON file containing the rotation matrix

    print("Starting Automatic Origin Setting Process...\n")

    # Step 1: Load the point cloud
    pcd = load_las_point_cloud(input_las)

    # Step 2: Calculate the voxel center with lowest coordinates
    voxel_size = 0.2  # Adjust based on your point cloud scale
    try:
        voxel_center, voxel_indices = calculate_voxel_center(pcd, voxel_size=voxel_size)
    except ValueError as e:
        print(f"Error in calculate_voxel_center: {e}")
        sys.exit(1)

    # Step 3: Translate to local origin using calculated voxel center
    try:
        pcd_translated, translation_params = translate_to_local_origin(pcd, voxel_center)
    except Exception as e:
        print(f"Error in translate_to_local_origin: {e}")
        sys.exit(1)

    # Step 4: Shift the point cloud to have positive coordinates
    try:
        pcd_shifted, shift = shift_point_cloud_to_positive(pcd_translated)
    except Exception as e:
        print(f"Error in shift_point_cloud_to_positive: {e}")
        sys.exit(1)

    # Step 5: Normalize the point cloud
    # Commenting out normalization as per your request
    # try:
    #     pcd_normalized, normalization_params = normalize_point_cloud(pcd_shifted)
    # except Exception as e:
    #     print(f"Error in normalize_point_cloud: {e}")
    #     sys.exit(1)

    # If normalization is commented out, set scale to 1.0 and exclude it from transformation_params
    normalization_params = {}

    # Step 6: Collect all transformation parameters
    transformation_params = {
        # "rotation_matrix": rotation_matrix_manual.tolist(),
        "translation_vector": translation_params["translation_vector"],
        "shift_vector": shift.tolist(),
        "scale": normalization_params.get("scale", 1.0),
        "voxel_indices": {
            "XI": int(voxel_indices[0]),
            "YI": int(voxel_indices[1]),
            "ZI": int(voxel_indices[2])
        },
        "voxel_center": translation_params["voxel_center"]
    }

    # Step 7: Save transformation parameters to JSON
    try:
        save_transformation_to_json(transformation_params, json_file)
    except Exception as e:
        print(f"Error in save_transformation_to_json: {e}")
        sys.exit(1)

    # Step 8: Save the transformed point cloud back to LAS
    try:
        save_point_cloud_to_las(pcd_shifted, input_las, output_las)
    except Exception as e:
        print(f"Error in save_point_cloud_to_las: {e}")
        sys.exit(1)

    print("\nTransformation complete.")

if __name__ == "__main__":
    try:
        main_automatic_origin()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
