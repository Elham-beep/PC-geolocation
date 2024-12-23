import laspy
import numpy as np
import json
import os
from datetime import datetime

'''This Python script performs the following tasks on a LAS (LiDAR) point cloud file:

Load the LAS File: Reads the input LAS file containing point cloud data with RGB color information.

Extract Point Coordinates and RGB Data: Retrieves the X, Y, Z coordinates and RGB color values of each point.

Define and Validate a Custom Rotation Matrix: Uses a provided 3x3 rotation matrix to rotate the point cloud.

Apply Rotation to Points: Rotates the point cloud around the origin without altering the original coordinate system.

Save the Transformed Point Cloud: Writes the rotated points back to a new LAS file, preserving the RGB data.'''


def load_las_file(file_path):
    """
    Loads a LAS file using laspy.

    Parameters:
        file_path (str): Path to the input LAS file.

    Returns:
        laspy.LasData: Loaded LAS data object.
    """
    try:
        las = laspy.read(file_path)
        print(f"Successfully loaded LAS file: {file_path}")
        return las
    except Exception as e:
        raise IOError(f"Error loading LAS file: {e}")

def extract_coordinates(las):
    """
    Extracts X, Y, Z coordinates from the LAS data.

    Parameters:
        las (laspy.LasData): LAS data object.

    Returns:
        np.ndarray: Nx3 array of point coordinates.
    """
    points = np.vstack((las.x, las.y, las.z)).transpose()
    print(f"Extracted {points.shape[0]} points from LAS file.")
    return points

def extract_rgb(las):
    """
    Extracts RGB color data from the LAS file.

    Parameters:
        las (laspy.LasData): LAS data object.

    Returns:
        np.ndarray: Nx3 array of RGB values.

    Raises:
        ValueError: If RGB data is not present in the LAS file.
    """
    color_dims = ['red', 'green', 'blue']
    if all(dim in las.point_format.dimension_names for dim in color_dims):
        rgb = np.vstack((las.red, las.green, las.blue)).transpose()
        print("RGB data successfully extracted from LAS file.")
        return rgb
    else:
        raise ValueError("Input LAS file does not contain RGB data.")

def define_rotation_matrix():
    """
    Defines the custom 3x3 rotation matrix.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    rotation_matrix = np.array([
        [0.724907219410, -0.688846528530, 0.000000000000],
        [0.688846528530, 0.724907219410, 0.000000000000],
        [0.000000000000 ,0.000000000000, 1.000000000000]
    ])
    print("Custom rotation matrix defined.")
    return rotation_matrix

def validate_rotation_matrix(rotation_matrix):
    """
    Validates that the rotation matrix is orthogonal and has a determinant of +1.

    Parameters:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.

    Raises:
        ValueError: If the rotation matrix is not orthogonal or its determinant is not +1.
    """
    orthogonality_check = np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.identity(3), atol=1e-6)
    determinant = np.linalg.det(rotation_matrix)
    determinant_check = np.isclose(determinant, 1.0, atol=1e-6)
    
    if not orthogonality_check:
        raise ValueError("The rotation matrix is not orthogonal.")
    if not determinant_check:
        raise ValueError(f"The rotation matrix does not have a determinant of +1 (determinant={determinant}).")
    
    print("Rotation matrix validation passed (orthogonal and determinant = +1).")

def apply_rotation(points, rotation_matrix):
    """
    Applies the rotation matrix to the point coordinates.

    Parameters:
        points (np.ndarray): Nx3 array of point coordinates.
        rotation_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Nx3 array of rotated point coordinates.
    """
    rotated_points = points.dot(rotation_matrix.T)
    print("Rotation matrix applied to point coordinates.")
    return rotated_points

def update_las_coordinates(las, rotated_points):
    """
    Updates the LAS data with rotated coordinates.

    Parameters:
        las (laspy.LasData): LAS data object.
        rotated_points (np.ndarray): Nx3 array of rotated point coordinates.
    """
    las.x = rotated_points[:, 0]
    las.y = rotated_points[:, 1]
    las.z = rotated_points[:, 2]
    print("LAS file coordinates updated with rotated points.")

def save_las_file(las, output_path):
    """
    Saves the LAS data to a new LAS file.

    Parameters:
        las (laspy.LasData): LAS data object with updated coordinates.
        output_path (str): Path to save the output LAS file.
    """
    try:
        las.write(output_path)
        print(f"Transformed LAS file saved as {output_path}")
    except Exception as e:
        raise IOError(f"Error saving LAS file: {e}")

def save_rotation_data(output_path, rotation_matrix, metadata=None):
    """
    Saves rotation data and metadata to a JSON file with the same base name as the output LAS file.

    Parameters:
        output_path (str): Path to the output LAS file.
        rotation_matrix (np.ndarray): 3x3 rotation matrix applied to the point cloud.
        metadata (dict, optional): Additional metadata to include in the JSON file.
    """
    # Define the JSON file path based on the output LAS file name
    base_name = os.path.splitext(output_path)[0]
    json_file_path = f"{base_name}.json"

    # Prepare the data to be saved
    data_to_save = {
        "rotation_matrix": rotation_matrix.tolist(),
        "timestamp": datetime.utcnow().isoformat() + "Z",  # UTC time in ISO format
        "metadata": metadata if metadata else {}
    }

    try:
        with open(json_file_path, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        print(f"Rotation data saved as {json_file_path}")
    except Exception as e:
        raise IOError(f"Error saving JSON file: {e}")

def main(input_file, output_file):
    """
    Main function to perform the rotation of point cloud data and save rotation metadata.

    Parameters:
        input_file (str): Path to the input LAS file.
        output_file (str): Path to save the rotated LAS file.
    """
    # Load the LAS file
    las = load_las_file(input_file)
    
    # Extract point coordinates
    points = extract_coordinates(las)
    
    # Extract RGB data
    rgb = extract_rgb(las)
    
    # Define the custom rotation matrix
    rotation_matrix = define_rotation_matrix()
    
    # Validate the rotation matrix
    validate_rotation_matrix(rotation_matrix)
    
    # Apply rotation to the points
    rotated_points = apply_rotation(points, rotation_matrix)
    
    # Update LAS data with rotated points
    update_las_coordinates(las, rotated_points)
    
    # Save the rotated point cloud to a new LAS file
    save_las_file(las, output_file)
    
    # Prepare additional metadata if needed
    additional_metadata = {
        "input_file": input_file,
        "output_file": output_file,
        "rotation_description": "Custom rotation applied using predefined rotation matrix.",
        "rotation_applied_by": "Your Name or System",  # Replace with appropriate identifier
        # Add more metadata fields as necessary
    }
    
    # Save rotation data and metadata to a JSON file
    save_rotation_data(output_file, rotation_matrix, metadata=additional_metadata)

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "C:/Users/bimax/Desktop/Scan2Bim-new-workflow/translated_point_cloud_auto.las"            # Replace with input LAS file path
    output_file = "C:/Users/bimax/Desktop/Scan2Bim-new-workflow/step1-normalisation/rotated_point_cloud3.las"  # Replace with desired output LAS file path
    
    # Execute the main function
    main(input_file, output_file)


