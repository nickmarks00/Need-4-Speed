import cv2
import os

# Directory where your starting and ending images are stored
input_dir = "input_pics"

# Directory where you want to save the interpolated images
output_dir = "output_pics"

# Number of intermediate frames to generate
num_intermediate_frames = 2  # Adjust as needed

# Number of images in input folder
all_items = os.listdir(input_dir)

# Use a list comprehension to filter out only the files (not subfolders)
files = [item for item in all_items if os.path.isfile(os.path.join(input_dir, item))]

# Get the number of files
num_files = len(files)

# Iterate through your pairs of starting and ending images
for i in range(0, num_files - 1):  # Assuming you have 10 pairs
    # Load starting and ending images
    start_image_path = os.path.join(input_dir, f"img_{i}.png")
    end_image_path = os.path.join(input_dir, f"img_{i+1}.png")
    start_image = cv2.imread(start_image_path)
    end_image = cv2.imread(end_image_path)

    # Interpolate and save intermediate images
    for j in range(1, num_intermediate_frames + 1):
        alpha = j / (num_intermediate_frames + 1)
        interpolated_image = cv2.addWeighted(start_image, 1 - alpha, end_image, alpha, 0)
        output_path = os.path.join(output_dir, f"trajectory_{i}_{j}.png")
        cv2.imwrite(output_path, interpolated_image)
