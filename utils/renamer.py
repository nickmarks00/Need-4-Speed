"""
This takes a folder of images and updates the starting index.
Useful when combining disparate datasets and you didn't set the config.txt
flag properly, such as "straight", "curved" and "loop" datasets.
"""

import os


def batch_image_rename():
    # Directory containing the images
    directory = input("Specify a path to the images folder: ")
    print(f"Using images folder at {directory}")

    # New starting number for the images
    new_starting_number = int(input("Give the new starting index: "))
    print(f"Using starting index {new_starting_number}")

    # Get list of files in the directory
    files = os.listdir(directory)

    # Filter out only the image files
    image_files = [file for file in files if file.endswith(".png")]
    indices = sorted([int(file.split("_")[1].split(".")[0]) for file in image_files])

    # Rename the images
    for i, old_index in enumerate(indices):
        old_name = f"img_{old_index}.png"
        new_index = new_starting_number + i
        new_name = f"img_{new_index}.png"
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")

    print("All images renamed successfully.")


if __name__ == "__main__":
    batch_image_rename()
