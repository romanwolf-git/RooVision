"""
This script checks for images without annotations and deletes them.
"""

import os
from pathlib import Path


def remove_empty_images(label_dir):
    """
    function to remove unannotated images from the dataset
    :param label_dir: label directory
    """

    # Iterate over files
    for file_path in sorted(label_dir.iterdir()):
        if file_path.is_file():
            # Open annotation file
            with file_path.open('r') as f:
                lines = f.readlines()
                # Check if the file is empty
                if len(lines) == 0:
                    # Construct the corresponding image file path
                    image_path = file_path.with_suffix('.jpg').parent / file_path.name.replace('labels',
                                                                                               'images').replace('txt',
                                                                                                                 'jpg')
                    # Remove the image file
                    image_path.unlink(missing_ok=True)
                    # Remove the label file
                    file_path.unlink()
                    # Print a message indicating the file has been removed
                    print(f'{file_path.name} removed.')


if __name__ == "__main__":

    LABEL_DIR = Path.cwd().parent.parent / 'data' / 'images' / 'test' / 'labels'
    remove_empty_images(label_dir=LABEL_DIR)
