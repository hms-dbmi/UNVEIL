from pathlib import PosixPath
import os
from typing import List
import torch

from pathlib import Path, PosixPath
from typing import List, Union

def get_files_in_dir(dir_path: Union[PosixPath, str], extension: str = "") -> List[PosixPath]:
    """
    Returns a list of files in the given directory and its subdirectories.

    Args:
        dir_path (PosixPath | str): Directory path.
        extension (str): File extension filter (optional).

    Returns:
        List[PosixPath]: List of files in the directory and subdirectories.

    Raises:
        ValueError: If the path is not a directory.
    """

    # Convert to PosixPath if necessary
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    # Check if the provided path is a directory
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a valid directory")
    
    # Use rglob to recursively find files with the specified extension
    return [file for file in dir_path.rglob(f"*{extension}") if file.is_file()]



def get_device() -> str:
    """ Gets the available device in order of decreasing priority. """

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    

def safe_create_directory(dir_path):
    
    # Check if the directory already exists
    if os.path.exists(dir_path):
        while True:
            user_input = input(
                f"Directory '{dir_path}' already exists. Do you want to proceed? (yes/no): "
            ).strip().lower()
            if user_input == 'yes':
                break  # Proceed if user says yes
            elif user_input == 'no':
                print("Operation cancelled.")
                exit()
            else:
                print("Please answer 'yes' or 'no'.")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
