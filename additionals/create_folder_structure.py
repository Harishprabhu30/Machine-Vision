## Run this script to create folder structure
import os


def create_folder_structure(base_path, folder_names, nested_folder = False):
    """
    function that creates a folder structure
    """

    try:
        if nested_folder = False:
            for folder in folder_names:
                folder_path = os.path.join(base_path, folder)
                os.makedirs(folder_path, exist_ok = True)
                print(f"Created {folder_path}")
        else:
            initial_folder_name = input("Enter First Folder Name: ")
            os.makedirs(base_path, exist_ok = True)

            initial_folder_path = os.path.join(base_path, intial_folder_name)
            os.makedirs(initial_folder_path, exist_ok = True)
            print()
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    folder_names = []
