import os

def rename_files_in_folder(folder_path, prefix):
    """
    Renames all files in the specified folder with a given prefix and sequential numbering.

    Parameters:
    folder_path (str): The path of the folder containing files to rename.
    prefix (str): The prefix to add to the file names.
    """
    files = os.listdir(folder_path)  # Get a list of all files in the folder
    for index, filename in enumerate(files):
        # Generate new filename with prefix and zero-padded numbering
        new_name = f"{prefix}{index+1:02d}{os.path.splitext(filename)[1]}"
        # Rename the file
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

def main():
    """
    Main function that sets the folder paths and calls the rename function.
    """
    base_path = 'd:/Git/360-degree-Surround-View-application/Program/chessboard'
    # List of folders to process (modify as needed)
    # folders = ['front', 'left', 'rear', 'right']
    folders = ['left2']  # Currently processing only 'left2'

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):  # Check if the folder exists before renaming
            rename_files_in_folder(folder_path, 'chessboard')

if __name__ == "__main__":
    main()  # Execute the script
