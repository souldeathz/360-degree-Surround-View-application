import os

def rename_files_in_folder(folder_path, prefix):
    files = os.listdir(folder_path)
    for index, filename in enumerate(files):
        new_name = f"{prefix}{index+1:02d}{os.path.splitext(filename)[1]}"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

def main():
    base_path = 'd:/Git/360-degree-Surround-View-application/Program/chessboard'
    # folders = ['front', 'left', 'rear', 'right']
    folders = ['left2'] 

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            rename_files_in_folder(folder_path, 'chessboard')

if __name__ == "__main__":
    main()