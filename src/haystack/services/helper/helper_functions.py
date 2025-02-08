import glob

def get_all_file_paths(folder_path):
    # Use glob to get all file paths in the folder
    file_paths = glob.glob(folder_path + "/*")
    return file_paths