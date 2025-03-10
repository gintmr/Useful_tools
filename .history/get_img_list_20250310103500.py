import os
import shutil

def copy_images_from_subfolders(src_folder, dst_folder, img_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    """
    Copy all images from subfolders of src_folder to dst_folder.

    :param src_folder: The path to the parent folder.
    :param dst_folder: The path to the destination folder.
    :param img_extensions: A list of file extensions to consider as images.
    """
    # Create the destination folder if it does not exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Walk through all subfolders and files in the source folder
    for root, dirs, files in os.walk(src_folder):
        if len(files) != 10:
            print(root)
        for file in files:
            # Check if the file is an image based on its extension
            if any(file.lower().endswith(ext) for ext in img_extensions):
                # Construct full file path
                file_path = os.path.join(root, file)
                
                # Construct destination file path
                rel_path = os.path.relpath(root, src_folder)
                ##ddd root is the full path of the file，src_folder is the parent folder
                ##ddd rel_path is the relative path of the file
                dst_path = os.path.join(dst_folder, file)
                
                # # Create subfolder in destination if it does not exist
                # dst_subfolder = os.path.join(dst_folder, rel_path)
                # if not os.path.exists(dst_subfolder):
                #     os.makedirs(dst_subfolder)
                
                # Copy the file to the destination
                shutil.copy(file_path, dst_path)
                # print(f"Copied: {file_path} to {dst_path}")

# Example usage
src_folder = '/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/train'
dst_folder = '/data2/wuxinrui/Projects/ICCV/MIMC_FINAL/seen/train_list'
copy_images_from_subfolders(src_folder, dst_folder)