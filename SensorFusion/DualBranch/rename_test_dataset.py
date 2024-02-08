import os

folder_path = "./data/rgb"  # replace with the path to your folder

for filename in os.listdir(folder_path):
    if "_Clipped" in filename:
        new_filename = filename.replace("_Clipped", "")
        os.rename(
            os.path.join(folder_path, filename), os.path.join(folder_path, new_filename)
        )
