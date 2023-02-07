from pathlib import Path
import os
import shutil
import glob

directory_master = "COVID-CT-master/"

folder_master = Path(directory_master)
print(f"folder_master : {folder_master}")
folder_data_split = directory_master+"Data-split/"# Change the string to the actual path to the .txt file.
print(f"folder_data_split : {folder_data_split}")
folder_with_images_path = directory_master+"Images-processed/"
folder_to_copy_to_path = directory_master+"Dataset/"

if not os.path.exists(folder_to_copy_to_path):
   os.makedirs(folder_to_copy_to_path)
for type in ["test","train","val"]:
    if not os.path.exists(folder_to_copy_to_path+type):
        os.makedirs(folder_to_copy_to_path+type)
        for label in ["COVID","NonCOVID"]:
            if not os.path.exists(folder_to_copy_to_path+type+"/"+label):
                os.makedirs(folder_to_copy_to_path+type+"/"+label)

for label in ["COVID","NonCOVID"]:
    #path = folder_data_split+label+"/*.txt"
    path = folder_data_split+label
    for file in os.listdir(path):
        if file.endswith(".txt"):
            print(f"file : {file}")
            type_data = file.split(".")[0].split("CT")[0]
            print(f"type_data : {type_data}")
            with open(path+"/"+file) as txt_file:
                images_to_copy_list = txt_file.read().split('\n')
            for image in images_to_copy_list:
                print(f"image : {image}")
                print(f"label : {label}")
                dirs = os.listdir(folder_with_images_path+"CT_"+label)
                if image in dirs:
                    print("true")
                    shutil.copy(folder_with_images_path+"CT_"+label+"/"+image, folder_to_copy_to_path+type_data+"/"+label+"/"+image)
        