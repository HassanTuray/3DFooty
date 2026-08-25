from pathlib import Path
import random
import cv2
import os

IMAGES_DIR = Path(
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images"
)

LABELS_DIR = Path(
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\labels"
)

file_extensions = [".jpg", ".jpeg", ".png"]

random.seed(42)

def split_data(train_dir, split=0.7):
    '''
    1. Randomize labels in label dir
    2. Move data to correct folders
        - calculate split point
        - create file to write to in the train dir
        - Iterate through randomize list up till split point
            - get the corresponding image from the label
            - write file paths to the correct file 
    '''

    label_path_list = os.listdir(LABELS_DIR)
    annotation_set = set(map(lambda x: x[:-4], label_path_list))
    image_path_list = os.listdir(IMAGES_DIR)

    # image_path_list = []

    # for image in image_path_list1:
    #     if "frame" in image:
    #         game = int(image[6:7])
    #         image_num = int(image[8:][:-4])
    #         if image_num <= 100 and game == 0:
    #             image_path_list.append(image)

    annotated_image_path_list = []
    for image_path in image_path_list:
        if image_path[:-4] in annotation_set:
            annotated_image_path_list.append(image_path)

    random.shuffle(annotated_image_path_list)

    split_index = len(annotated_image_path_list) * split

    train_text = os.path.join(train_dir, "train.txt")
    val_text = os.path.join(train_dir, "val.txt")

    with (
        open(train_text, "w") as train_file,
        open(val_text, "w") as val_file
    ):
        file_count = 0
        for image_path in annotated_image_path_list:
            
            label_path = image_path[:-4] + ".txt"

            if (file_count < split_index):
                train_file.write(os.path.join(IMAGES_DIR, image_path) + "\n")
            else:
                val_file.write(os.path.join(IMAGES_DIR, image_path) + "\n")
            
            file_count += 1

split_data(r"C:\Users\Hassa\Coding\Projects\3dfooty\model_configs\model3")

