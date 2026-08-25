import os

IMAGE_DIR = r"C:\Users\Hassa\Downloads\Soccer Field Keypoints Detection.v6i.yolov5pytorch\train\images"
LABEL_DIR = r"C:\Users\Hassa\Downloads\Soccer Field Keypoints Detection.v6i.yolov5pytorch\train\labels"

image_paths = os.listdir(IMAGE_DIR)
label_paths = os.listdir(LABEL_DIR)

def clean_images():
    for image_path in image_paths:
        if "png" in image_path:
            new_name = image_path.split("_png.rf.")[0] + ".png"
        else:
            new_name = image_path.split("_jpg.rf.")[0] + ".jpg"
        
        os.rename(os.path.join(IMAGE_DIR, image_path), os.path.join(IMAGE_DIR, new_name))

def clean_labels():
    for label_path in label_paths:
        if "png" in label_path:
            new_name = label_path.split("_png.rf.")[0] + ".txt"
        else:
            new_name = label_path.split("_jpg.rf.")[0] + ".txt"
        
        os.rename(os.path.join(LABEL_DIR, label_path), os.path.join(LABEL_DIR, new_name))

clean_images()
clean_labels()



