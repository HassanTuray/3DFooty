
import os
from ultralytics import YOLO
import random
import yaml
import cv2

IMAGE_DIR = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images"
LABEL_DIR = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\labels"
TEMP_DIR = r"C:\Users\Hassa\Coding\Projects\3dfooty\scripts\tmp_images"

random.seed(42)

image_paths = os.listdir(IMAGE_DIR)

model = YOLO(r"C:\Users\Hassa\Coding\Projects\3dfooty\runs\pose\model2_vertical_fine_tuned-2\weights\best.pt")

dataset1_filter = (lambda x: not ("frame" in x or "screenshot" in x or "dila" in x))
dataset2_filter = (lambda x: "frame" in x)
dataset3_filter = (lambda x: "screenshot" in x or "dila" in x)

dataset1 = [
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\0a2d9b_2_12.png",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\0a2d9b_6_11.png",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\2e57b9_3_1.png",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\08fd33_0_9.png",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\ec783a_7_3.png"
]

dataset2 = [
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_0_1.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_0_3.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_0_5.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_0_22.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_0_30.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_0_76.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_1_18.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_2_1.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_2_10.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_2_12.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_2_62.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_2_87.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\frame_2_100.jpg"
]

dataset3 = [
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\dila_samtr1.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\dila_samtr6.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\dila_samtr8.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\dila_samtr45.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\dila_samtr12.jpg",
    r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\screenshot_19.jpg"
]

SKELETON_CONNECTIONS = [
    # left byline
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),

    # top touchline
    (0, 13),
    (13, 24),

    # bottom touchline
    (5, 16),
    (16, 29),

    # Left 6-yard box
    (2, 6),
    (6, 7),
    (7, 3),

    # left 18 yard box
    (1, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 4),

    # Center line
    (13, 14),
    (15, 16),
    (14, 30),
    (14, 31),
    (15, 30),
    (15, 31),

    # right 18 yard box
    (17, 25),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 28),

    # Right 6-yard box
    (22, 23),
    (22, 26),
    (23, 27),

    # right byline
    (24, 25),
    (25, 26),
    (26, 27),
    (27, 28),
    (28, 29),

    # Center / halfway connections
    (13, 14),
    (15, 16)
]

def validate_all_date():
    for data in dataset1:
        results = model(data)
        for result in results:
            result.save(os.path.join(TEMP_DIR, os.path.basename(data)))

    for data in dataset2:
        results = model(data)
        for result in results:
            result.save(os.path.join(TEMP_DIR, os.path.basename(data)))

    for data in dataset3:
        results = model(data)
        for result in results:
            result.save(os.path.join(TEMP_DIR, os.path.basename(data)))

def validate_dataset(
    dataset_number,
    show_confidence=False,
    show_names=False,
    show_skeleton=False,
    keypoint_threshold=0.5
):
    if dataset_number == 1:
        images = list(filter(dataset1_filter, image_paths))
    elif dataset_number == 2:
        images = list(filter(dataset2_filter, image_paths))
    elif dataset_number == 3:
        images = list(filter(dataset3_filter, image_paths))
    else:
        raise ValueError("dataset_number must be 1, 2, or 3")

    with open(r"C:\Users\Hassa\Coding\Projects\3dfooty\model_configs\model1\data.yaml", "r") as f:
        data = yaml.safe_load(f)

    kpt_names = data["kpt_names"]

    for image_name in images:
        image_path = os.path.join(IMAGE_DIR, image_name)

        image = cv2.imread(image_path)

        results = model(image_path)

        for result in results:

            keypoints = result.keypoints.xy.cpu().numpy()

            confidences = None
            if result.keypoints.conf is not None:
                confidences = result.keypoints.conf.cpu().numpy()

            for detection_idx, detection in enumerate(keypoints):

                if show_skeleton:
                    for idx1, idx2 in SKELETON_CONNECTIONS:
                        if (
                            confidences is not None
                            and confidences[detection_idx][idx1] >= keypoint_threshold
                            and confidences[detection_idx][idx2] >= keypoint_threshold
                        ):
                            x1, y1 = detection[idx1]
                            x2, y2 = detection[idx2]

                            cv2.line(
                                image,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA
                            )

                for kpt_idx, (x, y) in enumerate(detection):

                    if (
                        confidences is not None
                        and confidences[detection_idx][kpt_idx] < keypoint_threshold
                    ):
                        continue

                    cv2.circle(
                        image,
                        (int(x), int(y)),
                        5,
                        (0, 255, 0),
                        -1
                    )

                    label_parts = []

                    if show_names:
                        label_parts.append(kpt_names[kpt_idx])

                    if show_confidence and confidences is not None:
                        confidence = confidences[detection_idx][kpt_idx]
                        label_parts.append(f"{confidence:.2f}")

                    if label_parts:
                        label = " ".join(label_parts)

                        cv2.putText(
                            image,
                            label,
                            (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

            output_path = os.path.join(
                TEMP_DIR,
                os.path.basename(image_name)
            )

            cv2.imwrite(output_path, image)

validate_dataset(1, show_names=True, show_confidence=True, show_skeleton=True)
validate_dataset(2, show_names=True, show_confidence=True, show_skeleton=True)
validate_dataset(3, show_names=True, show_confidence=True, show_skeleton=True)






