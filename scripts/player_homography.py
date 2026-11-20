from ultralytics import YOLO
from ultralytics import SAM
import numpy as np 
import cv2
from sklearn.cluster import KMeans

keypoint_model = r"C:\Users\Hassa\Coding\Projects\3dfooty\runs\pose\model3\weights\best.pt"
player_detector = r"C:\Users\Hassa\Coding\Projects\3dfooty\player_detector.pt"

test_image = r"C:\Users\Hassa\Coding\Projects\3dfooty\dataset\images\2e57b9_5_7.png"
field_map = r"C:\Users\Hassa\Coding\Projects\3dfooty\birds_eye_view_field.png"

keypoint_detector = YOLO(keypoint_model)
player_detector = YOLO(player_detector)
player_segmentor = SAM()

field_keypoints = np.array([
    [47, 45],    # 0: TLC
    [47, 92],    # 1: L18BYTC
    [47, 145],   # 2: L6BYTC
    [47, 233],   # 3: L6BYBC
    [47, 286],   # 4: L18BYBC
    [47, 333],   # 5: BLC
    [73, 145],   # 6: L6TRC
    [73, 233],   # 7: L6BRC
    [99, 189],   # 8: LPK
    [126, 92],   # 9: L18TC
    [126, 153],  # 10: L18TPKA
    [126, 225],  # 11: L18BPKA
    [126, 286],  # 12: L18BC
    [263, 45],   # 13: HTT
    [263, 145],  # 14: HCT
    [263, 233],  # 15: HCB
    [263, 333],  # 16: HBT
    [401, 92],   # 17: R18TC
    [401, 153],  # 18: R18TPKA
    [401, 225],  # 19: R18BPKA
    [401, 286],  # 20: R18BC
    [427, 189],  # 21: RPK
    [454, 145],  # 22: R6TRC
    [454, 233],  # 23: R6BRC
    [480, 45],   # 24: TRC
    [480, 92],   # 25: R18BYTC
    [480, 145],  # 26: R6BYTC
    [480, 233],  # 27: R6BYBC
    [480, 286],  # 28: R18BYBC
    [480, 333],  # 29: BRC
    [219, 189],  # 30: HCL
    [307, 189],  # 31: HCR
], dtype=np.float32)

def detect_keypoints(img):
    result = keypoint_detector(img)[0]

    result = result.cpu()
    result.show()

    confidences = result.keypoints.conf.numpy()
    kps = result.keypoints.xy.numpy()

    valid_confidences = confidences[0] >= 0.75

    kps = kps[0][valid_confidences]
    return kps, valid_confidences

def detect_players(img):
    result = player_detector(img)[0]

    result = result.cpu()
    boxes = result.boxes.xyxy.numpy()
    classes = result.boxes.cls.numpy()

    player_mask = classes == 0
    ball_mask = classes == 1
    referee_mask = classes == 2

    player_coords = boxes[player_mask]
    ball_coords = boxes[ball_mask]
    referee_coords = boxes[referee_mask]

    return player_coords, ball_coords, referee_coords

def segment_players(img, detected_players):
    result = player_segmentor(img, bboxes=detected_players)
    
    print(result[0].masks.data)
    return result[0].masks.data.cpu().numpy()

def detect_teams(img, masks):
    image = cv2.imread(img)

    player_colors = []
    for mask in masks:
        filtered_image = image[mask]
        r_channel = filtered_image[:,2]
        g_channel = filtered_image[:,1]
        b_channel = filtered_image[:,0]

        avg_red = r_channel.mean()
        avg_green = g_channel.mean()
        avg_blue = b_channel.mean()

        player_colors.append([avg_red, avg_green, avg_blue])
    
    player_colors = np.array(player_colors)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(player_colors)
    
    return labels


def get_player_coords(coords):
    return ((coords[0] + coords[2]) / 2, coords[3])

def get_player_upper_coords(coords):
    return (coords[0], coords[1], coords[2], (coords[1] + coords[3]) // 2)

def get_homography(detected_kps, valid_confidences):
    return cv2.findHomography(detected_kps, field_keypoints[valid_confidences])

def display_player_positions(player_positions, team_assignments):
    field_image = cv2.imread(field_map)
    overlay = field_image.copy()

    height, width = overlay.shape[:2]

    for (x, y), team in zip(player_positions.reshape(-1, 2), team_assignments):
        if not np.isfinite([x, y]).all():
            continue

        point = (int(round(x)), int(round(y)))
        if 0 <= point[0] < width and 0 <= point[1] < height:

            if team == 0:
                color = (255, 0 , 0)
            else:
                color = (0, 0, 255)

            cv2.circle(overlay, point, 5, color, -1)
            cv2.circle(overlay, point, 5, (255, 255, 255), 1)

    cv2.imshow("Player positions", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_segments(masks, img):
    image = cv2.imread(img)
    fake_image = image.copy()

    colors = np.random.randint(0, 255, size=(len(masks), 3))

    for mask, color in zip(masks, colors):
        mask = mask.astype(bool)

        fake_image[mask] = (
            0.5 * fake_image[mask] + 0.5 * color
        ).astype(np.uint8)

    result = cv2.addWeighted(image, 0.4, fake_image, 0.6, 0)

    cv2.imshow("Player Segmentation", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detected_kps, valid_confidences = detect_keypoints(test_image)

    image = cv2.imread(test_image)
    fake_image = image.copy()

    for x, y in detected_kps:
        point = (int(round(x)), int(round(y)))
        cv2.circle(fake_image, point, 5, (0, 0, 255), -1)

    cv2.imshow("Detected Keypoints", fake_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detected_players, detected_ball, detected_referees = detect_players(test_image)

    masks = segment_players(test_image, detected_players)
    team_assigments = detect_teams(test_image, masks)
    visualize_segments(masks, test_image)

    player_coords = []
    for player in detected_players:
        player_coords.append(get_player_coords(player.tolist()))

    player_coords = np.array(player_coords)

    h, status = get_homography(detected_kps, valid_confidences)

    # approximate_field_points = cv2.perspectiveTransform(detected_kps, h)

    player_coords = player_coords.reshape(-1, 1, 2)
    player_positions = cv2.perspectiveTransform(player_coords, h)

    display_player_positions(player_positions, team_assigments)