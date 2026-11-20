from ultralyics import YOLO
player_detector = r"C:\Users\Hassa\Coding\Projects\3dfooty\player_detector.pt"

model = YOLO(player_detector)

def detect_players(img):
    result = model(img)[0]

    return result.boxes.xyxy
    