from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\Hassa\Coding\Projects\3dfooty\runs\pose\model2_vertical_fine_tuned\weights\best.pt")

    model.train(
        epochs=600,
        data=r"C:\Users\Hassa\Coding\Projects\3dfooty\model_configs\model2\data.yaml", 
        name="model2_vertical_fine_tuned",
        batch=8,
        workers=8,
        device=0,
        lr0=0.01,
        patience=600,
        mosaic=0.0
    )

if __name__ == "__main__":
    main()