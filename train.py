from ultralytics import YOLO

def main():
    model = YOLO("yolo26m.pt")

    model.train(
        data="./data/data.yaml",
        epochs=600,
        imgsz=640,
        batch=-1,
        device=0,
        workers=0,
        cache="ram",
        seed=0,
        patience=100,
        close_mosaic=10,
        project="runs/detect",
        name="yolo26m_pt"
    )

if __name__ == "__main__":
    main()