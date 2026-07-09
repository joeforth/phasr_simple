from ultralytics import YOLO

def main():
    model = YOLO("yolo26m.pt")

    model.train(
        data="./data/data.yaml",
        epochs=600,
        imgsz=640,
        batch=8,          # explicit; try 8, drop to 4 if it still OOMs
        device="mps",
        workers=0,
        cache=False,      # or "disk" if I/O is the bottleneck
        seed=0,
        patience=100,
        close_mosaic=10,
        project="runs/detect",
        name="yolo26m_pt"
    )

if __name__ == "__main__":
    main()