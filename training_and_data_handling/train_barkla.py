import yaml
from ultralytics import YOLO

def main():
    # Requires best_hyperparameters.yaml to be in ./
    with open("best_hyperparameters.yaml") as f:
        hyp = yaml.safe_load(f)

    model = YOLO("yolo26l.yaml")

    model.train(
        data="./data/data.yaml",
        epochs=500,
        imgsz=640,
        batch=16,
        device=0,
        workers=6,
        cache="ram",
        seed=0,
        patience=100,
        optimizer="AdamW"
        project="runs",
        name="yolo26l_yaml",
        **hyp,
    )

if __name__ == "__main__":
    main()