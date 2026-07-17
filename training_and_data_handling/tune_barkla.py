from ultralytics import YOLO

def main():
  model = YOLO("yolo26m.yaml")

  model.tune(
      data="/mnt/scratch/users/jwforth/data/data.yaml",
      epochs=30,
      imgsz=640,
      iterations=100,
      optimizer="AdamW",
      plots=False,
      batch=16,
      device=0,
      workers=6,
      cache="ram",
      project="runs",
      name="yolo26myaml_tune",
  )

if __name__ == "__main__":
    main()