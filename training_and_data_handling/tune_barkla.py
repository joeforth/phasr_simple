from ultralytics import YOLO

def main():
  model = YOLO("yolo26l.yaml")

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
      patience=100,
      close_mosaic=10,
      project="runs",
      name="yolo26lyaml_tune",
  )

if __name__ == "__main__":
    main()