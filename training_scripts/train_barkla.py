from ultralytics import YOLO

def main():
  model = YOLO("yolo26l.yaml")
    
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
        lr0=0.00504,
        lrf=0.01,
        momentum=0.88535,
        weight_decay=0.00083,
        warmup_epochs=2.65386,
        warmup_momentum=0.77011,
        box=8.42975,
        cls=0.55703,
        cls_pw=0.0,
        dfl=2.70051,
        hsv_h=0.0118,
        hsv_s=0.82671,
        hsv_v=0.36187,
        degrees=0.0022,
        translate=0.11543,
        scale=0.50516,
        shear=0.00892,
        perspective=0.00037,
        flipud=0.0014,
        fliplr=0.50265,
        bgr=0.00308,
        mosaic=0.781,
        mixup=0.00149,
        cutmix=0.00289,
        copy_paste=0.01467,
        close_mosaic=8,
        project="runs",
        name="yolo26l_pt"
      )

if __name__ == "__main__":
    main()