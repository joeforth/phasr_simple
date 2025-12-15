from ultralytics import YOLO
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

model = YOLO("yolo11l.yaml")

results = model.train(task="detect",data="yourdataset/data.yaml", name="ShampooSTFA", device = [0,1], 
                                                                              # Hyperparameters
                                                                              epochs=150,
                                                                              patience=150,
                                                                              batch=8,
                                                                              imgsz=640, 
                                                                              workers= 1,
                                                                              bgr= 0.0,
                                                                              box= 0.1887,
                                                                              cls= 0.27033,
                                                                              lr0= 0.0010317,
                                                                              lrf= 0.64506,
                                                                              momentum= 0.64506,
                                                                              warmup_epochs= 3.1,
                                                                              optimizer= "AdamW",
                                                                              warmup_momentum= 0.45676,
                                                                              weight_decay= 0.0005
                                                                              # Augmentation hyperparameters
                                                                              cutmix= 0.0,
                                                                              degrees= 0.0,
                                                                              fliplr= 0.5,
                                                                              flipud= 0.0,
                                                                              hsv_h= 0.3,
                                                                              hsv_s= 0.6872304839691561,
                                                                              hsv_v= 0.8889545777864033,
                                                                              mixup= 0.0,
                                                                              mosaic= 0.0,
                                                                              perspective= 0.01,
                                                                              scale= 0.5,
                                                                              shear= 0.0,
                                                                              translate= 0.5,
                                                                              ) 
