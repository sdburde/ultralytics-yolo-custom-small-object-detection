from ultralytics import YOLO

model = YOLO('yolo11m.pt')

results = model.train(
    data = 'yolo_det.yaml',
    batch = 4,
    epochs = 400,
    imgsz = 1280,
    device = 0,
    name = '20250825_1280_yolov11m_tau_10000-otter',
    verbose = True, 
    workers=8, 
    resume=False, 
    val=True,
    optimizer = 'AdamW',
    lr0=0.00001,
    lrf= 0.0001,
    cos_lr=True,
    single_cls=True,
    degrees=8,
    mosaic=0.95,
    fliplr= 0.5,
    translate= 0.1,
    scale= 0.05,
    augment= True
)