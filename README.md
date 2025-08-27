# ultralytics-YOLO-Custom
## YOLOv11 Training Changes

### Summary
    - Modified to support separate true positive (TP) and false positive (FP) dataset paths for training and validation, 
    - TP and FP dataset handling in mosaic augmentation (2TP and 2FP in 4 tile mosiac). 
    - Additional change includes adjusting the EMA tau value for improved model stability.

### Changes
1. **yolo_det.yaml**
   - Updated dataset paths to include `train_tp`, `val_tp`, `train_fp`, and `val_fp` for TP/FP separation.

2. **ultralytics/models/yolo/detect/train.py**
   - Modified `build_dataset` to accept `tp_path` and `fp_path` for TP/FP dataset handling.

3. **ultralytics/data/utils.py**
   - Added logic in `check_det_dataset` to combine `train_tp`/`train_fp` and `val_tp`/`val_fp` into `train` and `val` lists.

4. **ultralytics/data/base.py**
   - Enhanced `BaseDataset` to handle `tp_path` and `fp_path`, combining TP/FP files into an alternating list.
   - Added dataset root path resolution for relative paths.

5. **ultralytics/data/dataset.py**
   - Updated `YOLODataset` to pass `tp_path` and `fp_path` to `BaseDataset`.
   - Added `is_tp` flag to labels and improved `collate_fn` for robust batch collation.

6. **ultralytics/data/augment.py**
   - Modified `Mosaic` class to alternate TP/FP samples in mosaic augmentation.
   - Adjusted mosaic center and label handling for balanced TP/FP inclusion.

7. **ultralytics/utils/torch_utils.py**
   - Updated `ModelEMA` class to increase the `tau` parameter from 2000 to 10000, improving the exponential moving average stability for model weights during training for large ammount of image data.