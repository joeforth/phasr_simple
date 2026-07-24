# Phasr — Phase Detection for Soft Matter Formulation

Phasr fine-tunes a YOLO26 object detection model to identify and classify liquid
phases in photographs of vials of soft matter formulations (e.g.
Creaming, Sedimentation, Cracking, etc), clusters detections per
vial, flags incomplete/unexpected results, and exports the results to JSON/CSV.

## Features

- Fine-tuned YOLO26 model for 6 emulsion phase classes, loaded from
  [classes.txt](classes.txt)
- Tkinter GUI ([main.py](main.py)) for tuning detection settings on a single
  image and running batches, plus a headless batch script
  ([main_nogui.py](main_nogui.py)) for scripted/unattended runs
- Bounding-box visualisation with a per-class colour legend
- Manual cross-class overlap suppression (`processing.suppress_overlapping_boxes`)
  to catch duplicate detections that the model's end-to-end (NMS-free) head
  doesn't filter out on its own
- Structured per-vial results via dataclasses ([data_models.py](data_models.py):
  `Vial`, `ImageResult`), exported to JSON and CSV
- Warnings for low vial counts or vials with no phase detected
  (`processing.flag_warnings`)
- Training scripts for fine-tuning new YOLO26 checkpoints on a custom dataset
  labeled with [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) (see
  [training_and_data_handling/](training_and_data_handling/))

## Project structure

```
processing.py                     Core detection/visualisation/export logic
data_models.py                     Dataclasses (Vial, ImageResult) for structured per-vial results
main.py                            Tkinter GUI — primary way to run detection
main_nogui.py                      Headless batch script (edit settings at the top, then run)
classes.txt                        Ordered list of the 6 phase class names
trained_models/                    YOLO26 checkpoints (.pt) — gitignored, supply your own
images/                            Input images — gitignored
output_data/                       Generated JSON/CSV results (gitignored)
output_detections/                 Generated annotated images (gitignored)
training_and_data_handling/        Training scripts (train.py, train_mac.py, train_barkla.py, ...)
```

## Installation

```
conda create -n phasr python=3.12
conda activate phasr
nvidia-smi                         # check installed CUDA driver version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install ultralytics
cd phasr
pip install -r requirements.txt
```

Install `torch`/`torchvision` and `ultralytics` separately from
[requirements.txt](requirements.txt) — installing them together as one
resolve tends to pull in a mismatched build. Pick the `--index-url` CUDA
version (`cu132` above) to match the driver reported by `nvidia-smi`; use
[torch_check.py](torch_check.py) afterwards to confirm CUDA is detected.

Finally, place a trained model checkpoint (`.pt`) in `trained_models/` — these
are gitignored, so you'll need to supply your own or download a shared one
separately.

## Usage

### GUI (main.py)

```
python main.py
```

1. **Load Model** — select a `.pt` checkpoint from `trained_models/`.
2. Set `conf` / `iou` / `n_vials` in the Settings panel.
3. **Choose Input Image**, then **Detect Vials and Phases** to preview
   detections and tune settings.
4. **Save Data** — writes `output_data/<image>.json` + `.csv` and
   `output_detections/<image>_vis.<ext>`.
5. For multiple images: **Choose Directory** under Batch Processing, then
   **Run Batch** — processes every image in the folder and additionally
   writes a combined `output_data/batch.json` / `batch.csv`.

Warnings (e.g. fewer vials detected than `n_vials`, or a vial with no phase
detected) are logged in the Log / Warnings panel as each image is processed.

### Headless batch (main_nogui.py)

Edit the settings at the top of the script (`conf`, `iou`, `n_vials`,
`filedir`), then run:

```
python main_nogui.py
```

This processes every image in `filedir`, printing warnings to the console and
writing the same `output_data/` / `output_detections/` outputs as the GUI's
batch mode.

## Training a new model

Training scripts live in [training_and_data_handling/](training_and_data_handling/):

1. Label a dataset (bounding boxes per class) using
   [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) and export it in
   YOLO format into `training_data/`.
2. Point `data=` in [train.py](training_and_data_handling/train.py) (or the
   Mac/Barkla-cluster variants) at your exported `data.yaml`.
3. Choose a base checkpoint size (`yolo26n/s/m/l/x.pt`) — larger models
   generally perform better but train/infer more slowly.
4. Run the training script; weights are written to `runs/detect/<name>/weights/`.
5. Copy the resulting `best.pt` into `trained_models/` for use with `main.py`.

## License

This project's own code is MIT licensed — see [LICENSE.txt](LICENSE.txt).
It depends on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is licensed under AGPL-3.0.
