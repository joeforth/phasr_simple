import ultralytics
from ultralytics.data.split import autosplit

autosplit(
    path="./data/images/",
    weights=(0.7, 0.15, 0.15),  # (train, validation, test) fractional splits
    annotated_only=False,  # split only images with annotation file when True
)