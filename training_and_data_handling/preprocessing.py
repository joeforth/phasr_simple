import ultralytics
from ultralytics.data.split import autosplit

autosplit(
    path="./data/images/",
    weights=(0.8, 0.1, 0.1),  # (train, validation, test) fractional splits
    annotated_only=True,  # split only images with annotation file when True
)