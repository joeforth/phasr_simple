from dataclasses import dataclass, field

# Example calls to DataClasses:
# for idx, v in enumerate(detections_class.vials):
#     print('Vial ', idx)
#     print(v)

@dataclass
class Vial:
    vial_ind: int
    vial_conf: float
    vial_bbox_x1y1_x2y2: list[float]
    phase_ind: int
    phase_name: str
    phase_score: int
    phase_conf: float
    phase_bbox_x1y1_x2y2: list[float]
    flag: str

@dataclass
class ImageResult:
    filename: str
    vials: list[Vial] = field(default_factory=list)