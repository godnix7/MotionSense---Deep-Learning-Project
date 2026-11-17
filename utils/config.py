"""Central configuration for MotionSense."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASETS_DIR = ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
UCF_DIR = DATASETS_DIR / "UCF101"                 # extracted videos: UCF101/<Class>/*.avi
SPLITS_DIR = DATASETS_DIR / "ucfTrainTestlist"    # official split files
FRAMES_DIR = DATASETS_DIR / "frames"              # preprocessed frame cache (.npy per video)
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
REPORTS_DIR = OUTPUT_DIR / "reports"
SAMPLES_DIR = ROOT / "samples"

UCF101_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
SPLITS_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"

CLASSES = [
    "JumpingJack",
    "PushUps",
    "BodyWeightSquats",
    "Lunges",
    "TaiChi",
    "BoxingSpeedBag",
]

DISPLAY_NAMES = {
    "JumpingJack": "Jumping Jack",
    "PushUps": "Push-Ups",
    "BodyWeightSquats": "Bodyweight Squats",
    "Lunges": "Lunges",
    "TaiChi": "Tai Chi",
    "BoxingSpeedBag": "Boxing (Speed Bag)",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Frames are sampled at a fixed temporal rate everywhere (training, video analysis and
# webcam) so that a window of SEQ_LEN frames always spans the same real-world duration.
SAMPLE_FPS = 8.0
SEQ_LEN = 16
IMG_SIZE = 160          # network input resolution (square, center crop)
CACHE_SHORT_SIDE = 160  # short side of frames stored in the preprocessing cache
VAL_GROUPS = (8, 9, 10)  # training-split groups held out for validation (no group overlap)


@dataclass
class TrainConfig:
    classes: list[str] = field(default_factory=lambda: list(CLASSES))
    split: int = 1                 # official UCF101 split (1, 2 or 3)
    seq_len: int = SEQ_LEN         # frames per prediction window
    sample_fps: float = SAMPLE_FPS
    img_size: int = IMG_SIZE
    clips_per_video: int = 3       # random clips drawn from each training video per epoch
    batch_size: int = 16
    epochs: int = 20
    lr_head: float = 3e-4
    lr_backbone: float = 3e-5
    weight_decay: float = 1e-2
    lstm_hidden: int = 256
    lstm_layers: int = 1
    dropout: float = 0.4
    unfreeze_blocks: int = 0       # trailing MobileNetV3 blocks fine-tuned (0 = frozen; selected on validation)
    use_deltas: bool = True        # LSTM input = [feature ; temporal feature difference]
    num_workers: int = 2
    amp: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferenceConfig:
    seq_len: int = SEQ_LEN
    img_size: int = IMG_SIZE
    sample_fps: float = SAMPLE_FPS  # camera frames are sub-sampled to this rate
    infer_every: int = 2            # run the LSTM every N newly sampled frames
    confidence_threshold: float = 0.60
    smoothing_alpha: float = 0.40   # EMA weight of the newest window's probabilities
    motion_gate: bool = True        # report UNKNOWN when there is almost no movement
    motion_threshold: float = 0.8   # mean abs frame diff (0-255). Chosen on validation data: still webcam ~0.4,
                                    # Tai Chi (slowest class) median 1.39, all other classes > 1.8
    video_window_stride: int = 4    # sampled frames between windows in video analysis


CHECKPOINT_PATH = CHECKPOINT_DIR / "motionsense_best.pt"
HISTORY_PATH = REPORTS_DIR / "training_history.json"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"
DISTRIBUTION_PATH = REPORTS_DIR / "class_distribution.json"
MANIFEST_PATH = DATASETS_DIR / "manifest.csv"
ASSETS_DIR = ROOT / "assets"


TEST_DATA_DIR = ROOT / "test_data"


def test_videos() -> list[Path]:
    """Demo/test clips for the UI: the mixed-activity video first, then per-class test-split MP4s,
    falling back to the raw AVI samples."""
    mixed = TEST_DATA_DIR / "mixed_activities.mp4"
    clips = sorted(p for p in TEST_DATA_DIR.glob("*.mp4") if p != mixed)
    vids = ([mixed] if mixed.exists() else []) + clips
    return vids or sorted(SAMPLES_DIR.glob("*.avi"))


def video_label(p: Path) -> str:
    """Human-readable name for a demo/test clip."""
    stem = p.stem
    if stem == "mixed_activities":
        return "All six activities · mixed sequence"
    cls = stem.split("_")[0]
    rest = stem.split("_v_")[-1].split("_")[-2:] if "_v_" in stem else stem.split("_")[1:3]
    return f"{DISPLAY_NAMES.get(cls, cls)} · {' '.join(rest)}"
