# Dataset & Keypoint Features (`data/`)

This directory holds the lightweight metadata tracked on GitHub (`Labels.csv` and `test_set_filenames.txt`). Heavy media and full keypoint datasets are hosted on **Google Drive** to keep the repository lightweight.

## Required Local Setup (Download from Google Drive)

Before running `src/cross_validation.py` or `src/gait_prediction_pipeline.py` locally, **download the entire Google Drive project folder** to your local machine and place the folders at the repository root (`PROJECT_ROOT`):

PROJECT_ROOT/
├── Assets/                                                         # Overlay font (NotoSansJP-VariableFont_wght.ttf)
├── data/
│   ├── Labels.csv                                                  # Ground-truth gait metrics per clip
│   ├── test_set_filenames.txt                                      # Held-out 10% test split (seed=42)
│   ├── Raw MediaPipe Blazepose/                                    # Unprocessed raw MediaPipe BlazePose coordinates
│   ├── Raw YOLOv11m/                                               # Unprocessed raw YOLOv11m coordinates
│   └── YOLOv11m_with_linear_interpolation_and_upper_limbs_removed/ # Cleaned 12-feature lower-limb CSVs
├── models/                                                         # Pretrained YOLO pose & trained Keras checkpoints
└── Resized Video Clips (960x540)/                                  # Resized 960x540 input video clips

