# Contactless Gait Parameter Estimation with CNN-BiLSTM & Hyperparameter Optimization

End-to-end deep learning pipeline for estimating spatio-temporal gait parameters (**swing time**, **stance time**, **stride time**, and **cadence**) from lower-limb 2D pose keypoints extracted via **YOLOv11m-Pose**.

Note:
Video clips numbered 27, 406 and 423 were excluded due to issues
---

## 1. Setup & Downloading Google Drive Assets

Large datasets, trained model weights, overlay fonts, and completed benchmark archives are stored on Google Drive to keep this GitHub repository lightweight and fast to clone.
https://drive.google.com/drive/folders/1W0xZ12ejdtQfeLQs13VmaxNdZyFhKHPy?usp=sharing

1. **Clone this repository:**
2. **Download the entire Google Drive project folder** into your local repository root (`<REPO_FOLDER>/`) so that `Resized Video Clips (960x540)/`, `Assets/`, `Data/`, `Models/`, and `Results Obtained/` sit directly inside `<REPO_FOLDER>/` (merge `Data/` contents with `data/` if your OS distinguishes uppercase/lowercase folder names).
3. **Install Python dependencies:**
---

## 2. Repository & Local Workspace Directory Structure

<REPO_ROOT>/
├── assets/                                                         # Downloaded from Drive (Assets/)
│   └── NotoSansJP-VariableFont_wght.ttf                            # Overlay font for annotated videos
├── configs/
│   ├── best_hp.json                                                # Tuned hyperparameters passed into CV/inference
│   └── norm_stats_max.json                                         # Label max-normalization reference stats
├── data/                                                           # Merged with Drive (Data/)
│   ├── Labels.csv                                                  # Ground-truth gait labels & stratification metadata
│   ├── test_set_filenames.txt                                      # Reserved 10% hold-out test filenames (seed=42)
│   ├── Raw MediaPipe Blazepose/                                    # Raw MediaPipe BlazePose keypoint CSVs (Drive)
│   ├── Raw YOLOv11m/                                               # Raw YOLOv11m keypoint CSVs (Drive)
│   ├── YOLOv11m_with_linear_interpolation_and_upper_limbs_removed/ # Cleaned 12-feature lower-limb CSVs (Drive)
│   └── README.md
├── models/                                                         # Downloaded from Drive (Models/)
│   ├── yolo11m-pose.pt                                             # YOLOv11m-Pose weights
│   ├── norm_stats_trainval_max.json                                # Feature max-normalization stats
│   ├── singlemode/                                                 # Multi-target (single-head) HPO models
│   │   ├── 01_GP_singlecnn_per_pool/                               # GP + Every-Conv Pooling
│   │   ├── 02_MultiTPE_Hyperband_singlecnn_per_pool/               # Multivariate TPE + Hyperband + Every-Conv Pooling
│   │   ├── 03_UniTPE_Hyperband_singlecnn_per_pool/                 # Univariate TPE + Hyperband + Every-Conv Pooling
│   │   └── 04_MultiTPE_Hyperband_multicnn_per_pool/                # Multivariate TPE + Hyperband + Staggered Pooling
│   └── multimode/                                                  # Target-specific (multi-head) models
│       ├── cadence/
│       ├── stance/
│       ├── stride/
│       └── swing/
├── notebooks/
│   └── HPO_search.ipynb                                            # Self-contained Google Colab Optuna HPO notebook
├── Resized Video Clips (960x540)/                                  # Downloaded from Drive: full video dataset
├── results/                                                        # Local execution outputs (kept separate from archive)
│   └── README.md
├── Results Obtained/                                               # Downloaded from Drive: completed reference archive
│   ├── HPO optimization/
│   ├── Cross Validation/
│   └── Prediction pipeline/
├── src/
│   ├── cross_validation.py                                         # 10-fold stratified CV + final hold-out evaluation
│   ├── gait_prediction_pipeline.py                                 # Batch video/CSV inference, metrics, BA plots & videos
│   └── utils_scaler.py                                             # Shared feature normalization utilities
├── .gitignore
├── requirements.txt
└── README.md


---

## 3. End-to-End Pipeline Workflow

### Stage 1: Hyperparameter Optimization (`notebooks/HPO_search.ipynb`)
* Runs independently on **Google Colab** with Google Drive mounted (`utils_scaler` functions are embedded inline so no local imports are required).
* Optimizes data-windowing, CNN-BiLSTM architectural, and optimizer hyperparameters (`window_size`, `stride`, `batch_size`, CNN stacks/filters/kernels/pooling, Bi-LSTM layers/units, dropout, dense units, learning rate, and weight decay) via **Optuna** (`TPESampler` + `HyperbandPruner` or `GPSampler`).
* Exports `best_hp.json`, `best_model.keras`, `norm_stats_max.json`, `norm_stats_trainval_max.json`, `test_set_filenames.txt`, `optuna_results.csv`, `param_importance.json`, `training_summary.xlsx`, and `<study>.db`.

### Stage 2: Stratified Cross-Validation & Hold-Out Training (`src/cross_validation.py`)
* Executes locally using `data/Labels.csv`, `data/YOLOv11m_with_linear_interpolation_and_upper_limbs_removed/`, and `best_hp.json`.
* Performs a deterministic 90/10 stratified train-val / hold-out split (`SEED=42`), runs **10-fold stratified cross-validation**, and retrains a final model on the full 90% train-val split.
* Computes **MSE, MAE, RMSE, MAPE, R^2, Pearson r, ICC(2,1), and Bland–Altman limits**, writing `training_summary.xlsx`, model checkpoints, and normalization statistics to `results/cross_validation_run/`.

### Stage 3: Batch Inference, Evaluation & Visualization (`src/gait_prediction_pipeline.py`)
* Executes locally in either **`singlehead`** mode (a single multi-output network predicting all 4 gait targets simultaneously) or **`multihead`** mode (four dedicated single-target networks under `models/multimode/`).
* Supports both raw video clips (`Resized Video Clips (960x540)/`) via YOLOv11m-Pose and pre-cleaned 12-keypoint CSVs (`data/YOLOv11m_with_linear_interpolation_and_upper_limbs_removed/`) by switching `CONFIG["SOURCE"]`.
* Exports `gait_predictions.csv`, `gait_evaluation.xlsx` (evaluated across `all`, `test_set`, and `trainval_set`), Bland–Altman PNG plots (`ba_plots/`), and annotated videos (`annotated_videos/`).