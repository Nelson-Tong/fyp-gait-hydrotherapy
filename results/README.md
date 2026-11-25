# Results & Reference Archive (`results/`)

This directory is the **local output target** for new runs of `src/cross_validation.py` and `src/gait_prediction_pipeline.py`. Keeping local outputs here ensures your runs stay separate from the completed historical archive on Google Drive (`Results Obtained/`).

## Local Run Outputs

When you execute the scripts locally, outputs are written to dedicated subfolders here:

* **`results/cross_validation_run/`** (from `src/cross_validation.py`):
  * `training_summary.xlsx` (`summary`, `per_fold`, `model_architecture`, and `hyperparameters` sheets)
  * `norm_stats_max.json`, `norm_stats_fold<k>_max.json`, and `norm_stats_trainval_max.json`
  * `fold_progress.json` and `test_set_filenames.txt`
  * `checkpoints/` (`fold1.weights.h5` … `fold10.weights.h5`, `final_model.keras`)
  * `csv_logs/` (`fold1_log.csv` … `fold10_log.csv`, `final_log.csv`)

* **`results/predictions/` & `results/annotated_videos/`** (from `src/gait_prediction_pipeline.py`):
  * `gait_predictions.csv` — clip-level predictions joined with `Labels.csv`
  * `gait_evaluation.xlsx` — six-sheet evaluation report (`all_predictions`, `test_set`, `trainval_set`, `metrics_all`, `metrics_test`, `metrics_trainval` with MAE, RMSE, MAPE, R^2, Pearson r, ICC(2,1), and Bland–Altman limits)
  * `ba_plots/` — Bland–Altman agreement plots (`ba_<subset>_<target>.png`) across `all`, `test`, and `trainval` splits
  * `annotated_videos/` — skeleton-annotated `.mp4`/`.avi` videos with rolling gait-parameter overlays and a 2-second summary trailer

---

## Completed Reference Archive (`Results Obtained/` on Google Drive)

All completed experiment outputs from the study are preserved on **Google Drive** under `Results Obtained/` so you can inspect metrics, Optuna trials, trained weights, and Bland–Altman plots without re-running training:
```text
Results Obtained/
├── HPO optimization/
│   ├── Multi CNN per pool/
│   │   ├── All_Multi_convbilstm_adamw/
│   │   ├── Cadence_Multi_convbilstm_adamw/
│   │   ├── Stance_Multi_convbilstm_adamw/
│   │   ├── Stride_Multi_convbilstm_adamw/
│   │   └── Swing_Multi_convbilstm_adamw/
│   └── Single CNN per pool/
│       ├── MultiTPE vs UniTPE vs GP/
│       │   ├── All_GP_EI_convbilstm/
│       │   ├── All_Multi_Hyperband_convbilstm/
│       │   └── All_Uni_Hyperband_convbilstm/
│       └── MultiTPEs/
│           ├── All_Multi_Hyperband_convbilstm/
│           ├── Cadence_Multi_Hyperband_convbilstm/
│           ├── Stance_Multi_Hyperband_convbilstm/
│           ├── Stride_Multi_Hyperband_convbilstm/
│           └── Swing_Multi_Hyperband_convbilstm/
│       # Each folder contains: <study>.db, best_hp.json, best_model.keras,
│       # norm_stats_max.json, norm_stats_trainval_max.json, optuna_results.csv,
│       # param_importance.json, test_set_filenames.txt, training_summary.xlsx
│
├── Cross Validation/
│   ├── Multi CNN per pool/
│   │   └── Multi CNNs per pool_{all,cadence,stance,stride,swing}/
│   └── Single CNN per pool/
│       └── Single CNNs per pool_{all,cadence,stance,stride,swing}/
│       # Each folder contains: best_hp.json, fold_progress.json,
│       # norm_stats_fold{}_max.json, norm_stats_max.json,
│       # norm_stats_trainval_max.json, test_set_filenames.txt,
│       # training_summary.xlsx, checkpoints/, csv_logs/
│
└── Prediction pipeline/
    ├── (Multi mode) ConvBiLSTM 70 MultiTPEHyper/      # 12 Bland-Altman PNGs + prediction summary
    ├── (Single mode) ConvBiLSTM 70 GP_EI/               # 12 Bland-Altman PNGs + prediction summary
    ├── (Single mode) ConvBiLSTM 70 MultiTPEHyper/      # 12 Bland-Altman PNGs + prediction summary
    ├── (Single mode) ConvBiLSTM 70 UniTPEHyper/      # 12 Bland-Altman PNGs + prediction summary
    ├── Corresponding Raw Resized Video/
    └── Output/                                         # Annotated output videos
```