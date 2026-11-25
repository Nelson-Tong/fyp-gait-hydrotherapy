#!/usr/bin/env python3
"""
Multi-Output LSTM Regression with Single Sliding Window,
Stratified CV + Held-Out Test Set + Checkpointing & Resume,
with GPU memory growth, configurable pooling mode, and NaN-safe windowing.
Predicting [swing, stance, stride, cadence] simultaneously.

After training, writes:
 - summary metrics, per-fold scores, model architecture summary, and hyperparameters to an Excel workbook with four sheets
 - evaluates RMSE, MAPE, and R^2 on hold-out and includes them in the summary
"""
from utils_scaler import fit_scaler, apply_scaler, save_stats, load_stats

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from tensorflow.keras.optimizers import AdamW

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ── GLOBAL SEED & DETERMINISM ───────────────────────────────
import random, numpy as np
SEED = 42
# tf.config.optimizer.set_jit(True)
os.environ["PYTHONHASHSEED"]       = str(SEED)   # A
os.environ["TF_DETERMINISTIC_OPS"] = "1"         # B

random.seed(SEED)                                # C
np.random.seed(SEED)                             # D
tf.keras.utils.set_random_seed(SEED)             # E
tf.config.experimental.enable_op_determinism()   # F  (works in TF-2.10)
# ------------------------------------------------------------

import io, json, gc, pathlib
from pathlib import Path
import time
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score
from tensorflow.keras import layers, Model, callbacks, optimizers

# ─── EXTRA METRICS ────────────────────────────────────────────────────────────
from scipy.stats import pearsonr, combine_pvalues
import pingouin as pg        #  pip install pingouin  (tiny dependency)

def pearson_r(y_true, y_pred):
    """
    Returns:
        r_mean  – mean Pearson r across all targets (same definition you used)
        p_comb  – global two-tailed p-value for the null ‘no correlation’.
                  Individual target p-values are combined with Fisher’s method.
    """
    r_vals, p_vals = [], []
    for i in range(y_true.shape[1]):
        r, p = pearsonr(y_true[:, i], y_pred[:, i])
        r_vals.append(r)
        p_vals.append(p)

    # Fisher method → one overall p (more informative than just max/mean p)
    _stat, p_comb = combine_pvalues(p_vals, method="fisher")
    return float(np.mean(r_vals)), float(p_comb)

def icc2_1(y_true, y_pred):
    """Two-way random effects, single measurement ICC(2,1)."""
    n = y_true.shape[0]
    stacked = np.vstack([y_true.flatten(), y_pred.flatten()]).T
    df = (
        pd.DataFrame(stacked, columns=["rater1", "rater2"])
          .assign(target=np.repeat(np.arange(n), repeats=1))
          .melt(id_vars="target", var_name="rater", value_name="score")
    )
    icc = pg.intraclass_corr(data=df,
                             targets="target",
                             raters="rater",
                             ratings="score").query("Type=='ICC2' & Unit=='single'")["ICC"]
    return float(icc.iloc[0])

def bland_altman(y_true, y_pred):
    """Return mean diff and ±1.96 SD limits."""
    diff = (y_pred - y_true).flatten()
    mean_diff = diff.mean()
    sd = diff.std(ddof=1)
    return mean_diff, mean_diff - 1.96*sd, mean_diff + 1.96*sd
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# 1) USER PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
LOSS_FN = tf.keras.losses.MeanSquaredError()

# "swing", "stance", "stride", "cadence"
TARGETS             = ["swing"]
EPOCHS              = 100
RUN_CV              = True
VAL_FRAC            = 0.1      # 0.1 or 0.2
POOLING_MODE        = "staggered"  # "staggered" (every 2nd conv or final conv) or "every_conv" (after every conv)

# "zscore" or "max"
NORM_METHOD         = "max"
FEATURE_NORM_METHOD = "max"    # or "zscore"
COUPLED_GROUPS      = None     # None ⇒ per-feature scaling, example if got: [[0,1,2],[3,4,5]]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CONFIG_DIR   = PROJECT_ROOT / "configs"

FEATURE_DIR  = DATA_DIR / "YOLOv11m_with_linear_interpolation_and_upper_limbs_removed"
LABELS_CSV   = DATA_DIR / "Labels.csv"

SAVE_PATH    = PROJECT_ROOT / "results" / "cross_validation_run"

hp_path      = CONFIG_DIR / "best_hp.json"
with open(hp_path, encoding="utf-8") as f:
    BEST_HP = json.load(f)
print(BEST_HP)


TEST_FRAC           = 0.10
N_FOLDS             = 10
N_FEATS             = 12
PATIENCE            = 10       # In HPO, patience and reduce_on_plateau were set to 5, means no reduction of learning rate (to save time, because will be stopped earlier)
                               # Changing to 10 to allow this, perhaps better result will be obtained.

WINDOW_SIZE         = BEST_HP["window_size"]
STRIDE              = BEST_HP["stride"]
BATCH_SIZE          = BEST_HP["batch_size"]
NUM_CNN_STACKES     = BEST_HP["num_cnn_stacks"]
CNN_FILTERS         = BEST_HP["cnn_filters"]
CNN_KERNELS         = BEST_HP["cnn_kernel"]
POOL_SIZE           = BEST_HP["pool_size"]
USE_BATCHNORM       = BEST_HP["use_batchnorm"]
CNN_DROPOUT         = BEST_HP["cnn_dropout"]
NUM_LSTM_LAYERS     = BEST_HP["lstm_layers"]
LSTM_UNITS          = BEST_HP["lstm_units"]
DROPOUT_RATE        = BEST_HP["dropout"]
DENSE_UNIT          = BEST_HP["dense_units"]
LEARNING_RATE       = BEST_HP["learning_rate"]
WEIGHT_DECAY        = BEST_HP["weight_decay"]

PADDING             = "same"
ONLY_ONCE_FINAL_DROPOUT = True

# ensure output dirs
os.makedirs(os.path.join(SAVE_PATH, "checkpoints"),         exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "csv_logs"),            exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) READ & NORMALIZE LABELS
# ──────────────────────────────────────────────────────────────────────────────
df_labels = pd.read_csv(LABELS_CSV)

NORM_STATS = {}
df_labels = df_labels[["filename", "type", *TARGETS]]

for t in TARGETS:
    col = df_labels[t].astype(np.float32)

    if NORM_METHOD == "max":
        mx = col.abs().max() or 1.0
        df_labels[t] = col / mx
        NORM_STATS[t] = {"max": float(mx)}

    elif NORM_METHOD == "zscore":
        mu  = col.mean()
        sig = col.std(ddof=0) or 1.0
        df_labels[t] = (col - mu) / sig
        NORM_STATS[t] = {"mean": float(mu), "std": float(sig)}

    else:
        raise ValueError("Unknown NORM_METHOD")

stats_path = pathlib.Path(SAVE_PATH, f"norm_stats_{NORM_METHOD}.json")
stats_path.write_text(json.dumps(NORM_STATS, indent=2))
print(f"✓ wrote normalisation stats → {stats_path}")
df_labels["stratify_key"] = df_labels.get("type", 0)

# ──────────────────────────────────────────────────────────────────────────────
# 3) HOLD-OUT SPLIT
# ──────────────────────────────────────────────────────────────────────────────
trainval_df, test_df = train_test_split(
    df_labels,
    test_size=TEST_FRAC,
    stratify=df_labels["stratify_key"],
    random_state=SEED
)

# Save reserved test set filenames for inspection
test_filenames = test_df["filename"].tolist()
with open(os.path.join(SAVE_PATH, "test_set_filenames.txt"), "w") as f:
    for fn in test_filenames:
        f.write(f"{fn}\n")
print(f"✓ Test set filenames saved to {os.path.join(SAVE_PATH, 'test_set_filenames.txt')}")

# ──────────────────────────────────────────────────────────────────────────────
# 4) SLIDING WINDOW & WINDOW MAKING
# ──────────────────────────────────────────────────────────────────────────────
def sliding_window(seq, window_size, stride):
    T, F = seq.shape
    if T < window_size:
        seq = np.vstack([seq, np.repeat(seq[-1:], window_size - T, axis=0)])
        T = window_size
    return np.stack([seq[i:i + window_size] for i in range(0, T - window_size + 1, stride)])

def make_windows_and_labels(df, feature_dir):
    X, y = [], []
    for _, r in df.iterrows():
        path = os.path.join(feature_dir, f"{r.filename}.csv")
        arr = pd.read_csv(path).iloc[:, 1:13].values.astype(np.float32)
        wds = sliding_window(arr, WINDOW_SIZE, STRIDE)
        for w in wds:
            if not np.isnan(w).any():
                X.append(w)
                y.append([r[t] for t in TARGETS])
    if not X:
        raise ValueError(f"No valid windows from {len(df)} samples!")
    return np.stack(X), np.array(y, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# 5) MODEL BUILDING FUNCTION (no SE)
# ──────────────────────────────────────────────────────────────────────────────
def build_model(window_size, n_feats,
                cnn_filters=CNN_FILTERS,
                cnn_kernel=CNN_KERNELS,
                pool_size=POOL_SIZE):
    if POOLING_MODE not in {"staggered", "every_conv"}:
        raise ValueError(f"Unknown POOLING_MODE: {POOLING_MODE}")

    inp = layers.Input((window_size, n_feats))
    x = inp

    for i in range(NUM_CNN_STACKES):
        x = layers.Conv1D(
            filters=cnn_filters,
            kernel_size=cnn_kernel,
            padding=PADDING,
            activation="relu",
            kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED),
        )(x)

        should_pool = (
            ((i % 2 == 1) or (i == NUM_CNN_STACKES - 1))
            if POOLING_MODE == "staggered"
            else True
        )
        if should_pool:
            x = layers.MaxPooling1D(pool_size=pool_size)(x)

    if USE_BATCHNORM:
        x = layers.BatchNormalization()(x)
    if CNN_DROPOUT > 0:
        x = layers.Dropout(CNN_DROPOUT, seed=SEED)(x)

    # --- 2) LSTM stack (Bi-LSTM) ---
    for i in range(NUM_LSTM_LAYERS):
        return_seq = (i < NUM_LSTM_LAYERS - 1)
        x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=return_seq,
                                             kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED),
                                             recurrent_initializer=tf.keras.initializers.Orthogonal(seed=SEED)))(x)
        if not ONLY_ONCE_FINAL_DROPOUT:
            x = layers.Dropout(DROPOUT_RATE, seed=SEED)(x)

    # --- 3) final dropout if requested ---
    if ONLY_ONCE_FINAL_DROPOUT:
        x = layers.Dropout(DROPOUT_RATE, seed=SEED)(x)

    # --- 4) dense & output heads ---
    x = layers.Dense(DENSE_UNIT, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal(seed=SEED))(x)
    out = layers.Dense(len(TARGETS), activation="linear")(x)

    opt = AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, clipnorm=1.0)

    model = Model(inp, out)
    model.compile(
        optimizer=opt,
        loss=LOSS_FN,
        metrics=[
            "mae",
            "mse",
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape")
        ]
    )
    return model

# ──────────────────────────────────────────────────────────────────────────────
# 6) CROSS-VALIDATION  *or* SINGLE SPLIT TRAINING
# ──────────────────────────────────────────────────────────────────────────────
PROGRESS = Path(SAVE_PATH) / "fold_progress.json"
done = set(json.loads(PROGRESS.read_text())) if PROGRESS.exists() else set()

PROGRESS.write_text(json.dumps(sorted(done)))
print("Autodetected completed folds:", sorted(done))

cv_scores    = []
script_start = time.time()

if RUN_CV:
    skf = StratifiedKFold(n_splits=N_FOLDS,
                          shuffle=True,
                          random_state=SEED)

    for fold, (tr, val) in enumerate(
            skf.split(trainval_df, trainval_df["stratify_key"]), 1):
        if fold in done:
            print(f"✓ fold {fold} already done – skipping")
            continue

        print(f"\n=== Fold {fold}/{N_FOLDS} ===")
        df_tr, df_val = trainval_df.iloc[tr], trainval_df.iloc[val]
        X_tr, y_tr    = make_windows_and_labels(df_tr,  FEATURE_DIR)
        X_val, y_val  = make_windows_and_labels(df_val, FEATURE_DIR)

        scaler_stats = fit_scaler(X_tr, method=FEATURE_NORM_METHOD,
                                  groups=COUPLED_GROUPS)
        save_stats(scaler_stats,
                   os.path.join(SAVE_PATH, f"norm_stats_fold{fold}_{FEATURE_NORM_METHOD}.json"))

        X_tr  = apply_scaler(X_tr,  scaler_stats, FEATURE_NORM_METHOD)
        X_val = apply_scaler(X_val, scaler_stats, FEATURE_NORM_METHOD)

        ckpt = os.path.join(SAVE_PATH, "checkpoints", f"fold{fold}.weights.h5")
        logf = os.path.join(SAVE_PATH, "csv_logs",    f"fold{fold}_log.csv")

        model = build_model(WINDOW_SIZE, n_feats=N_FEATS)
        start = 0
        if Path(ckpt).exists() and Path(logf).exists():
            model.load_weights(ckpt)
            start = pd.read_csv(logf).shape[0]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping("val_rmse",
                                        patience=PATIENCE,
                                        restore_best_weights=True),
                callbacks.ReduceLROnPlateau("val_rmse", factor=0.5, patience=5),
                callbacks.CSVLogger(logf, append=True),
                callbacks.ModelCheckpoint(ckpt, save_best_only=True,
                                          save_weights_only=True,
                                          monitor="val_rmse"),
                callbacks.TerminateOnNaN()
            ],
            verbose=2,
            initial_epoch=start
        )

        loss, mae, mse, rmse, mape = model.evaluate(X_val, y_val, verbose=0)
        y_val_pred = model.predict(X_val, verbose=0)

        r_val, p_val = pearson_r(y_val, y_val_pred)
        icc_val      = icc2_1(y_val, y_val_pred)
        ba_mean, ba_lo, ba_hi = bland_altman(y_val, y_val_pred)

        print(f"Fold {fold} →  MSE {mse:.4f}  RMSE {rmse:.4f}  Pearson r {r_val:.3f}  "
              f"ICC2,1 {icc_val:.3f}")

        cv_scores.append(
            (fold, mse, mae, rmse, mape, r_val, p_val, icc_val, ba_mean, ba_lo, ba_hi)
        )

        done.add(fold)
        PROGRESS.write_text(json.dumps(sorted(done)))
        tf.keras.backend.clear_session()
        gc.collect()

else:
    print("RUN_CV is False ➜ skipping intermediate train/val. "
          "Going straight to final 90 % train / 10 % hold-out.")

# ──────────────────────────────────────────────────────────────────────────────
# 7) FINAL TRAIN & EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Final train on 90% → test on 10% hold-out ===")
X_tv, y_tv = make_windows_and_labels(trainval_df, FEATURE_DIR)

scaler_stats_tv = fit_scaler(X_tv, method=FEATURE_NORM_METHOD,
                             groups=COUPLED_GROUPS)
save_stats(scaler_stats_tv,
           os.path.join(SAVE_PATH, f"norm_stats_trainval_{FEATURE_NORM_METHOD}.json"))

X_tv = apply_scaler(X_tv, scaler_stats_tv, FEATURE_NORM_METHOD)

final_model = build_model(WINDOW_SIZE, n_feats=N_FEATS)

# capture architecture summary
buf = io.StringIO()
final_model.summary(print_fn=lambda line: buf.write(line + "\n"))
sum_lines = buf.getvalue().splitlines()
buf.close()

ckpt = os.path.join(SAVE_PATH, "checkpoints", "final_model.keras")
logf = os.path.join(SAVE_PATH, "csv_logs",    "final_log.csv")

# train on trainval
final_model.fit(
    X_tv, y_tv,
    validation_split=VAL_FRAC,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        callbacks.EarlyStopping("val_rmse", patience=PATIENCE, restore_best_weights=True),
        callbacks.ReduceLROnPlateau("val_rmse", factor=0.5, patience=5),
        callbacks.CSVLogger(logf),
        callbacks.ModelCheckpoint(ckpt, save_best_only=True, save_weights_only=False, monitor="val_rmse"),
        callbacks.TerminateOnNaN()
    ],
    verbose=2
)

# evaluate on hold-out
X_test, y_test = make_windows_and_labels(test_df, FEATURE_DIR)
X_test = apply_scaler(X_test, scaler_stats_tv, FEATURE_NORM_METHOD)
results = final_model.evaluate(X_test, y_test, verbose=0)
_, test_mae, test_mse, test_rmse, test_mape = results

y_pred = final_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_r, test_r_p = pearson_r(y_test, y_pred)
test_icc = icc2_1(y_test, y_pred)
ba_mean_t, ba_lo_t, ba_hi_t = bland_altman(y_test, y_pred)

print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}, R2: {test_r2:.4f}")
script_end = time.time()
total_time = script_end - script_start

# ──────────────────────────────────────────────────────────────────────────────
# 8) AGGREGATE METRICS & SAVE
# ──────────────────────────────────────────────────────────────────────────────
mse_vals  = [s[1] for s in cv_scores] or [np.nan]
mae_vals  = [s[2] for s in cv_scores] or [np.nan]
rmse_vals = [s[3] for s in cv_scores] or [np.nan]
mape_vals = [s[4] for s in cv_scores] or [np.nan]
r_vals    = [s[5] for s in cv_scores] or [np.nan]
p_vals    = [s[6] for s in cv_scores] or [np.nan]
icc_vals  = [s[7] for s in cv_scores] or [np.nan]
ba_means  = [s[8] for s in cv_scores] or [np.nan]
ba_lo     = [s[9] for s in cv_scores] or [np.nan]
ba_hi     = [s[10] for s in cv_scores] or [np.nan]

summary = {
    "pooling_mode":  POOLING_MODE,
    "mean_cv_mse":   np.mean(mse_vals),  "std_cv_mse":    np.std(mse_vals),
    "mean_cv_mae":   np.mean(mae_vals),  "std_cv_mae":    np.std(mae_vals),
    "mean_cv_rmse":  np.mean(rmse_vals), "std_cv_rmse":   np.std(rmse_vals),
    "mean_cv_mape":  np.mean(mape_vals), "std_cv_mape":   np.std(mape_vals),
    "mean_cv_r":     np.mean(r_vals),    "std_cv_r":      np.std(r_vals),
    "mean_cv_icc":   np.mean(icc_vals),  "std_cv_icc":    np.std(icc_vals),
    "mean_cv_ba":    np.mean(ba_means),  "std_cv_ba":     np.std(ba_means),
    "test_mse":      test_mse,
    "test_mae":      test_mae,
    "test_rmse":     test_rmse,
    "test_mape":     test_mape,
    "test_r2":       test_r2,
    "test_r":        test_r,             "test_r_p":      test_r_p,
    "test_icc":      test_icc,
    "test_ba_mean":  ba_mean_t,
    "test_ba_lo":    ba_lo_t,
    "test_ba_hi":    ba_hi_t,
    "total_time_s":  total_time
}

folds_df = pd.DataFrame(
    cv_scores,
    columns=["fold", "val_mse", "val_mae", "val_rmse", "val_mape",
             "val_r",  "val_r_p", "val_icc2_1", "ba_mean", "ba_lo", "ba_hi"]
).sort_values("fold").reset_index(drop=True)

summary_df = pd.DataFrame([summary])
model_df   = pd.DataFrame({"architecture_summary": sum_lines})
hyperparams = {
    "pooling_mode": POOLING_MODE,
    "val_frac": VAL_FRAC,
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "test_frac": TEST_FRAC,
    "n_folds": N_FOLDS,
    "random_state": SEED,
    "batch_size": BATCH_SIZE,
    "n_feats": N_FEATS,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "patience": PATIENCE,
    "lstm_layers": NUM_LSTM_LAYERS,
    "lstm_units": LSTM_UNITS,
    "dropout_rate": DROPOUT_RATE,
    "dense_unit": DENSE_UNIT,
    "only_final_dropout": ONLY_ONCE_FINAL_DROPOUT,
    "CNN_stackes": NUM_CNN_STACKES,
    "CNN_filters": CNN_FILTERS,
    "CNN_kernels": CNN_KERNELS,
    "pool_size": POOL_SIZE,
    "padding": PADDING,
    "use_batchnorm": USE_BATCHNORM,
    "cnn_dropout": CNN_DROPOUT,
    "weight_decay": WEIGHT_DECAY,
}
hyper_df = pd.DataFrame(list(hyperparams.items()), columns=["hyperparameter", "value"])

with pd.ExcelWriter(os.path.join(SAVE_PATH, "training_summary.xlsx")) as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    folds_df.to_excel(writer, sheet_name="per_fold", index=False)
    model_df.to_excel(writer, sheet_name="model_architecture", index=False)
    hyper_df.to_excel(writer, sheet_name="hyperparameters", index=False)

print("Saved results and hyperparameters (including RMSE, MAPE, R2) to training_summary.xlsx")
