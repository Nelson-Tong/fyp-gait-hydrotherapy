#!/usr/bin/env python3
"""
Batch gait-parameter inference + evaluation + Bland-Altman plots
────────────────────────────────────────────────────────────────
• Predicts every video / CSV in a folder (single- or multi-target nets)
• Joins predictions with Labels.csv
• Writes Excel workbook with:
    – all_predictions, test_set, trainval_set
    – metrics_all, metrics_test, metrics_trainval
• Saves Bland-Altman plots (PNG) and annotated videos (MP4/AVI)
"""

import collections
import glob
import json
import os
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend for saving PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pingouin as pg
from scipy.stats import pearsonr
import tensorflow as tf
from ultralytics import YOLO

from utils_scaler import apply_scaler, load_stats


# ═════════ 0) CONFIG (relative paths for GitHub repository) ═════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR   = PROJECT_ROOT / "models" / "singlemode" / "04_MultiTPE_Hyperband_multicnn_per_pool"

# Options of MODEL_DIR:
# Single head models: 01_GP_singlecnn_per_pool, 02_MultiTPE_Hyperband_singlecnn_per_pool, 03_UniTPE_Hyperband_singlecnn_per_pool, 04_MultiTPE_Hyperband_multicnn_per_pool
# Multihead models are in PROJECT_ROOT / "models" / "multimode" / <target> (swing, stance, stride, cadence)

CONFIG = dict(
    # inference options ──────────────────────────────────────────
    # SOURCE accepts a directory containing either video clips (.mp4, .mov, .avi, .mkv)
    # or pre-extracted keypoint .csv files.
    # NOTE: If using CSV inputs, ensure they are already cleaned to retain only the
    # 12 selected lower-limb keypoints (columns 1..12, after the leading frame/index column).
    # Example CSV path: PROJECT_ROOT / "data" / "YOLOv11m_with_linear_interpolation_and_upper_limbs_removed"

    SOURCE              = str(PROJECT_ROOT / "Resized Video Clips (960x540)"),
    POSE_MODEL          = str(PROJECT_ROOT / "models" / "yolo11m-pose.pt"),

    MODEL_MODE          = "singlehead",  # "singlehead" | "multihead"
    DL_MODEL            = str(MODEL_DIR / "best_model.keras"),
    DL_MODELS           = {
        "swing"  : str(PROJECT_ROOT / "models" / "multimode" / "swing"),
        "stance" : str(PROJECT_ROOT / "models" / "multimode" / "stance"),
        "stride" : str(PROJECT_ROOT / "models" / "multimode" / "stride"),
        "cadence": str(PROJECT_ROOT / "models" / "multimode" / "cadence"),
    },

    FEATURE_NORM_METHOD = "max",
    FEAT_STATS_FILE     = str(PROJECT_ROOT / "models" / "norm_stats_trainval_max.json"),
    LABEL_CSV           = str(PROJECT_ROOT / "data" / "Labels.csv"),
    FRAME_RATE          = 30,
    TARGETS             = ["swing", "stance", "stride", "cadence"],

    # outputs ───────────────────────────────────────────────────
    CANVAS_HEIGHT       = 80,
    DRAW_SKELETON       = True,  # set False for text-only annotation
    SAVE_CSV            = True,
    CSV_PATH            = str(PROJECT_ROOT / "results" / "predictions" / "gait_predictions.csv"),
    SAVE_EXCEL          = True,
    EXCEL_PATH          = str(PROJECT_ROOT / "results" / "predictions" / "gait_evaluation.xlsx"),
    SAVE_BA_PLOTS       = True,
    BA_PLOT_DIR         = str(PROJECT_ROOT / "results" / "predictions" / "ba_plots"),
    SAVE_VIDEO          = True,
    VIDEO_DIR           = str(PROJECT_ROOT / "results" / "annotated_videos"),

    # optional test/train split ─────────────────────────────────
    TESTSET_FILE        = str(PROJECT_ROOT / "data" / "test_set_filenames.txt"),
)

# ═════════ 1) STATIC HELPERS ═════════
FONT_PATH            = str(PROJECT_ROOT / "assets" / "NotoSansJP-VariableFont_wght.ttf")
FONT_SIZE            = 24
FONT_COLOR           = (0, 255, 0, 255)

REFRESH_EVERY_FRAMES = 15  # redraw rate
SMOOTH_N_PREDICTS    = 15  # averaging window
LOWER_KP             = [11, 12, 13, 14, 15, 16]

LABELS = {
    "swing"  : "Swing time:",
    "stance" : "Stance time:",
    "stride" : "Stride time:",
    "cadence": "Cadence:",
}

UNITS = {
    "swing"  : "s",
    "stance" : "s",
    "stride" : "s",
    "cadence": "spm",
}


# ═════════ 2) FEATURE & LABEL STATS ═════════
FEAT_STATS = {int(k): v for k, v in load_stats(CONFIG["FEAT_STATS_FILE"]).items()}

LBL = {}
if CONFIG["MODEL_MODE"] == "singlehead":
    with open(os.path.join(MODEL_DIR, "norm_stats_max.json"), encoding="utf-8") as f:
        full = json.load(f)
    LBL = {t: full[t] for t in CONFIG["TARGETS"]}
else:
    for t in CONFIG["TARGETS"]:
        p = os.path.join(CONFIG["DL_MODELS"][t], "norm_stats_max.json")
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        LBL[t] = s if ("max" in s or "std" in s) else s[t]


def denorm_vals(raw, t):
    return np.asarray(raw, np.float32) * LBL[t]["max"]


# ═════════ 3) LOAD NETWORKS ═════════
pose_net = YOLO(CONFIG["POSE_MODEL"])

TRAIN_ORD = ["swing", "stance", "stride", "cadence"]
hp_cfg = {}
if CONFIG["MODEL_MODE"] == "singlehead":
    net = tf.keras.models.load_model(CONFIG["DL_MODEL"], compile=False)
    IDX_SEL = [TRAIN_ORD.index(t) for t in CONFIG["TARGETS"]]
    with open(os.path.join(MODEL_DIR, "best_hp.json"), encoding="utf-8") as f:
        g_hp = json.load(f)
    hp_cfg = {t: g_hp for t in CONFIG["TARGETS"]}
else:
    nets = {}
    for t, folder in CONFIG["DL_MODELS"].items():
        if t in CONFIG["TARGETS"]:
            nets[t] = tf.keras.models.load_model(
                os.path.join(folder, "best_model.keras"), compile=False
            )
            with open(os.path.join(folder, "best_hp.json"), encoding="utf-8") as f:
                hp_cfg[t] = json.load(f)

MAX_WIN = max(hp_cfg[t]["window_size"] for t in CONFIG["TARGETS"])
print(f"✓ Models loaded (max window = {MAX_WIN} frames)")


# ═════════ 4) SIGNAL & OVERLAY HELPERS ═════════
def sliding_window_safe(a, w, s):
    if len(a) < w:
        a = np.vstack([a, np.repeat(a[-1:], w - len(a), 0)])
    return np.stack([a[i : i + w] for i in range(0, len(a) - w + 1, s)])


def extract_best_lower_kp(res):
    if res.keypoints is None:
        return np.full(12, np.nan)
    xy = res.keypoints.xy.cpu().numpy()
    if xy.shape[1] < 17:
        return np.full(12, np.nan)
    conf = (
        res.boxes.conf.cpu().numpy().ravel()
        if hasattr(res, "boxes") and res.boxes.conf is not None
        else res.keypoints.conf.cpu().numpy().mean(1)
    )
    return xy[conf.argmax(), LOWER_KP].reshape(-1)


def _get_font():
    if not hasattr(_get_font, "_font"):
        if os.path.exists(FONT_PATH):
            _get_font._font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        else:
            _get_font._font = ImageFont.load_default()
    return _get_font._font


def draw_overlay(img, text, pos=(10, 25)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=_get_font(), fill=FONT_COLOR)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame, kp12):
    if not CONFIG["DRAW_SKELETON"] or np.isnan(kp12).any():
        return
    pts = kp12.reshape(6, 2).astype(int)
    for a, b in [(0, 2), (2, 4), (1, 3), (3, 5)]:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, tuple(p), 4, (0, 255, 0), -1)


def _format(vals):
    return "  ".join(
        f"{LABELS[t]}{vals[t]:5.2f}{UNITS[t]}"
        if not np.isnan(vals[t])
        else f"{LABELS[t]}…"
        for t in CONFIG["TARGETS"]
    )


# ═════════ 5) METRICS & BLAND-ALTMAN PLOTS ═════════
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else np.nan


def icc2_1(y_true, y_pred):
    """Two-way random effects, single measurement ICC(2,1)."""
    n = len(y_true)
    if n < 2:
        return np.nan
    stacked = np.vstack([y_true, y_pred]).T
    df_icc = (
        pd.DataFrame(stacked, columns=["rater1", "rater2"])
        .assign(target=np.arange(n))
        .melt(id_vars="target", var_name="rater", value_name="score")
    )
    icc = pg.intraclass_corr(
        data=df_icc,
        targets="target",
        raters="rater",
        ratings="score",
    ).query("Type=='ICC2' & Unit=='single'")["ICC"]
    return float(icc.iloc[0])


def bland_altman(y_true, y_pred):
    diff = y_pred - y_true
    bias = float(diff.mean())
    loa = float(1.96 * diff.std(ddof=1)) if len(diff) > 1 else np.nan
    return bias, loa


def compute_metrics(df_sub, targets):
    """Compute MAE, RMSE, MAPE, R2, Pearson r, ICC(2,1), and Bland-Altman limits per target."""
    rows = []
    for t in targets:
        pred_col = f"{t}_pred"
        if pred_col not in df_sub.columns or t not in df_sub.columns:
            continue
        valid = df_sub[[t, pred_col]].dropna()
        n = len(valid)
        if n == 0:
            continue

        yt = valid[t].to_numpy(dtype=np.float64)
        yp = valid[pred_col].to_numpy(dtype=np.float64)
        err = yp - yt

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        nonzero = yt != 0
        mape = float(np.mean(np.abs(err[nonzero] / yt[nonzero])) * 100.0) if np.any(nonzero) else np.nan
        r2 = r2_score(yt, yp) if n > 1 else np.nan
        r_val, p_val = pearsonr(yt, yp) if n > 1 else (np.nan, np.nan)
        icc_val = icc2_1(yt, yp)
        bias, loa = bland_altman(yt, yp)

        rows.append(
            {
                "target": t,
                "n": n,
                "mae": mae,
                "rmse": rmse,
                "mape_pct": mape,
                "r2": r2,
                "pearson_r": float(r_val),
                "pearson_p": float(p_val),
                "icc2_1": icc_val,
                "ba_bias": bias,
                "ba_loa_lower": bias - loa if not np.isnan(loa) else np.nan,
                "ba_loa_upper": bias + loa if not np.isnan(loa) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_bland_altman_plots(df_sub, targets, out_dir, subset_name="all"):
    """Save Bland-Altman agreement PNG plots for each target."""
    os.makedirs(out_dir, exist_ok=True)
    for t in targets:
        pred_col = f"{t}_pred"
        if pred_col not in df_sub.columns or t not in df_sub.columns:
            continue
        valid = df_sub[[t, pred_col]].dropna()
        if len(valid) < 2:
            continue

        yt = valid[t].to_numpy(dtype=np.float64)
        yp = valid[pred_col].to_numpy(dtype=np.float64)
        mean_vals = 0.5 * (yt + yp)
        diffs = yp - yt
        bias, loa = bland_altman(yt, yp)

        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
        ax.scatter(mean_vals, diffs, alpha=0.75, edgecolors="k", linewidths=0.5)
        ax.axhline(bias, color="tab:blue", linestyle="-", linewidth=1.5, label=f"Bias: {bias:.4f}")
        ax.axhline(bias + loa, color="tab:red", linestyle="--", linewidth=1.2, label=f"+1.96 SD: {bias + loa:.4f}")
        ax.axhline(bias - loa, color="tab:red", linestyle="--", linewidth=1.2, label=f"-1.96 SD: {bias - loa:.4f}")
        ax.set_title(f"Bland–Altman: {t.capitalize()} ({subset_name})")
        ax.set_xlabel(f"Mean of Ground Truth & Predicted ({UNITS[t]})")
        ax.set_ylabel(f"Difference: Pred − GT ({UNITS[t]})")
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()

        plot_path = os.path.join(out_dir, f"ba_{subset_name}_{t}.png")
        fig.savefig(plot_path)
        plt.close(fig)


# ═════════ 6) PREDICTION HELPERS ═════════
def _predict_windows(arr):
    out = {}
    for t in CONFIG["TARGETS"]:
        ws, st = hp_cfg[t]["window_size"], hp_cfg[t]["stride"]
        wds = sliding_window_safe(arr, ws, st)

        wds = (
            pd.DataFrame(wds.reshape(-1, 12))
            .interpolate("linear", axis=0, limit_direction="both")
            .values.reshape(wds.shape)
        )
        valid_wds = [w for w in wds if not np.isnan(w).any()]
        if not valid_wds:
            continue
        wds = np.stack(valid_wds)

        wds = apply_scaler(wds, FEAT_STATS, CONFIG["FEATURE_NORM_METHOD"])
        raw = (
            net.predict(wds, verbose=0)[:, IDX_SEL[CONFIG["TARGETS"].index(t)]]
            if CONFIG["MODEL_MODE"] == "singlehead"
            else nets[t].predict(wds, verbose=0).ravel()
        )
        out[t] = float(denorm_vals(raw, t).mean())
    return out if len(out) == len(CONFIG["TARGETS"]) else None


def _init_writer(fname, w, h):
    """
    Tries MP4/mp4v first, falls back to AVI/XVID if the codec is unavailable.
    Returns (writer, final_path).
    """
    out_mp4 = os.path.join(CONFIG["VIDEO_DIR"], fname + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, CONFIG["FRAME_RATE"], (w, h))

    if vw.isOpened():
        return vw, out_mp4

    print(f"[WARN] mp4v codec failed; falling back to *.avi/XVID → {fname}.avi")
    out_avi = os.path.join(CONFIG["VIDEO_DIR"], fname + ".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(out_avi, fourcc, CONFIG["FRAME_RATE"], (w, h))

    if not vw.isOpened():
        raise RuntimeError(
            "Both mp4v and XVID writers failed – install an ffmpeg-enabled OpenCV."
        )
    return vw, out_avi


def predict_video(path):
    """
    Read → predict → write annotated video with a 2-s summary trailer.
    """
    recent = {t: collections.deque(maxlen=SMOOTH_N_PREDICTS) for t in CONFIG["TARGETS"]}
    cap = cv2.VideoCapture(path)
    buf = collections.deque(maxlen=MAX_WIN)
    rows = []
    frame_idx = 0
    running_sum = {t: 0.0 for t in CONFIG["TARGETS"]}
    running_cnt = 0
    display_txt = "…"

    vw = None
    out_path = None
    full_w = full_h = None
    full_frame = None
    fname = os.path.splitext(os.path.basename(path))[0]

    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if fr.ndim == 2:
            fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)

        h, w = fr.shape[:2]
        if vw is None and CONFIG["SAVE_VIDEO"]:
            os.makedirs(CONFIG["VIDEO_DIR"], exist_ok=True)
            full_w, full_h = w, h + CONFIG["CANVAS_HEIGHT"]
            vw, out_path = _init_writer(fname, full_w, full_h)

        # ─── inference ───
        res = pose_net(fr, verbose=False)[0]
        kp_raw = extract_best_lower_kp(res)
        rows.append(kp_raw)
        buf.append(kp_raw)

        pred_map = {}
        for t in CONFIG["TARGETS"]:
            ws, st = hp_cfg[t]["window_size"], hp_cfg[t]["stride"]
            if len(buf) >= ws and frame_idx % st == 0 and not np.isnan(buf[-1]).any():
                seg = np.asarray(list(buf)[-ws:], np.float32)
                seg = (
                    pd.DataFrame(seg)
                    .interpolate("linear", axis=0, limit_direction="both")
                    .values
                )
                seg = apply_scaler(
                    seg.reshape(1, ws, 12),
                    FEAT_STATS,
                    CONFIG["FEATURE_NORM_METHOD"],
                )
                raw = (
                    net.predict(seg, verbose=0)[0][IDX_SEL[CONFIG["TARGETS"].index(t)]]
                    if CONFIG["MODEL_MODE"] == "singlehead"
                    else nets[t].predict(seg, verbose=0)[0][0]
                )
                pred_val = float(denorm_vals(raw, t))
                pred_map[t] = pred_val
                recent[t].append(pred_val)

        if pred_map:
            for t, v in pred_map.items():
                running_sum[t] += v
            running_cnt += 1

        # overlay
        vals = {
            t: (sum(recent[t]) / len(recent[t]) if recent[t] else np.nan)
            for t in CONFIG["TARGETS"]
        }
        if frame_idx % REFRESH_EVERY_FRAMES == 0:
            display_txt = _format(vals)
            for dq in recent.values():
                dq.clear()

        bar = np.zeros((CONFIG["CANVAS_HEIGHT"], w, 3), np.uint8)
        draw_overlay(bar, display_txt, (10, CONFIG["CANVAS_HEIGHT"] // 2))
        draw_skeleton(fr, kp_raw)
        full_frame = np.vstack([fr, bar])

        if CONFIG["SAVE_VIDEO"]:
            vw.write(full_frame)

        frame_idx += 1

    cap.release()

    # summary trailer + always release
    preds = _predict_windows(np.asarray(rows, np.float32)) if rows else None
    if CONFIG["SAVE_VIDEO"] and vw:
        if preds and full_frame is not None:
            summary = {
                t: running_sum[t] / max(running_cnt, 1)
                for t in CONFIG["TARGETS"]
            }
            bar = np.zeros((CONFIG["CANVAS_HEIGHT"], full_w, 3), np.uint8)
            draw_overlay(bar, _format(summary), (10, CONFIG["CANVAS_HEIGHT"] // 2))
            trailer = np.vstack(
                [np.zeros_like(full_frame[: -CONFIG["CANVAS_HEIGHT"]]), bar]
            )
            for _ in range(int(CONFIG["FRAME_RATE"] * 2)):
                vw.write(trailer)
        vw.release()
        print("✓ Annotated video saved →", out_path)

    return dict(video=os.path.basename(path), **(preds or {})) if preds else None


def predict_csv(path):
    arr = pd.read_csv(path).iloc[:, 1:13].values.astype(np.float32)
    preds = _predict_windows(arr)
    return dict(video=os.path.basename(path), **(preds or {})) if preds else None


# ═════════ 7) GROUND-TRUTH & SPLIT HELPERS ═════════
def load_truth_table(path, name_cols=("basename", "filename", "file", "id")):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    name_col = next((c for c in name_cols if c in df.columns), None)
    if name_col is None:
        raise KeyError(f"None of {name_cols} found in {path}")
    df["basename"] = df[name_col].astype(str).str.replace(r"\.\w+$", "", regex=True)
    return df.set_index("basename")


def load_testset_names(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {
            os.path.splitext(line.strip())[0]
            for line in f
            if line.strip()
        }


truth_table = load_truth_table(CONFIG["LABEL_CSV"])


# ═════════ 8) MAIN SWEEP & EXPORT ═════════
def run_folder(folder):
    exts = {".mp4", ".mov", ".avi", ".mkv", ".csv"}
    clips = sorted(
        p
        for p in glob.glob(os.path.join(folder, "*"))
        if os.path.splitext(p)[1].lower() in exts
    )
    print("Files found:")
    for p in clips:
        print("  ", os.path.basename(p))
    print("Total:", len(clips))

    if not clips:
        print("No clips found in", folder)
        return

    rows = []
    for c in clips:
        res = predict_csv(c) if c.lower().endswith(".csv") else predict_video(c)
        if res:
            rows.append(res)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No predictions generated.")
        return

    df["basename"] = df["video"].str.replace(r"\.\w+$", "", regex=True)
    df = df.join(truth_table, on="basename", rsuffix="_gt", how="left")
    for t in CONFIG["TARGETS"]:
        df.rename(columns={t: f"{t}_pred"}, inplace=True)
        gt_col = f"{t}_gt"
        if gt_col in df.columns:
            df.rename(columns={gt_col: t}, inplace=True)

    # Split into test_set vs trainval_set using TESTSET_FILE
    test_names = load_testset_names(CONFIG["TESTSET_FILE"])
    df_test = df[df["basename"].isin(test_names)].copy()
    df_trainval = df[~df["basename"].isin(test_names)].copy()

    # Compute evaluation metrics across all, test, and trainval subsets
    metrics_all = compute_metrics(df, CONFIG["TARGETS"])
    metrics_test = compute_metrics(df_test, CONFIG["TARGETS"])
    metrics_trainval = compute_metrics(df_trainval, CONFIG["TARGETS"])

    if CONFIG["SAVE_CSV"]:
        os.makedirs(os.path.dirname(CONFIG["CSV_PATH"]), exist_ok=True)
        df.to_csv(CONFIG["CSV_PATH"], index=False)
        print(f"✓ Predictions saved to CSV → {CONFIG['CSV_PATH']}")

    if CONFIG["SAVE_EXCEL"]:
        os.makedirs(os.path.dirname(CONFIG["EXCEL_PATH"]), exist_ok=True)
        with pd.ExcelWriter(CONFIG["EXCEL_PATH"]) as writer:
            df.to_excel(writer, sheet_name="all_predictions", index=False)
            df_test.to_excel(writer, sheet_name="test_set", index=False)
            df_trainval.to_excel(writer, sheet_name="trainval_set", index=False)
            metrics_all.to_excel(writer, sheet_name="metrics_all", index=False)
            metrics_test.to_excel(writer, sheet_name="metrics_test", index=False)
            metrics_trainval.to_excel(writer, sheet_name="metrics_trainval", index=False)
        print(f"✓ Evaluation workbook saved → {CONFIG['EXCEL_PATH']}")

    if CONFIG["SAVE_BA_PLOTS"]:
        save_bland_altman_plots(df, CONFIG["TARGETS"], CONFIG["BA_PLOT_DIR"], subset_name="all")
        if not df_test.empty:
            save_bland_altman_plots(df_test, CONFIG["TARGETS"], CONFIG["BA_PLOT_DIR"], subset_name="test")
        if not df_trainval.empty:
            save_bland_altman_plots(df_trainval, CONFIG["TARGETS"], CONFIG["BA_PLOT_DIR"], subset_name="trainval")
        print(f"✓ Bland–Altman plots saved → {CONFIG['BA_PLOT_DIR']}")


# ═════════ 9) ENTRYPOINT ═════════
if __name__ == "__main__":
    src = CONFIG["SOURCE"]
    if not os.path.isdir(src):
        print(f"CONFIG['SOURCE'] must be a directory: {src}")
        sys.exit(1)
    run_folder(src)
