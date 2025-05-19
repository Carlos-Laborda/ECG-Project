"""
ppg_preprocessing.py
--------------------
End-to-end preprocessing for Empatica EmbracePlus PPG (BVP) signals:
1. .avro → HDF5 (raw segments)
2. cleaning: 0.5–4 Hz band-pass 
3. z-score: user-specific (per participant)
4. windowing: 10-s windows, 5-s stride  (=> 640 × 1 arrays)
"""

import os, glob, h5py
from datetime import datetime, timezone

import numpy as np
import fastavro
from scipy.signal import butter, filtfilt, iirnotch

# ───────────────────────────────
# helper funct.
# ───────────────────────────────
def list_avro_files(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "P*_empatica", "raw_data", "v6", "*.avro")))

# def butter_bandpass(sig, fs, low=0.5, high=4.0, order=4):
#     nyq = 0.5 * fs
#     b, a = butter(order, [low/nyq, high/nyq], btype="band")
#     return filtfilt(b, a, sig)

# def notch_50hz(sig, fs, q=30):
#     nyq = 0.5 * fs
#     b, a = iirnotch(50/nyq, q)
#     return filtfilt(b, a, sig)

# def clean_ppg(sig, fs):
#     sig = butter_bandpass(sig, fs)
#     sig = notch_50hz(sig, fs)     
#     return sig.astype(np.float32)

def butter_bandpass_safe(sig, fs, low=0.5, high=4.0, order=4):
    """Band-pass 0.5–4 Hz robusto a fs bajos."""
    if fs <= 0:
        print(f"[WARN] fs={fs} Hz no válido → skip band-pass")
        return sig
    nyq = 0.5 * fs
    high = min(high, 0.45 * fs)       # que high < 0.5*fs
    if low >= high:                   # aún inválido
        print(f"[WARN] fs={fs} Hz demasiado bajo para 0.5–4 Hz → skip band-pass")
        return sig
    low_norm, high_norm = low/nyq, high/nyq
    try:
        b, a = butter(order, [low_norm, high_norm], btype="band")
        return filtfilt(b, a, sig)
    except ValueError as e:
        print(f"[WARN] butter() falló ({e}) → skip band-pass")
        return sig

def notch_50_safe(sig, fs, q=30):
    """Notch 50 Hz solo si fs lo permite."""
    if fs < 100:                      # Nyquist < 50 Hz
        return sig
    nyq = 0.5 * fs
    w0 = 50 / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, sig)

def clean_ppg(sig, fs):
    sig = butter_bandpass_safe(sig, fs)
    sig = notch_50_safe(sig, fs)
    return sig.astype(np.float32)

# ───────────────────────────────
# .avro to raw HDF5
# ───────────────────────────────
def avro_to_hdf5(root_dir, out_h5):
    with h5py.File(out_h5, "w") as fout:
        for fpath in list_avro_files(root_dir):
            participant = os.path.basename(os.path.dirname(os.path.dirname(fpath)))  # P41_empatica
            group = fout.require_group(participant)
            seg_name = os.path.splitext(os.path.basename(fpath))[0]                  # 1-1-P41_…
            with open(fpath, "rb") as f:
                rec = next(fastavro.reader(f))          # exactly one record per file
                bvp = rec["rawData"]["bvp"]
                fs  = bvp["samplingFrequency"]          # 64
                vals = np.asarray(bvp["values"], dtype=np.float32)
                dset = group.create_dataset(
                    seg_name, data=vals,
                    compression="gzip", compression_opts=4, dtype='float32'
                )
                dset.attrs["fs"] = fs
                dset.attrs["t0_us"] = bvp["timestampStart"]
    print(f"Raw PPG segments saved → {out_h5}")

# ───────────────────────────────
# cleaning
# ───────────────────────────────
def clean_hdf5(in_h5, out_h5):
    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for part in fin:
            pg_out = fout.create_group(part)
            for seg in fin[part]:
                sig  = fin[part][seg][...]
                fs   = fin[part][seg].attrs["fs"]
                sig_c = clean_ppg(sig, fs)
                ds = pg_out.create_dataset(
                    seg, data=sig_c,
                    compression="gzip", compression_opts=4, dtype='float32'
                )
                ds.attrs.update(fin[part][seg].attrs)
    print(f"Cleaned PPG saved → {out_h5}")

# ───────────────────────────────
# user-specific z-score
# ───────────────────────────────
def normalize_hdf5(in_h5, out_h5):
    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for part in fin:
            concat = np.concatenate([fin[part][seg][...] for seg in fin[part]])
            mu, sigma = concat.mean(), concat.std() or 1.0
            pg_out = fout.create_group(part)
            for seg in fin[part]:
                norm = (fin[part][seg][...] - mu) / sigma
                ds = pg_out.create_dataset(
                    seg, data=norm.astype(np.float32),
                    compression="gzip", compression_opts=4, dtype='float32'
                )
                ds.attrs.update(fin[part][seg].attrs)
                ds.attrs["mu"], ds.attrs["sigma"] = float(mu), float(sigma)
    print(f"Normalised PPG saved → {out_h5}")

# ───────────────────────────────
# windowing (10 s, 5 s stride)
# ───────────────────────────────
def sliding_windows(sig, w, s):
    if len(sig) < w: return []
    idx = np.arange(0, len(sig) - w + 1, s)
    return np.stack([sig[i:i+w] for i in idx])

def window_hdf5(in_h5, out_h5, win_sec=10, step_sec=5):
    with h5py.File(in_h5, "r") as fin, h5py.File(out_h5, "w") as fout:
        for part in fin:
            fs = float(list(fin[part].values())[0].attrs["fs"])   # same for all segs
            w, s = int(win_sec*fs), int(step_sec*fs)
            pg_out = fout.create_group(part)
            for seg in fin[part]:
                sig = fin[part][seg][...]
                win = sliding_windows(sig, w, s)
                if win.size == 0: continue
                pg_out.create_dataset(
                    seg, data=win, compression="gzip", compression_opts=4
                )
            print(f"{part}: windowed.")
    print(f"Windowed PPG saved → {out_h5}")

# ───────────────────────────────
# main
# ───────────────────────────────
if __name__ == "__main__":
    ROOT_DIR       = "../data/raw/Empatica Avro Files"
    RAW_H5         = "../data/interim/ppg_raw.h5"
    CLEAN_H5       = "../data/interim/ppg_clean.h5"
    NORM_H5        = "../data/interim/ppg_norm.h5"
    WIN_H5         = "../data/interim/ppg_windows.h5"

    avro_to_hdf5(ROOT_DIR, RAW_H5)
    clean_hdf5(RAW_H5, CLEAN_H5)
    normalize_hdf5(CLEAN_H5, NORM_H5)
    window_hdf5(NORM_H5, WIN_H5, win_sec=10, step_sec=5)
