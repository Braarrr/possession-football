import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("./data/tactical")
ATD_TXT   = DATA_DIR / "BAR-ATH_tactical_atd.txt"
ATD_XML   = DATA_DIR / "BAR-ATH_tactical_atd.xml"
SMART_TAG = DATA_DIR / "BAR-ATH_tactical_smart_tagging.xml"
GT_PATH   = Path("evaluation/gt/BAR-ATH_possession_gt_public.csv")

# ── Match constants ────────────────────────────────────────────────────────────
FPS            = 25.0
PITCH_X        = 105.0    # meters
PITCH_Y        = 68.0     # meters
TOTAL_FRAMES   = 199_319  # ATD frame_ids: 1 → 199319
N_SLOTS        = 46

# Graded video frames (0-indexed). Everything outside these ranges → X for free.
FIRST_HALF  = (11_185, 90_651)    # video frames, inclusive
SECOND_HALF = (110_808, 190_197)  # video frames, inclusive

def is_graded(video_frame: int) -> bool:
    return (FIRST_HALF[0]  <= video_frame <= FIRST_HALF[1] or
            SECOND_HALF[0] <= video_frame <= SECOND_HALF[1])

# ── Slot → team mapping (confirmed from XML) ──────────────────────────────────
HOME_SLOTS = list(range(0, 24))   # Barcelona (slot 23 = GK)
AWAY_SLOTS = list(range(24, 46))  # Athletic Club
SLOT_TEAM  = np.array(["H"] * 24 + ["A"] * 22, dtype=object)

# ── Dead ball event codes (from DATA.md table) ─────────────────────────────────
# Only SET PIECES and GOALS stop play. All other codes are live-play tactical tags.
DEAD_BALL_CODES = {"SET PIECES", "GOALS"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. SMART TAGGING
# ══════════════════════════════════════════════════════════════════════════════

def parse_smart_tagging(path=SMART_TAG):
    """
    Parse smart_tagging.xml into a DataFrame.

    Time format: <start> and <end> are seconds from the first frame of the
    tactical video. Convert to video frames with round(seconds * FPS).
    This is already 0-indexed (video frame = seconds * 25), confirmed by DATA.md.

    Key insight from DATA.md:
      - 638 total events, but only 63 are dead ball (SET PIECES=59, GOALS=4).
      - The other 575 events (Build Up, Progression, etc.) happen during live play.
      - Copying all 638 events as dead ball gives 70% dead — completely wrong.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for inst in root.iter("instance"):
        labels = {
            lbl.findtext("group", ""): lbl.findtext("text", "")
            for lbl in inst.findall("label")
        }
        start_s = float(inst.findtext("start", "0"))
        end_s   = float(inst.findtext("end",   "0"))
        rows.append({
            "id":           inst.findtext("ID", ""),
            "code":         inst.findtext("code", ""),
            "type":         labels.get("Type", ""),
            "team":         labels.get("Team", ""),
            "half":         labels.get("Half", ""),
            "start_s":      start_s,
            "end_s":        end_s,
            "start_frame":  round(start_s * FPS),  # video frame (0-indexed)
            "end_frame":    round(end_s   * FPS),  # video frame (0-indexed)
            "duration_s":   end_s - start_s,
        })

    df = pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)

    # Show breakdown
    dead = df[df["code"].isin(DEAD_BALL_CODES)]
    live = df[~df["code"].isin(DEAD_BALL_CODES)]
    print(f"Smart Tagging: {len(df)} total events")
    print(f"\n  Dead ball events ({len(dead)}) — codes: {DEAD_BALL_CODES}")
    if len(dead):
        print(dead["type"].value_counts().to_string())
    print(f"\n  Live play events ({len(live)}) — these are NOT dead ball:")
    print(live["code"].value_counts().to_string())

    return df


def build_dead_ball_mask(smart_df, total_video_frames=TOTAL_FRAMES):
    """
    Boolean array indexed by VIDEO FRAME (0-indexed): True = dead ball.

    Fix vs v2: only mark SET PIECES and GOALS as dead ball.
    The 575 live-play tactical events (Progression, Build Up, etc.) are NOT dead ball.

    Note: Smart Tagging start/end are already video frames (seconds * 25), so no
    offset is needed here. But ATD frame_ids need offset (-1) when converting.
    """
    n    = total_video_frames   # index 0 → total_video_frames-1
    mask = np.zeros(n, dtype=bool)

    dead_events = smart_df[smart_df["code"].isin(DEAD_BALL_CODES)]
    for _, row in dead_events.iterrows():
        sf = max(0, int(row["start_frame"]))
        ef = min(n - 1, int(row["end_frame"]))
        mask[sf:ef + 1] = True

    n_dead = mask.sum()
    print(f"\nDead ball: {n_dead:,} video frames ({n_dead / n * 100:.1f}%)")
    print(f"Live ball: {n - n_dead:,} video frames ({(n - n_dead) / n * 100:.1f}%)")
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARSE ATD TXT
# ══════════════════════════════════════════════════════════════════════════════

def parse_atd_txt(path=ATD_TXT, max_frames=None, skip_frames=0):
    """
    Parse tracking data → numpy arrays.

    skip_frames: number of ATD lines to skip before starting to fill arrays.
      Use skip_frames=11_185 to jump straight to the public GT window
      (ATD frame_id 11186 = video frame 11185).

    Frame indexing (important!):
      ATD file uses frame_id 1…199319 (1-indexed).
      To get video frame: video_frame = frame_id - 1.
      We store the raw frame_ids here; callers must subtract 1 when needed.

    Coordinate system:
      Values are normalized 0–1. Multiply by PITCH_X/PITCH_Y for meters.
      Third value per slot = speed in m/s (can be NaN even when x,y exist).

    Bug #1 fix:
      Old: if "NaN" not in ball_str → skips "0.55,0.34,NaN" (speed=NaN)
      New: split first, only check parts[0] and parts[1] (x and y)
      DATA.md says ball is visible on 56.9% of frames — expect ~113k after fix.

    Returns dict:
      frame_ids  [N]      int32   — ATD frame_ids (1-indexed, NOT video frames)
      px         [N, 46]  float32 — player x in meters (NaN = not tracked)
      py         [N, 46]  float32 — player y in meters
      pspd       [N, 46]  float32 — player speed m/s (NaN if unavailable)
      bx         [N]      float32 — ball x in meters (NaN = not detected)
      by         [N]      float32 — ball y in meters
    """
    N  = max_frames or TOTAL_FRAMES
    t0 = time.time()

    frame_ids = np.zeros(N, dtype=np.int32)
    px        = np.full((N, N_SLOTS), np.nan, dtype=np.float32)
    py        = np.full((N, N_SLOTS), np.nan, dtype=np.float32)
    pspd      = np.full((N, N_SLOTS), np.nan, dtype=np.float32)
    bx        = np.full(N, np.nan, dtype=np.float32)
    by        = np.full(N, np.nan, dtype=np.float32)

    if skip_frames and max_frames:
        mode = f"[GT window: skip {skip_frames:,}, read {max_frames:,} frames]"
    elif max_frames:
        mode = f"[test: first {N:,} frames]"
    else:
        mode = "[full match]"
    print(f"Parsing {path.name} {mode} ...")

    with open(path, "r") as fh:
        # Jump over lines we don't need (e.g. pre-kickoff frames)
        if skip_frames:
            for _ in range(skip_frames):
                next(fh, None)

        i = -1
        for i, line in enumerate(fh):
            if max_frames and i >= max_frames:
                i -= 1   # last stored index is i-1
                break
            line = line.strip()
            if not line:
                continue

            # Line format: frame_id:p0;p1;...;p45:ball
            c_first = line.index(":")
            c_last  = line.rindex(":")
            frame_ids[i] = int(line[:c_first])

            # ── Ball (Bug #1 fix) ────────────────────────────────────────────
            # Split first, then check x and y individually.
            # Speed (parts[2]) can be NaN without the ball being absent.
            ball_parts = line[c_last + 1:].split(",")
            if ball_parts[0] != "NaN" and ball_parts[1] != "NaN":
                bx[i] = float(ball_parts[0]) * PITCH_X
                by[i] = float(ball_parts[1]) * PITCH_Y

            # ── Players ──────────────────────────────────────────────────────
            player_section = line[c_first + 1 : c_last]
            for s, slot_str in enumerate(player_section.split(";")):
                if s >= N_SLOTS:
                    break
                parts = slot_str.split(",")
                if parts[0] == "NaN":
                    continue
                px  [i, s] = float(parts[0]) * PITCH_X
                py  [i, s] = float(parts[1]) * PITCH_Y
                if parts[2] != "NaN":
                    pspd[i, s] = float(parts[2])

            if i > 0 and i % 20_000 == 0:
                pct = i / N * 100
                print(f"  {i:,}/{N:,} ({pct:.0f}%)  {time.time()-t0:.0f}s")

    n = i + 1
    data = dict(frame_ids=frame_ids[:n], px=px[:n], py=py[:n],
                pspd=pspd[:n], bx=bx[:n], by=by[:n])

    # Stats
    ball_ok  = ~np.isnan(bx[:n])
    n_det    = (~np.isnan(px[:n])).sum(axis=1)
    n_h      = (~np.isnan(px[:n, :24])).sum(axis=1)
    n_a      = (~np.isnan(px[:n, 24:])).sum(axis=1)

    print(f"\n✅ Parsed {n:,} frames in {time.time()-t0:.1f}s")
    print(f"\n── Ball detection (expect ~57% after Bug #1 fix) ──")
    print(f"  Detected : {ball_ok.sum():>8,}  ({ball_ok.mean()*100:.1f}%)")
    print(f"  Missing  : {(~ball_ok).sum():>8,}  ({(~ball_ok).mean()*100:.1f}%)")
    print(f"\n── Players tracked per frame ──────────────────────")
    print(f"  Total avg : {n_det.mean():.1f} / {N_SLOTS}  (expect ~20.5 during play)")
    print(f"  Home avg  : {n_h.mean():.1f} / 24")
    print(f"  Away avg  : {n_a.mean():.1f} / 22")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 3. POSSESSION LABELS
# ══════════════════════════════════════════════════════════════════════════════

def nearest_player_labels(data, dead_mask):
    """
    Assign a raw possession label to each tracked frame.

    Priority order:
      1. Dead ball (from dead_mask) → D
      2. Ball detected → team of nearest player → H or A
      3. Ball missing, not dead → X (abstain)

    Frame indexing note:
      data["frame_ids"] are ATD frame_ids (1-indexed).
      dead_mask is indexed by video frame (0-indexed).
      Conversion: video_frame = frame_id - 1.

    Vectorized: compute all distances at once for frames where ball exists.
    This is ~100x faster than a Python loop over 200k frames.
    """
    bx       = data["bx"]
    by       = data["by"]
    px       = data["px"]
    py       = data["py"]
    fids     = data["frame_ids"]
    N        = len(bx)

    labels       = np.full(N, "X", dtype=object)
    ball_present = ~np.isnan(bx)
    valid_idx    = np.where(ball_present)[0]

    # Vectorized distance calculation
    if len(valid_idx) > 0:
        bx_v = bx[valid_idx][:, None]     # [M, 1] broadcast
        by_v = by[valid_idx][:, None]
        px_v = px[valid_idx]               # [M, 46]
        py_v = py[valid_idx]

        dist = np.sqrt((px_v - bx_v)**2 + (py_v - by_v)**2)
        dist[np.isnan(dist)] = np.inf      # untracked slot = infinitely far

        nearest = np.argmin(dist, axis=1)  # [M]
        labels[valid_idx] = SLOT_TEAM[nearest]

    # Override: dead ball takes priority over everything else
    # Convert ATD frame_ids to video frames for dead_mask lookup
    for i, fid in enumerate(fids):
        vid = int(fid) - 1   # Bug #3 fix: ATD frame_id → video frame
        if 0 <= vid < len(dead_mask) and dead_mask[vid]:
            labels[i] = "D"

    _label_dist("Raw (nearest player)", labels)
    return labels


def majority_vote_smooth(labels, window=50):
    """
    Sliding window majority vote to reduce label flicker.

    Why this matters for spellF1:
      Raw labels change every few frames → hundreds of tiny fake possessions.
      After smoothing → label only changes when a team consistently holds the ball.
      Fewer possessions → better match with ground truth possession count.

    Dead ball frames (D) are preserved and excluded from other frames' votes.
    This prevents dead ball periods from being "overruled" by nearby H/A frames.
    """
    enc  = {"X": 0, "H": 1, "A": 2, "D": 3}
    dec  = {v: k for k, v in enc.items()}
    num  = np.array([enc.get(str(l), 0) for l in labels], dtype=np.int8)
    out  = num.copy()
    half = window // 2
    N    = len(labels)

    for i in range(N):
        if labels[i] == "D":
            continue               # dead ball is sticky, not subject to voting
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        chunk = num[lo:hi]
        chunk = chunk[chunk != 3]  # exclude dead ball frames from the vote
        if len(chunk) == 0:
            continue
        vals, counts = np.unique(chunk, return_counts=True)
        out[i] = vals[np.argmax(counts)]

    smoothed = np.array([dec[v] for v in out], dtype=object)
    _label_dist(f"Smoothed (window={window})", smoothed)
    return smoothed


def _label_dist(name, labels):
    n = len(labels)
    print(f"  {name:<35} "
          f"H={np.sum(labels=='H'):,}  "
          f"A={np.sum(labels=='A'):,}  "
          f"D={np.sum(labels=='D'):,}  "
          f"X={np.sum(labels=='X'):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EVALUATE AGAINST PUBLIC GROUND TRUTH
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_against_gt(labels, frame_ids, gt_path=GT_PATH):
    """
    Compare predictions against the 5-minute public ground truth.

    Frame alignment (Bug #3 fix):
      GT uses 0-indexed video frames (e.g. 11185, 11186, ...).
      ATD frame_ids are 1-indexed (11186, 11187, ...).
      We convert with: video_frame = frame_id - 1.

    Metrics reported here: per-class precision, recall, F1, macroF1.
    This is NOT spellF1 — run evaluation/run_evaluation.py for the real score.

    Think of this as a sanity check, not a performance number.
    """
    if not Path(gt_path).exists():
        print(f"Ground truth not found: {gt_path}")
        print("Run from the repo root directory.")
        return None, 0.0

    gt = pd.read_csv(gt_path)

    # Bug #3 fix: convert ATD frame_ids (1-indexed) to video frames (0-indexed)
    frame_to_label = {
        int(fid) - 1: str(lbl)
        for fid, lbl in zip(frame_ids, labels)
    }

    rows = [
        {"frame": int(r["frame"]),
         "gt":    str(r["label"]),
         "pred":  frame_to_label.get(int(r["frame"]), "X")}
        for _, r in gt.iterrows()
        if str(r["label"]) != "X"    # skip ungraded frames in GT
    ]
    df = pd.DataFrame(rows)

    # Hard assertion: if no GT frames were matched, the frame window is wrong.
    # This happens when parse_atd_txt only reads pre-kickoff frames (e.g. max_frames=7500).
    if len(df) == 0:
        raise ValueError(
            "evaluate_against_gt: 0 GT frames matched predictions.\n"
            "  GT window starts at video frame 11185 (ATD frame_id 11186).\n"
            "  Run with --gt or --full so the parser reaches that range.\n"
            "  Using max_frames=7500 without skip_frames only covers frames 0-7499."
        )

    print(f"Evaluating on {len(df):,} graded GT frames "
          f"(public 5-min window: video frames {FIRST_HALF[0]}–18684)")

    # Per-class F1 (no sklearn needed)
    print(f"\n{'Class':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("─" * 50)
    f1s = []
    for lbl, name in [("H", "Home"), ("A", "Away"), ("D", "Dead")]:
        tp = ((df["gt"] == lbl) & (df["pred"] == lbl)).sum()
        fp = ((df["gt"] != lbl) & (df["pred"] == lbl)).sum()
        fn = ((df["gt"] == lbl) & (df["pred"] != lbl)).sum()
        sup = (df["gt"] == lbl).sum()
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
        print(f"{name:<8} {p:>10.3f} {r:>8.3f} {f1:>8.3f} {sup:>9,}")

    macro_f1 = np.mean(f1s)
    print("─" * 50)
    print(f"{'macroF1':<27} {macro_f1:>8.3f}")
    print(f"\n⚠️  This is macroF1 only. For spellF1 (the ranking metric):")
    print(f"    python evaluation/run_evaluation.py")

    # Confusion matrix
    print(f"\nConfusion matrix  (rows = ground truth, cols = prediction)")
    all_lbls = ["H", "A", "D", "X"]
    print(f"{'GT/Pred':<8}" + "".join(f"{l:>8}" for l in all_lbls))
    for gt_l in all_lbls:
        row = f"{gt_l:<8}"
        for pr_l in all_lbls:
            row += f"{((df['gt']==gt_l)&(df['pred']==pr_l)).sum():>8,}"
        print(row)

    _print_timeline(df)
    return df, macro_f1


def _print_timeline(df, frames_per_char=25):
    """ASCII timeline: compare GT vs our prediction over the 5-min window."""
    sym = {"H": "H", "A": "A", "D": "·", "X": " "}
    gt_line = pred_line = err_line = ""

    for i in range(0, len(df), frames_per_char):
        chunk   = df.iloc[i:i + frames_per_char]
        gt_m    = chunk["gt"].mode()[0]
        pred_m  = chunk["pred"].mode()[0]
        gt_line   += sym.get(gt_m,   "?")
        pred_line += sym.get(pred_m, "?")
        err_line  += "^" if gt_m != pred_m else " "

    w = len(gt_line)
    print(f"\n── 5-minute timeline (1 char = {frames_per_char} frames = 1s) ──────────")
    print(f"  GT  : {gt_line}")
    print(f"  Pred: {pred_line}")
    print(f"  Err : {err_line}")
    print(f"  {'0s':<{w//2}}{'5min':>{w//2}}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD SUBMISSION CSV
# ══════════════════════════════════════════════════════════════════════════════

def make_submission(labels, frame_ids, out_path="submission.csv"):
    """
    Build the submission CSV: exactly 199,319 rows, frame 0…199318.

    Bug #3 fix:
      ATD frame_ids are 1-indexed. Submission needs 0-indexed video frames.
      Conversion: video_frame = frame_id - 1.

    Ungraded frames (before kickoff, halftime, after whistle) default to X.
    This is safe: ungraded frames are never scored by either metric.

    Graded frame ranges (video frames):
      1st half : 11,185 → 90,651
      2nd half : 110,808 → 190,197
    """
    # Build lookup: video_frame (0-indexed) → label
    vid_to_label = {
        int(fid) - 1: str(lbl)      # ATD frame_id - 1 = video frame
        for fid, lbl in zip(frame_ids, labels)
    }

    rows = [
        {"frame": vf, "label": vid_to_label.get(vf, "X")}
        for vf in range(0, TOTAL_FRAMES)    # 0 to 199318 inclusive
    ]

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    print(f"\n✅ Submission saved → {out_path}")
    print(f"   Rows          : {len(out):,}  (expect 199,319)")
    print(f"   Frame range   : {out['frame'].min()} → {out['frame'].max()}  "
          f"(expect 0 → 199318)")
    vc = out["label"].value_counts()
    print(f"   Label counts  : {dict(vc)}")

    # Sanity check graded vs ungraded
    graded_mask = out["frame"].apply(is_graded)
    graded      = out[graded_mask]
    ungraded    = out[~graded_mask]
    print(f"\n   Graded frames   : {len(graded):,}  "
          f"(labels: {dict(graded['label'].value_counts())})")
    print(f"   Ungraded frames : {len(ungraded):,}  "
          f"(these are never scored — X is optimal here)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os

    # ── Mode selection ────────────────────────────────────────────────────────
    # --gt   : parse only the public GT window (ATD 11186-18685 = video 11185-18684)
    #          Fast (~5s), evaluates spellF1 + macroF1. Default dev loop mode.
    # --full : parse all 199,319 frames, build submission CSV, validate + score.
    #
    # OLD bug: max_frames=7500 with no skip read ATD lines 1-7500 = video frames
    # 0-7499, which is entirely pre-kickoff. GT window starts at video 11185, so
    # evaluate_against_gt always got 0 matching frames (silently wrong).
    # Fix: --gt mode skips 11,185 lines then reads 7,500 (the exact GT window).

    full_mode = "--full" in sys.argv
    # default (no flag) = GT window mode
    GT_START_ATD  = FIRST_HALF[0]   # 11185 lines to skip  (ATD is 1-indexed so
    GT_WINDOW_LEN = 7_500           # frame_id 11186 = video frame 11185)

    print("=" * 60)
    print("MTAH 2027 — Possession Pipeline v4")
    print("=" * 60)
    if full_mode:
        print("Full match mode — will build submission_v4.csv\n")
    else:
        print("GT-window mode: skip 11,185 pre-kickoff frames, read 7,500.\n"
              "Use --full to parse the complete match and build a submission.\n")

    # 1. Smart Tagging
    print("\n── 1. SMART TAGGING ─────────────────────────────────")
    smart_df  = parse_smart_tagging()
    dead_mask = build_dead_ball_mask(smart_df)

    # 2. Tracking
    print("\n── 2. TRACKING ──────────────────────────────────────")
    if full_mode:
        data = parse_atd_txt(max_frames=None, skip_frames=0)
    else:
        data = parse_atd_txt(max_frames=GT_WINDOW_LEN, skip_frames=GT_START_ATD)

    # 3. Raw labels
    print("\n── 3. RAW LABELS ────────────────────────────────────")
    raw_labels = nearest_player_labels(data, dead_mask)

    # 4. Smoothing
    print("\n── 4. SMOOTHING ─────────────────────────────────────")
    smoothed = {}
    for w in [25, 50, 75, 125]:
        smoothed[w] = majority_vote_smooth(raw_labels, window=w)

    # 5. macroF1 evaluation vs public GT
    print("\n── 5. EVALUATION vs PUBLIC GT (window=50) ───────────")
    eval_df, macro_f1 = evaluate_against_gt(smoothed[50], data["frame_ids"])

    # 5b. Inline spellF1 via the official metric library (the actual ranking metric)
    print("\n── 5b. spellF1 (ranking metric) ─────────────────────")
    try:
        _eval_dir = str(Path("evaluation").resolve())
        if _eval_dir not in sys.path:
            sys.path.insert(0, _eval_dir)
        import possession_metrics as pm

        # Build full-length prediction arrays (199,319 frames), fill X outside our window
        def _full_pred(smoothed_labels):
            pred = np.full(TOTAL_FRAMES, "X", dtype=object)
            for fid, lbl in zip(data["frame_ids"], smoothed_labels):
                vf = int(fid) - 1
                if 0 <= vf < TOTAL_FRAMES:
                    pred[vf] = str(lbl)
            return pm.label_codes(
                pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": pred})
            )

        # Build full-length GT array from the public GT CSV
        gt_csv = pm.read_label_csv(str(GT_PATH))
        gt_arr = np.full(TOTAL_FRAMES, "X", dtype=object)
        for _, row in gt_csv.iterrows():
            gt_arr[int(row["frame"])] = str(row["label"])
        gt_codes = pm.label_codes(
            pd.DataFrame({"frame": np.arange(TOTAL_FRAMES), "label": gt_arr})
        )

        # Evaluate all smoothing windows
        print(f"  {'Window':>8}  {'spellF1':>9}  {'macroF1':>9}")
        print(f"  {'─'*8}  {'─'*9}  {'─'*9}")
        best_w, best_s = 50, -1.0
        for w in [25, 50, 75, 125]:
            pc = _full_pred(smoothed[w])
            r = pm.evaluate(gt_codes, pc)
            s = r["spell_f1_mean"]
            print(f"  {w:>8}  {s:>9.4f}  {r['macro_f1']:>9.4f}")
            if s > best_s:
                best_w, best_s = w, s

        print(f"\n  Best window = {best_w}  →  spellF1 {best_s:.4f}")
        print(f"  (baseline_smooth50 target to beat = 0.2433 on full match)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  spellF1 inline eval failed: {e}")
        print(f"  Run: cd evaluation && python run_evaluation.py")
        best_w = 50

    # 6. Full match: build submission and validate
    if full_mode:
        print("\n── 6. BUILD SUBMISSION ──────────────────────────────")
        sub_path = "submission_v4.csv"
        make_submission(smoothed[best_w], data["frame_ids"], sub_path)

        print("\n── 7. VALIDATE ──────────────────────────────────────")
        ret = os.system(f"python evaluation/validate_submission.py {sub_path}")
        if ret == 0:
            print("\n── 8. SCORE ─────────────────────────────────────────")
            import shutil
            shutil.copy(sub_path, f"evaluation/results/{sub_path}")
            os.system("cd evaluation && python run_evaluation.py")

    print("\n✅ Done.")