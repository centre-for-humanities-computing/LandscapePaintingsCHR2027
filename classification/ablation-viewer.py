"""
Interactive terminal attribution viewer. Loads the single combined attribution parquet produced by
ablation.py.

Controls:
  i <id>    Show image by its stable identifier if ID-column is present, 
            otherwise row number.
  b <1-2>   Browse ranked list. Each opens one grid window per active model.
              1 strongest single-segment classification flips
              2 strongest segment-pair contrast within an image (highlighted)
  c         Toggle active columns (chooses the embedding models)
  p         Filter to a single painter (search by name, pick from matches;
            Enter with no text clears the filter).
  q         Quit

BASELINE: every pp value/ranking is relative to ablation_baseline_<clf>,
read directly from the attribution parquet.

"""

import io
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────

ATTRIBUTION_PARQUET = "data/ablation_scores-fold.parquet"   
# or _full -- matches ablation.py's output

GRID_SIZE           = 4        # must match training
HEATMAP_ALPHA       = 0.55
ABLATION_CMAP       = "PRGn"   # green = Danish, purple = Norwegian
FIG_WIDTH_PER_PANEL = 3.8
FIG_HEIGHT          = 4.2
PCT_SCALE_MAX       = 40.0     # pp shift at which color is fully saturated
COLOR_GAMMA         = 0.5      # stretches color contrast for SMALL pp shifts
TOP_N               = 20       # images per browse mode
GRID_COLS           = 5       
SAVEFIG_DPI         = 300 

plt.rcParams["savefig.dpi"] = SAVEFIG_DPI

# ─────────────────────────────────────────────────────────────────────────────

class SignedPowerNorm(Normalize):
    """Diverging color normalization that stretches contrast for SMALL pp
    shifts at the cost of compressing contrast among large-magnitude outliers. 
    Maps value -> sign(value) * (|value|/vmax)**gamma before coloring, so e.g.
    with gamma=0.5 a modest shift like 10pp already shows a clearly
    different color from ~0pp, instead of being washed out on a straight
    linear scale calibrated to also fit rare 60pp extremes."""
    def __init__(self, gamma=COLOR_GAMMA, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        value = np.ma.masked_invalid(np.asarray(value, dtype=float))
        clipped = np.clip(value, -self.vmax, self.vmax)
        t = np.sign(clipped) * (np.abs(clipped) / self.vmax) ** self.gamma
        return np.ma.masked_array((t + 1) / 2)

    def inverse(self, value):
        t = np.asarray(value) * 2 - 1
        return np.sign(t) * (np.abs(t) ** (1 / self.gamma)) * self.vmax


def make_ablation_norm():
    """Shared color normalization for every ablation-overlay panel — one
    place to tune PCT_SCALE_MAX/COLOR_GAMMA and have it apply everywhere."""
    return SignedPowerNorm(gamma=COLOR_GAMMA, vmin=-PCT_SCALE_MAX, vmax=PCT_SCALE_MAX)


def load_attribution_parquet(path):
    """Load the single combined attribution parquet."""
    print(f"[info] Loading: {path}")
    df = pl.read_parquet(path).to_pandas().reset_index(drop=True)
    model_cols = [c for c in df.columns if c.startswith("ablation_scores_")]
    print(f"[info] {len(df)} images, {len(model_cols)} model column(s)")
    return df


def _stable_id_col(attr_df):
    """'id' if present, else None -- callers then fall back to the raw
    row position."""
    return "id" if "id" in attr_df.columns else None


def _row_stable_id(row, idx):
    """Same priority as _stable_id_col, but for a single row (Series)
    rather than a dataframe -- avoids repeating row.get('id', idx) at
    every call site that labels one image."""
    return row.get("id", idx)


# ── image / grid helpers ──────────────────────────────────────────────────────

def bytes_to_pil(raw):
    return Image.open(io.BytesIO(raw)).convert("RGB")


def grid_dims(img_w, img_h):
    """Crop cell size / row & col counts, matching training's load_image()."""
    cell_size = min(img_w, img_h) // GRID_SIZE
    return cell_size, img_h // cell_size, img_w // cell_size


def crop_to_grid(img):
    """Crop to the exact region covered by GRID_SIZE x GRID_SIZE crops."""
    cell_size, n_rows, n_cols = grid_dims(*img.size)
    return img.crop((0, 0, n_cols * cell_size, n_rows * cell_size))


def scores_to_overlay(scores, img_w, img_h):
    """Expand flat per-crop scores into a pixel-exact (H, W) overlay array."""
    cell_size, n_rows, n_cols = grid_dims(img_w, img_h)
    arr = np.asarray(scores, dtype=float).ravel()
    if len(arr) != n_rows * n_cols:
        side = int(round(np.sqrt(len(arr))))
        arr, n_rows, n_cols = arr[:side * side], side, side
    grid = arr.reshape(n_rows, n_cols)
    canvas = np.zeros((img_h, img_w), dtype=float)
    for r in range(n_rows):
        for c in range(n_cols):
            canvas[r * cell_size:(r + 1) * cell_size, c * cell_size:(c + 1) * cell_size] = grid[r, c]
    return canvas


def draw_grid_lines(ax, img_w, img_h):
    cell_size, n_rows, n_cols = grid_dims(img_w, img_h)
    for r in range(1, n_rows):
        ax.axhline(r * cell_size - 0.5, color="white", linewidth=0.8, alpha=0.8)
    for c in range(1, n_cols):
        ax.axvline(c * cell_size - 0.5, color="white", linewidth=0.8, alpha=0.8)


def pretty_label(suffix):
    return suffix.replace("embedding_projected", "head").replace("embedding_backbone", "backbone").replace("_", " ")


# ── baseline / score helpers ───────────────────────────────────────────────────

def get_baseline(row, suffix):
    """Returns the baseline P(danish) for this row/model, or None if absent."""
    val = row.get(f"ablation_baseline_{suffix}")
    return float(val) if val is not None else None


def get_pp_scores(row, suffix):
    """Returns the per-crop pp array (already *100), or None if this row has
    no ablation scores for this model."""
    raw = row.get(f"ablation_scores_{suffix}")
    if raw is None:
        return None
    return np.asarray(raw, dtype=float) * 100


# ── ablation figure (single image, one panel per model) ───────────────────────

def draw_pp_overlay(ax, img_arr, display, W, H, norm, fontsize=5.5, suffix=""):
    """Draws image + color-coded ablation overlay + grid lines + per-cell
    pp text on one axis."""
    cell_size, n_rows, n_cols = grid_dims(W, H)
    overlay = scores_to_overlay(display, W, H)
    ax.imshow(img_arr, zorder=1)
    im = ax.imshow(overlay, cmap=ABLATION_CMAP, alpha=HEATMAP_ALPHA,
                    norm=norm, interpolation="nearest", zorder=2)
    draw_grid_lines(ax, W, H)
    grid_display = display.reshape(n_rows, n_cols)
    for r in range(n_rows):
        for c in range(n_cols):
            cx, cy = c * cell_size + cell_size / 2, r * cell_size + cell_size / 2
            ax.text(cx, cy, f"{grid_display[r, c]:+.1f}{suffix}", ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.45))
    return im


def render_ablation_figure(img_arr, W, H, row, ablation_cols):
    """Original + a color-coded, percentage-labeled grid per ablation column.
    """
    cols = [c for c in ablation_cols if row.get(c) is not None]
    if not cols:
        return None

    fig, axes = plt.subplots(1, len(cols) + 1,
                              figsize=(FIG_WIDTH_PER_PANEL * (len(cols) + 1), FIG_HEIGHT + 0.5),
                              squeeze=False)
    axes[0][0].imshow(img_arr)
    axes[0][0].set_title("Original", fontsize=8, pad=4)
    axes[0][0].axis("off")

    norm = make_ablation_norm()
    for i, col in enumerate(cols):
        suffix = col.replace("ablation_scores_", "")
        display = get_pp_scores(row, suffix)   # pp, + = Danish
        baseline_p = get_baseline(row, suffix)

        ax = axes[0][i + 1]
        draw_pp_overlay(ax, img_arr, display, W, H, norm, fontsize=6, suffix="pp")

        sm = plt.cm.ScalarMappable(cmap=ABLATION_CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=25)
        label = (f"← Norwegian (P={1 - baseline_p:.2f})   Danish (P={baseline_p:.2f}) →   (pp shift in P(Danish))"
                  if baseline_p is not None else "← Norwegian        Danish →   (pp shift in P(Danish))")
        cbar.set_label(label, fontsize=6)
        cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}pp"))
        cbar.ax.tick_params(labelsize=5)

        ax.set_title(f"Danish / Norwegian\n{pretty_label(suffix)}", fontsize=7, pad=4)
        ax.axis("off")

    fig.suptitle("Ablation scores", fontsize=9)
    fig.tight_layout(pad=0.5, rect=(0, 0, 1, 0.95))
    return fig


def build_info_text(idx, row, ablation_cols):
    """ID/artist/true-class/filename header, plus one P(danish) baseline
    line per active model."""
    artist, cls, filename = row.get("artist", "—"), row.get("main_class", "—"), str(row.get("filename", ""))
    stable_id = _row_stable_id(row, idx)
    header_parts = [f"ID: {stable_id}", f"Artist: {artist}", f"True: {cls}"]
    # only show File separately if it's not already shown as ID above
    # (the normal case now, since 'filename' usually IS the stable id)
    if filename and filename not in ("", "nan") and str(stable_id) != filename:
        header_parts.append(f"File: {filename}")
    header = "   |   ".join(header_parts)

    baseline_bits = []
    for col in ablation_cols:
        suffix = col.replace("ablation_scores_", "")
        baseline = get_baseline(row, suffix)
        if baseline is not None:
            baseline_bits.append(f"{pretty_label(suffix)}: P(Danish)={baseline:.3f}")

    if baseline_bits:
        return header + "\n" + "     ".join(baseline_bits)
    return header


def render_image(idx, attr_df, ablation_cols, artist_filter=None):
    """idx here is a resolved row position, might not be stable across sessions."""
    if idx < 0 or idx >= len(attr_df):
        print(f"  [error] Position {idx} out of range (0–{len(attr_df) - 1})")
        return
    row = attr_df.iloc[idx]
    if artist_filter and row.get("artist") != artist_filter:
        print(f"  [note] #{_row_stable_id(row, idx)} is by '{row.get('artist')}', not the current painter "
              f"filter '{artist_filter}' -- showing it anyway.")
    img = crop_to_grid(bytes_to_pil(row["image"]))
    W, H = img.size
    fig = render_ablation_figure(np.array(img), W, H, row, ablation_cols)
    if fig is not None:
        fig.text(0.5, 0.01, build_info_text(idx, row, ablation_cols), ha="center", va="bottom", fontsize=7.5, wrap=True,
                  bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", alpha=0.8))
        fig.subplots_adjust(bottom=0.24)
        plt.figure(fig.number)
        plt.show(block=False)
    plt.pause(0.1)


# ── ranking: the 'g' segment grid and browse modes 1-3 ────────────────────────

def _iter_valid_grids(attr_df, abl_col, artist_filter=None):
    """Yields (idx, row, pp_array, grid_side) for every row that has a
    valid, square-shaped ablation score array for this column. The
    setup is shared by every ranking function below."""
    if artist_filter:
        attr_df = attr_df[attr_df["artist"] == artist_filter]
    suffix = abl_col.replace("ablation_scores_", "")
    for idx, row in attr_df.iterrows():
        arr = get_pp_scores(row, suffix)
        if arr is None:
            continue
        side = int(round(np.sqrt(len(arr))))
        if side * side != len(arr) or side == 0:
            continue
        yield idx, row, arr, side


def top_flip_for_column(attr_df, abl_col, n=TOP_N, artist_filter=None):
    """Rank images where a single segment's removal flips the predicted
    class.
    Returns df[pp, cell_r, cell_c, grid_side]."""
    suffix = abl_col.replace("ablation_scores_", "")
    records = []
    for idx, row, arr, side in _iter_valid_grids(attr_df, abl_col, artist_filter):
        baseline = get_baseline(row, suffix)
        if baseline is None:
            continue
        p_removed = baseline - arr / 100.0   # arr is already in pp (see get_pp_scores)
        flips = (baseline - 0.5) * (p_removed - 0.5) < 0
        if not flips.any():
            continue
        flip_pp = np.where(flips, arr, 0.0)
        flat_i = int(np.argmax(np.abs(flip_pp)))
        r, c = divmod(flat_i, side)
        records.append({"idx": idx, "pp": float(arr[flat_i]), "cell_r": r, "cell_c": c, "grid_side": side})

    if not records:
        return pd.DataFrame(columns=["pp", "cell_r", "cell_c", "grid_side"])
    seg_df = pd.DataFrame(records).set_index("idx")
    return seg_df.reindex(seg_df["pp"].abs().sort_values(ascending=False).index).head(n)


def top_internal_contrast_for_column(attr_df, abl_col, n=TOP_N, artist_filter=None):
    """Rank images by the internal spread of a single model's segment scores:
    max(pp) - min(pp) across all crops in that image. 
    Returns df[spread, max_pp, max_r, max_c, min_pp, min_r, min_c, grid_side]."""
    records = []
    for idx, row, arr, side in _iter_valid_grids(attr_df, abl_col, artist_filter):
        max_i, min_i = int(np.argmax(arr)), int(np.argmin(arr))
        max_r, max_c = divmod(max_i, side)
        min_r, min_c = divmod(min_i, side)
        records.append({"idx": idx, "spread": float(arr[max_i] - arr[min_i]),
                         "max_pp": float(arr[max_i]), "max_r": max_r, "max_c": max_c,
                         "min_pp": float(arr[min_i]), "min_r": min_r, "min_c": min_c,
                         "grid_side": side})
    if not records:
        return pd.DataFrame(columns=["spread", "max_pp", "max_r", "max_c", "min_pp", "min_r", "min_c", "grid_side"])
    df = pd.DataFrame(records).set_index("idx")
    return df.sort_values("spread", ascending=False).head(n)


# ── label functions + text tables (one "family" per mode: prob/margin/segment/internal) ─
# NOTE: `idx` here is a row position needed for internal .iloc lookups

def _label_flip(idx, r, row):
    pp = float(r["pp"])
    return (f"#{_row_stable_id(row, idx)}  {str(row.get('artist', ''))[:20]}\n"
            f"FLIP: {pp:+.1f}pp \u2192 {'Danish' if pp > 0 else 'Norwegian'}")


def _label_internal_pair(idx, r, row):
    return f"#{_row_stable_id(row, idx)}  {str(row.get('artist', ''))[:20]}\nspread={r['spread']:.1f}pp"


# Single source of truth for label_fn + text-table format per mode.
MODE_FAMILY = {
    "flip_segment": "segment",
    "internal_pair": "internal",
}
LABEL_FNS = {"flip_segment": _label_flip, "internal_pair": _label_internal_pair}


def print_ranked_list(entries_df, attr_df, mode):
    family = MODE_FAMILY[mode]
    if family == "segment":
        print(f"\n  {'ID':<14} {'Artist':<30} {'Segment pp':<12} {'Direction':<10}\n  " + "-" * 70)
        for idx, r in entries_df.iterrows():
            artist, pp = str(attr_df.iloc[idx].get("artist", ""))[:28], r["pp"]
            print(f"  {str(_id_of(attr_df, idx)):<14} {artist:<30} {pp:+.1f}pp{'':<6} {'Danish' if pp > 0 else 'Norwegian':<10}")
    elif family == "internal":
        print(f"\n  {'ID':<14} {'Artist':<30} {'Spread':<8} {'Max cell (pp)':<18} {'Min cell (pp)':<18}\n  "
              + "-" * 92)
        for idx, r in entries_df.iterrows():
            artist = str(attr_df.iloc[idx].get("artist", ""))[:28]
            maxs = f"({int(r['max_r'])},{int(r['max_c'])}) {r['max_pp']:+.1f}"
            mins = f"({int(r['min_r'])},{int(r['min_c'])}) {r['min_pp']:+.1f}"
            print(f"  {str(_id_of(attr_df, idx)):<14} {artist:<30} {r['spread']:<8.1f} {maxs:<18} {mins:<18}")


# ── grid rendering (shared subplot layout for all grid views) ─────────────────

def _new_grid(n, wide):
    """Subplot grid sized for n thumbnails; returns (fig, axes) with exactly n
    live axes (empty trailing cells already hidden). wide=True gives the
    roomier sizing needed for the pp-labeled ablation overlay."""
    ncols = min(GRID_COLS, n)
    nrows = int(np.ceil(n / ncols))
    scale = 0.95 if wide else 0.85
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(FIG_WIDTH_PER_PANEL * ncols * scale,
                                       (FIG_HEIGHT + (0.3 if wide else 0)) * nrows * scale),
                              squeeze=False)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def render_ablation_grid_for_column(entries_df, attr_df, abl_col, title_prefix, label_fn, highlight_pair=False):
    """One window per model: thumbnail + full PRGn ablation overlay + pp per
    cell. If highlight_pair, also draws a box around the two specific
    segments (max pp in green / min pp in purple) that make up the ranked
    contrast pair (used only by mode 2)"""
    n = len(entries_df)
    if n == 0:
        print(f"  No usable data for {abl_col.replace('ablation_scores_', '')}.")
        return

    suffix = abl_col.replace("ablation_scores_", "")
    fig, axes = _new_grid(n, wide=True)
    norm, last_im = make_ablation_norm(), None

    for ax, (idx, entry_row) in zip(axes, entries_df.iterrows()):
        row = attr_df.iloc[idx]
        img = crop_to_grid(bytes_to_pil(row["image"]))
        W, H = img.size
        img_arr = np.array(img)
        cell_size, _, _ = grid_dims(W, H)

        display = get_pp_scores(row, suffix)   # +Danish/-Norwegian
        if display is not None:
            last_im = draw_pp_overlay(ax, img_arr, display, W, H, norm)

            if highlight_pair:
                for (r_cell, c_cell), edge_color in (((int(entry_row["max_r"]), int(entry_row["max_c"])), "#1a9850"),
                                                      ((int(entry_row["min_r"]), int(entry_row["min_c"])), "#762a83")):
                    rect = Rectangle((c_cell * cell_size, r_cell * cell_size), cell_size, cell_size,
                                      fill=False, edgecolor=edge_color, linewidth=2.5, zorder=3)
                    ax.add_patch(rect)
        else:
            ax.imshow(img_arr)

        ax.set_title(label_fn(idx, entry_row, row), fontsize=6.5, pad=3)
        ax.axis("off")

    fig.suptitle(f"{title_prefix} \u2014 {pretty_label(suffix)}", fontsize=10)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="horizontal", fraction=0.025, pad=0.02, aspect=45)
        cbar.set_label("\u2190 Norwegian        Danish \u2192   (pp shift in P(Danish))", fontsize=7)
        cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}pp"))
        cbar.ax.tick_params(labelsize=6)
    plt.figure(fig.number)
    plt.show(block=False)
    plt.pause(0.1)


def render_per_model_mode(mode, attr_df, active_cols, title_prefix, rank_fn, artist_filter=None, highlight_pair=False):
    """For each active column: rank_fn(attr_df, col), print table, open one
    grid window. artist_filter, if set, restricts every rank_fn to that one
    painter capped at TOP_N ). highlight_pair is passed straight through to 
    the grid renderer (used only by mode 2)."""
    label_fn = LABEL_FNS[mode]
    for col in active_cols:
        entries_df = rank_fn(attr_df, col, artist_filter=artist_filter)
        if entries_df.empty:
            print(f"  No usable data for {col.replace('ablation_scores_', '')}"
                  f"{f' (painter: {artist_filter})' if artist_filter else ''}.")
            continue
        print(f"\n  {title_prefix} \u2014 {pretty_label(col.replace('ablation_scores_', ''))}"
              f"{f'  [painter: {artist_filter}, {len(entries_df)} image(s)]' if artist_filter else ''}")
        print_ranked_list(entries_df, attr_df, mode)
        render_ablation_grid_for_column(entries_df, attr_df, col, title_prefix, label_fn, highlight_pair)


# ── column selection ──────────────────────────────────────────────────────────

def print_columns(ablation_cols, selected):
    print("\n  Available ablation columns:")
    for i, col in enumerate(ablation_cols):
        print(f"  [{'✓' if col in selected else ' '}] {i + 1:>2}. {col.replace('ablation_scores_', '')}")


def select_columns(ablation_cols, selected):
    print_columns(ablation_cols, selected)
    print("\n  Enter numbers to toggle (e.g. '1 3'), 'all', or 'none':")
    raw = input("  > ").strip().lower()
    if raw == "all":
        return set(ablation_cols)
    if raw == "none":
        return set()
    new_selected = set(selected)
    for part in raw.split():
        try:
            i = int(part) - 1
            if 0 <= i < len(ablation_cols):
                new_selected.symmetric_difference_update({ablation_cols[i]})
        except ValueError:
            pass
    return new_selected


# ── main loop ─────────────────────────────────────────────────────────────────

BROWSE_MODES = {"1": "flip_segment", "2": "internal_pair"}
BROWSE_LABELS = {
    "flip_segment": f"Top {TOP_N} strongest single-segment classification flips",
    "internal_pair": f"Top {TOP_N} strongest segment-pair contrast within an image",
}
RANK_FNS = {
    "flip_segment": top_flip_for_column,
    "internal_pair": top_internal_contrast_for_column,
}


def _id_of(attr_df, idx):
    """Stable identifier for a row position, falling back to the position
    itself only if neither 'id' nor 'filename' is present."""
    col = _stable_id_col(attr_df)
    return attr_df.iloc[idx].get(col, idx) if col else idx


def resolve_id(attr_df, id_text):
    """Looks up ID string against the stable identifier 
    column ('id' or 'filename'), returning the row's CURRENT position (for
    .iloc lookups elsewhere). Always resolved fresh rather than cached."""
    col = _stable_id_col(attr_df)
    if col is None:
        return None
    matches = attr_df.index[attr_df[col].astype(str) == str(id_text)].tolist()
    return matches[0] if matches else None


def resolve_id_or_position(attr_df, text):
    """Tries stable identifier first, then falls back to treating it
    as a raw row position."""
    pos = resolve_id(attr_df, text)
    if pos is not None:
        return pos, False
    if text.isdigit():
        pos = int(text)
        if 0 <= pos < len(attr_df):
            return pos, True
    return None, False


def _print_example_ids(attr_df, n=5):
    col = _stable_id_col(attr_df)
    if col is None:
        print(f"  [note] No 'id' column -- IDs are row positions (0-{len(attr_df) - 1}). "
              f"Example: {list(range(min(n, len(attr_df))))}")
        return
    examples = attr_df[col].astype(str).head(n).tolist()
    print(f"  Your actual IDs look like: {examples}")
    print("  (copy one of these -- 'ID' in every printed table is exactly what to type)")


def _print_artist_ids(attr_df, artist):
    """Prints the stable ID (see _stable_id_col) of every painting by this
    artist after setting the painter filter via 'p'."""
    positions = attr_df.index[attr_df["artist"] == artist].tolist()
    col = _stable_id_col(attr_df)
    ids = attr_df.loc[positions, col].astype(str).tolist() if col else positions
    print(f"  The following {len(ids)} ID(s) belong to {artist}: {', '.join(str(i) for i in ids)}")


def select_artist_filter(attr_df, current_filter):
    """Search-then-pick flow for the 'p' command: type part of a name, get
    matches, pick one by number. Enter with no text clears the filter."""
    artists = sorted(attr_df["artist"].dropna().unique().tolist())
    label = f" (current: {current_filter})" if current_filter else ""
    print(f"\n  Enter part of a painter's name to filter to{label}, or press Enter to clear:")
    raw = input("  > ").strip()
    if not raw:
        if current_filter:
            print("  Filter cleared.")
        return None

    matches = [a for a in artists if raw.lower() in str(a).lower()]
    if not matches:
        print(f"  No painter matching '{raw}'. Filter unchanged.")
        return current_filter
    if len(matches) == 1:
        print(f"  Filter set to: {matches[0]}")
        _print_artist_ids(attr_df, matches[0])
        return matches[0]

    print(f"\n  {len(matches)} matches:")
    for i, a in enumerate(matches):
        n_imgs = int((attr_df["artist"] == a).sum())
        print(f"  [{i + 1}] {a}  ({n_imgs} images)")
    sub = input("  Pick a number (or Enter to cancel): ").strip()
    if sub.isdigit() and 1 <= int(sub) <= len(matches):
        chosen = matches[int(sub) - 1]
        print(f"  Filter set to: {chosen}")
        _print_artist_ids(attr_df, chosen)
        return chosen
    print("  Cancelled -- filter unchanged.")
    return current_filter


def print_menu(artist_filter=None):
    filt = f"  |  painter: {artist_filter}" if artist_filter else ""
    print(f"\n  [i <id>] image  |  [b <1-2>] browse  |  [c] columns  |  "
          f"[p] painter{filt}  |  [q] quit")
    for k, v in BROWSE_MODES.items():
        print(f"    {k} = {BROWSE_LABELS[v]}")


def _fallback_note(attr_df, pos, typed_text):
    """Message shown when a plain-number fallback succeeds despite a real
    'id' column existing and tells the user this image's actual ID, so
    they can copy it for next time."""
    real_id = attr_df.iloc[pos].get("id")
    print(f"  [note] '{typed_text}' isn't a known ID -- treating it as a row position instead "
          f"(not guaranteed stable across sessions). This image's actual ID is '{real_id}' -- "
          f"use that next time.")


def main():
    attr_df = load_attribution_parquet(ATTRIBUTION_PARQUET)
    ablation_cols = [c for c in attr_df.columns if c.startswith("ablation_scores_")]
    if not ablation_cols:
        raise ValueError("No ablation_scores_* columns found in parquet(s).")
    print(f"[info] {len(attr_df)} images, {len(ablation_cols)} ablation column(s)")

    selected = set(ablation_cols)
    artist_filter = None
    plt.ion()
    print_menu(artist_filter)

    while True:
        try:
            raw_original = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        raw = raw_original.lower()   # command keywords are matched case-insensitively;

        if not raw:
            continue
        if raw in ("q", "quit", "exit"):
            print("[exit]")
            break

        active = [c for c in ablation_cols if c in selected]

        if raw == "c":
            selected = select_columns(ablation_cols, selected)
            print(f"  Active columns: {[c.replace('ablation_scores_', '') for c in sorted(selected, key=ablation_cols.index)]}")

        elif raw == "p":
            artist_filter = select_artist_filter(attr_df, artist_filter)

        elif raw.startswith("i "):
            parts = raw_original.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                print("  Usage: i <id>")
            elif not active:
                print("  No columns selected — use 'c' to select columns.")
            else:
                id_text = parts[1].strip()
                pos, was_fallback = resolve_id_or_position(attr_df, id_text)
                if pos is None:
                    print(f"  No image with ID '{id_text}'.")
                    _print_example_ids(attr_df)
                else:
                    if was_fallback and _stable_id_col(attr_df) is not None:
                        _fallback_note(attr_df, pos, id_text)
                    render_image(pos, attr_df, active, artist_filter)

        elif raw.startswith("b "):
            parts = raw.split()
            if len(parts) < 2 or parts[1] not in BROWSE_MODES:
                print("  Usage: b <1-2>")
                continue
            mode = BROWSE_MODES[parts[1]]

            if not active:
                print("  No columns selected — use 'c' to select columns.")
            elif artist_filter and not (attr_df["artist"] == artist_filter).any():
                print(f"  No images for painter '{artist_filter}'.")
            else:
                render_per_model_mode(mode, attr_df, active, BROWSE_LABELS[mode], RANK_FNS[mode],
                                      artist_filter, highlight_pair=(mode == "internal_pair"))

        else:
            print("  Unknown command.")
            print_menu(artist_filter)


if __name__ == "__main__":
    main()