import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from PIL import Image
import io


# ── Config ────────────────────────────────────────────────────────────────────

SHOW_FILENAMES = False

PREDICTIONS_PARQUET = "data\trees-with-predictions.parquet" #choose actual path / filename

IMAGE_COL = "full_image_with_box"        # switch to "image" for plain masked crop
SOURCE_FILENAME_COL = "source_filename"  # the original painting's filename, for captions
JOIN_KEY_COL = "instance_filename"       # the per-tree unique key
CLASS_COL = "class"                      # the class-label column in this file

TOP_N = 20  # how many artists/paintings to rank and show per menu action

CURRENT_MODEL = None  # is asked from user in the beginning if None

plt.rcParams["savefig.dpi"] = 300 # image quality when saving
plt.rcParams["savefig.bbox"] = "tight"

# ── Image utilities ────────────────────────────────────────────────────────────

def to_rgb(image):
    """Convert various image formats to RGB PIL Image."""
    if image is None:
        return None
 
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
 
    if isinstance(image, np.ndarray):
        img = image
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        if img.shape[-1] == 3 and img.dtype == np.uint8:
            img = img[..., ::-1]
        return Image.fromarray(img)
 
    return Image.open(io.BytesIO(image)).convert("RGB")
 
 
def plot_grid(df, n=None):
    """Plot a grid of images with metadata. Grid is always 4 columns; row
    count is derived from n and defaults to TOP_N."""
    n = TOP_N if n is None else n
    n_cols = 4
    n_rows = max(1, -(-n // n_cols))  # ceiling division
    df = df.head(n).reset_index(drop=True)
 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
 
    for i, ax in enumerate(axes):
        if i >= len(df):
            ax.axis("off")
            continue
 
        row = df.iloc[i]
        img = to_rgb(row[IMAGE_COL])
        ax.imshow(img)
        ax.axis("off")
 
        caption = (
            f"{row['artist']}\n"
            f"True: {row[CLASS_COL]} | Pred: {row['predicted_class']}\n"
            f"P(wrong)={row['p_wrong']:.3f} | margin={row['margin']:.3f}"
        )
 
        if SHOW_FILENAMES:
            caption += f"\n{row[SOURCE_FILENAME_COL]}"
 
        ax.text(
            0.5, -0.08,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            wrap=True
        )
 
    plt.show()

 
# ── Artist mode ────────────────────────────────────────────────────────────────────
 
def get_top_artists(df, mode="misclassified", top_n=20):
    """Get top N artists based on mode."""
    if mode == "misclassified":
        grp = df.groupby("artist").agg(
            total=("misclassified", "size"),
            misclassified=("misclassified", "sum")
        )
        grp["mis_rate"] = grp["misclassified"] / grp["total"]
        return grp["mis_rate"].sort_values(ascending=False).head(top_n)
 
    elif mode == "confident":
        grp = df.groupby("artist").agg(
            margin_mean=("margin", "mean"),
            total=("misclassified", "size"),
            misclassified=("misclassified", "sum")
        )
        grp["mis_rate"] = grp["misclassified"] / grp["total"]
        grp["confident_score"] = grp["margin_mean"] / (grp["mis_rate"] + 1e-6)
        return grp["confident_score"].sort_values(ascending=False).head(top_n)
 
    elif mode == "ambiguous":
        grp = df.groupby("artist")["abs_margin"].mean()
        return grp.sort_values(ascending=True).head(top_n)
 
    return pd.Series()
 
def display_top_artists(df, mode, top_n=20):
    """Display top artists with statistics."""
    top_artists = get_top_artists(df, mode, top_n=top_n)
    print(f"\nTop {top_n} painters:")
 
    for i, (artist, val) in enumerate(top_artists.items(), start=1):
        artist_df = df[df["artist"] == artist]
 
        if len(artist_df) == 0:
            continue
 
        samples = len(artist_df)
        cls = artist_df[CLASS_COL].iloc[0]
        acc = (artist_df["predicted_class"] == artist_df[CLASS_COL]).mean()
        mean_pwrong = artist_df["p_wrong"].mean()
        misclass = artist_df["misclassified"].sum()
 
        if mode == "misclassified":
            mis_rate = misclass / samples
            print(
                f"{i:2d}. {artist:35s} | n={samples:3d} | cls={cls:9s} | "
                f"mis_rate={mis_rate:.3f} | mis={misclass:3d} | "
                f"acc={acc:.3f} | p_wrong={mean_pwrong:.3f}"
            )
 
        elif mode == "ambiguous":
            mean_margin = artist_df["abs_margin"].mean()
            mean_margin_mis = artist_df[artist_df["misclassified"]]["abs_margin"].mean()
            print(
                f"{i:2d}. {artist:35s} | n={samples:3d} | cls={cls:9s} | "
                f"abs_margin={mean_margin:.3f} | abs_margin_mis={mean_margin_mis:.3f} | "
                f"acc={acc:.3f} | p_wrong={mean_pwrong:.3f}"
            )
 
        else:  # confident
            margin_mean = artist_df["margin"].mean()
            mis_rate = artist_df["misclassified"].mean()
            print(
                f"{i:2d}. {artist:35s} | n={samples:3d} | cls={cls:9s} | "
                f"mis_rate={mis_rate:.3f} | margin={margin_mean:.3f} | "
                f"acc={acc:.3f} | p_wrong={mean_pwrong:.3f}"
            )
 
    return top_artists
 
def load_lightweight_predictions(path):
    """Reads only the flat columns actually needed for the viewer.
    """
    schema_names = pq.ParquetFile(path).schema_arrow.names
 
    predicted_cols = [c for c in schema_names if c.startswith("predicted_class__")]
    model_names = [c[len("predicted_class__"):] for c in predicted_cols]
 
    suffix_bases = ["predicted_class", "p_true", "p_wrong", "margin", "misclassified", "fold_id"]
    suffixed_cols = [
        f"{base}__{m}" for m in model_names for base in suffix_bases
        if f"{base}__{m}" in schema_names
    ]
 
    if CLASS_COL not in schema_names:
        raise KeyError(
            f"CLASS_COL='{CLASS_COL}' not found in {path}. "
            f"Available columns: {schema_names}"
        )
 
    base_cols = [JOIN_KEY_COL, SOURCE_FILENAME_COL, "artist", CLASS_COL, IMAGE_COL]
    base_cols = [c for c in base_cols if c in schema_names]
 
    missing_base = {JOIN_KEY_COL, SOURCE_FILENAME_COL, "artist", IMAGE_COL} - set(base_cols)
    if missing_base:
        print(f"[warn] Base column(s) not found in file, continuing without them: {missing_base}")
 
    df = pd.read_parquet(path, columns=base_cols + suffixed_cols)
    return df
 
 
def detect_available_models(df):
    """Find every classifier name present as a predicted_class__<name> column."""
    names = []
    for col in df.columns:
        if col.startswith("predicted_class__"):
            names.append(col[len("predicted_class__"):])
    return sorted(names)
 
 
def choose_model(df, current=None):
    """Prompt the user to pick which classifier's columns to explore."""
    available = detect_available_models(df)
    if not available:
        raise ValueError("No 'predicted_class__<model>' columns found in the predictions file.")
 
    print("\nAvailable classifiers:")
    for i, name in enumerate(available, start=1):
        n_present = df[f"predicted_class__{name}"].notna().sum()
        marker = "  <- current" if name == current else ""
        print(f"[{i}] {name} (n={n_present}){marker}")
 
    choice = input("Choose a classifier: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(available):
            return available[idx]
    except ValueError:
        pass
 
    print("Invalid choice, keeping current selection." if current else
          f"Invalid choice, defaulting to '{available[0]}'.")
    return current or available[0]
 
 
def apply_model_columns(df, model_name):
    """Build the working predicted_class/p_true/p_wrong/margin/misclassified
    columns from the chosen classifier's suffixed columns."""
    df = df.copy()
    for base_col in ["predicted_class", "p_true", "p_wrong", "margin", "misclassified"]:
        suffixed = f"{base_col}__{model_name}"
        if suffixed in df.columns:
            df[base_col] = df[suffixed]
 
    before = len(df)
    df = df[df["predicted_class"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[info] {dropped} row(s) not scored by '{model_name}' -- excluded from this view.")
 
    df["misclassified"] = df["misclassified"].astype(bool)
    return df
 
 
def choose_class(df):
    """Prompt the user to filter to one class or keep both. Used for 'most confident' 
    since danish vs norwegian can  have very different confidence distributions and 
    mixing them together makes the top list less useful."""
    classes = sorted(df[CLASS_COL].unique())
    print("\nFilter by class:")
    for i, cls in enumerate(classes, start=1):
        print(f"[{i}] {cls}")
    print(f"[{len(classes) + 1}] both (no filter)")
 
    choice = input("Your choice: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(classes):
            selected = classes[idx]
            print(f"Filtering to: {selected}")
            return df[df[CLASS_COL] == selected].reset_index(drop=True)
    except ValueError:
        pass
 
    print("Showing both classes.")
    return df
 
 
def select_artists(top_artists):
    """Interactive artist selection."""
    print("\nSelect artists by number (comma-separated, e.g. '1,3,5') or 'all'")
    selection = input("Your choice: ").strip()
 
    if selection.lower() == "all":
        return list(top_artists.index)
 
    chosen = []
    for num in selection.split(","):
        try:
            idx = int(num.strip()) - 1
            if 0 <= idx < len(top_artists):
                chosen.append(top_artists.index[idx])
        except:
            continue
 
    return chosen
 
 
def load_model_view(df_all, model_name):
    """Build the working per-model dataframe: apply that classifier's
    columns, then derive abs_margin. Shared by the initial load and the
    'c' (change classifier) menu action so the two can't drift apart."""
    df = apply_model_columns(df_all, model_name)
    df["abs_margin"] = df["margin"].abs()
    return df
 
 
def run_artist_mode(df, mode, sort_col, ascending, only_misclassified=False, filter_class=False):
    """Shared implementation for the three 'top painters' menu modes
    (misclassified / confident / ambiguous). They only differ in which
    ranking mode to use, how to sort each artist's rows for display, and
    whether to pre-filter to misclassified rows or a chosen class."""
    working_df = choose_class(df) if filter_class else df
    top_artists = display_top_artists(working_df, mode, top_n=TOP_N)
    chosen = select_artists(top_artists)
    for i, artist in enumerate(chosen):
        sub = working_df[working_df["artist"] == artist]
        if only_misclassified:
            sub = sub[sub["misclassified"]]
        sub = sub.sort_values(sort_col, ascending=ascending)
        plot_grid(sub)
        if i < len(chosen) - 1:
            input("Press Enter for next artist...")
 
 
def run_single_mode(df, sort_col, ascending, only_misclassified=False, filter_class=False):
    """Shared implementation for the three 'single painting mode' options."""
    working_df = choose_class(df) if filter_class else df
    if only_misclassified:
        working_df = working_df[working_df["misclassified"]]
    sub = working_df.sort_values(sort_col, ascending=ascending)
    plot_grid(sub)
 
# ── Main Menu ──────────────────────────────────────────────────────────────
 
def menu():
    """Main interactive menu."""
    global CURRENT_MODEL  # reassigned by the 'c' toggle option below
 
    print("\nLoading predictions + images...")
    df_all = load_lightweight_predictions(PREDICTIONS_PARQUET)
 
    n_missing_img = df_all[IMAGE_COL].isna().sum()
    if n_missing_img:
        print(f"⚠ {n_missing_img} rows have no '{IMAGE_COL}' and will be dropped")
        df_all = df_all[df_all[IMAGE_COL].notna()].reset_index(drop=True)
 
    CURRENT_MODEL = choose_model(df_all, current=None)
    df = load_model_view(df_all, CURRENT_MODEL)
 
    print(f"Loaded {len(df)} predictions for classifier: {CURRENT_MODEL}")
 
    while True:
        print("\n" + "=" * 80)
        print("INTERACTIVE GRID EXPLORER")
        print(f"Current classifier: {CURRENT_MODEL}")
        print("=" * 80)
        print(f"[1] Top {TOP_N} misclassified painters")
        print(f"[2] Top {TOP_N} most confident painters")
        print(f"[3] Top {TOP_N} most ambiguous painters")
        print(f"[4] Single painting mode")
        print("[c] Change classifier")
        print("[q] Quit")
 
        choice = input("\nChoice: ").strip().lower()
 
        if choice == "1":
            run_artist_mode(df, "misclassified", sort_col="p_wrong", ascending=False,
                             only_misclassified=True)
 
        elif choice == "2":
            run_artist_mode(df, "confident", sort_col="margin", ascending=False,
                             filter_class=True)
 
        elif choice == "3":
            run_artist_mode(df, "ambiguous", sort_col="abs_margin", ascending=True)
 
        elif choice == "4":
            print("\nSingle painting mode:")
            print(f"[1] Top {TOP_N} misclassified paintings")
            print(f"[2] Top {TOP_N} ambiguous paintings")
            print(f"[3] Top {TOP_N} confident paintings")
            print("[q] Back")
 
            sub_choice = input("Choice: ").strip().lower()
 
            if sub_choice == "1":
                run_single_mode(df, sort_col="p_wrong", ascending=False, only_misclassified=True)
            elif sub_choice == "2":
                run_single_mode(df, sort_col="abs_margin", ascending=True)
            elif sub_choice == "3":
                run_single_mode(df, sort_col="margin", ascending=False, filter_class=True)
            elif sub_choice == "q":
                continue
            else:
                print("Invalid choice!")
 
        elif choice == "c":
            new_model = choose_model(df_all, current=CURRENT_MODEL)
            if new_model != CURRENT_MODEL:
                CURRENT_MODEL = new_model
                df = load_model_view(df_all, CURRENT_MODEL)
                print(f"Switched to classifier: {CURRENT_MODEL} ({len(df)} trees)")
 
        elif choice == "q":
            break
 
        else:
            print("Invalid choice!")
 
if __name__ == "__main__":
    menu()