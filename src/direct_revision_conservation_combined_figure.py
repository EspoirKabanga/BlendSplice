#!/usr/bin/env python3
"""Conservation fidelity heatmaps: (synthetic − Real) deltas.

Four heatmaps: 2 splice-site types (donor/acceptor) x 2 quantities
(Blend − Real, No-Blend − Real). Each heatmap is 18 conditions
(species x model x context) on y, position relative to the splice site on x.

Two target page widths, selected with --layout:
    wide  (double column): 2 rows (seq) x 2 cols (quantity)  -> wide & short
    tall  (single column): 2 rows (quantity) x 2 cols (seq)  -> narrow & tall

Run on real data:
    python conservation_delta_heatmap.py --layout wide --out fig.png
Preview both layouts on synthetic data (no real CSVs needed):
    python conservation_delta_heatmap.py --demo
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(REPO, "results", "direct_evaluation")

MODELS = ["GAN", "VAE", "DIFFUSION"]
MODEL_LABEL = {"GAN": "GAN", "VAE": "VAE", "DIFFUSION": "Diffusion"}
SPECIES = ["arabidopsis", "human", "danio"]
CONTEXTS = ["402", "2002"]
SEQ_TYPES = ["donor", "acceptor"]

SPECIES_LABEL = {"arabidopsis": "Arabidopsis", "human": "Human", "danio": "Danio"}
SITE_LABEL = {"donor": "Donor", "acceptor": "Acceptor"}

# CSV column -> display label (matches the line-grid script)
CSV_COLUMNS = {"Real": "Real", "Blend": "Blend", "No_Blend": "No-Blend"}

MOTIF_POS = {"402": 200, "2002": 1000}
REL_WINDOW = (-50, 50)
REL_TICKS = [-50, -25, 0, 25, 50]
REL_POS = np.arange(REL_WINDOW[0], REL_WINDOW[1] + 1)

DELTA_CMAP = "RdBu_r"
SPECIES_X_OFFSET_CM = 0.1
SPECIES_X_OFFSET_PT = SPECIES_X_OFFSET_CM / 2.54 * 72.0
SPECIES_LINE_EXTEND_CM = 2.3
DONOR_ACCEPTOR_GAP_CM = 0.2
QUANTITY_ROW_GAP_REDUCE_CM = 0.3

# Per-layout sizing. Offsets are axes-fraction x positions on the leftmost
# panel column (negative = left of the panel), tuned to each figure width.
LAYOUTS = {
    "wide": {  # double column: wide & short
        "figsize": (7.16, 4.5),
        "panel_wspace": 0.10, "panel_hspace": 0.24,
        "cond_fs": 6.5, "species_fs": 9.0, "group_fs": 12.0,
        "title_fs": 9.5, "cbar_fs": 8.5,
        "species_x": -0.58, "group_x": -0.84,
        "left": 0.16, "right": 0.99, "top": 0.90,
        "cbar_frac": 0.06, "cbar_gap": 0.20, "single_xlabel": False,
    },
    "tall": {  # single column: narrow & tall
        "figsize": (3.6, 6.8),
        "panel_wspace": 0.10, "panel_hspace": 0.16,
        "cond_fs": 6.0, "species_fs": 8.0, "group_fs": 10.0,
        "title_fs": 9.0, "cbar_fs": 7.5,
        "species_x": -0.92, "group_x": -1.12,
        "left": 0.27, "right": 0.90, "top": 0.94,
        "cbar_frac": 0.045, "cbar_gap": 0.14, "single_xlabel": True,
    },
}


def _panel_area_size_in(cfg, is_wide):
    w = cfg["figsize"][0] * (cfg["right"] - cfg["left"])
    h = cfg["figsize"][1] * (cfg["top"] - (0.085 if is_wide else 0.07))
    if is_wide:
        h /= 1.0 + cfg["cbar_frac"] + cfg["cbar_gap"]
    else:
        w /= 1.0 + cfg["cbar_frac"] + cfg["cbar_gap"]
    return w, h


def _gridspec_gap_adjust(avail_in, n_panels, base_frac, delta_gap_cm):
    """Adjust GridSpec wspace/hspace by delta_gap_cm (+ wider, − narrower)."""
    if n_panels < 2 or delta_gap_cm == 0:
        return base_frac
    delta_in = delta_gap_cm / 2.54
    cur_gap_in = base_frac * avail_in / (n_panels + base_frac)
    total_gap_in = max(0.01, cur_gap_in + delta_in)
    return n_panels * total_gap_in / (avail_in - total_gap_in)


def load_conservation_csv(context, model, species, seq_type):
    path = os.path.join(
        OUTPUT_ROOT, context, model, f"{species}_{seq_type}_conservation.csv"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _relative_series(df, context):
    center = MOTIF_POS[context]
    out = df.copy()
    out.index = out.index - center
    lo, hi = REL_WINDOW
    return out[(out.index >= lo) & (out.index <= hi)]


def build_matrices(seq_type):
    blend_d_rows, nb_d_rows = [], []
    row_labels, species_block, model_block, context_block = [], [], [], []
    for species in SPECIES:
        for model in MODELS:
            for context in CONTEXTS:
                df = _relative_series(
                    load_conservation_csv(context, model, species, seq_type)
                    .set_index("Position"),
                    context,
                ).reindex(REL_POS)
                real = df["Real"].to_numpy(dtype=float)
                blend = df["Blend"].to_numpy(dtype=float)
                noblend = df["No_Blend"].to_numpy(dtype=float)
                blend_d_rows.append(blend - real)
                nb_d_rows.append(noblend - real)
                row_labels.append(f"{MODEL_LABEL[model]} {context}bp")
                species_block.append(SPECIES_LABEL[species])
                model_block.append(model)
                context_block.append(context)
    return {
        "blend_delta": np.asarray(blend_d_rows),
        "noblend_delta": np.asarray(nb_d_rows),
        "row_labels": row_labels,
        "species_block": species_block,
        "model_block": model_block,
        "context_block": context_block,
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _species_boundaries(species_block):
    return [i + 0.5 for i in range(len(species_block) - 1)
            if species_block[i] != species_block[i + 1]]


def _draw_row_separators(ax, species_block, model_block, context_block,
                         draw_species_lines=True):
    for i in range(len(species_block) - 1):
        y = i + 0.5
        if species_block[i] != species_block[i + 1]:
            if draw_species_lines:
                ax.axhline(y, color="0.15", lw=1.0)
        elif model_block[i] != model_block[i + 1]:
            ax.axhline(y, color="0.50", lw=0.55)
        elif context_block[i] != context_block[i + 1]:
            ax.axhline(y, color="0.62", lw=0.42)


def _draw_extended_species_lines(fig, ax, species_block, extend_cm):
    """Species dividers on the left column, extended left into the label margin."""
    fig.canvas.draw()
    bbox = ax.get_position()
    extend_frac = (extend_cm / 2.54) / fig.get_figwidth()
    x_start = bbox.x0 - extend_frac
    x_end = bbox.x1
    trans = blended_transform_factory(fig.transFigure, ax.transData)
    for y in _species_boundaries(species_block):
        line = Line2D(
            [x_start, x_end], [y, y],
            transform=trans, color="0.15", lw=1.0,
            clip_on=False, zorder=10, solid_capstyle="butt",
        )
        fig.add_artist(line)


def _draw_heatmap(ax, mat, n_rows, cmap, vmin, vmax):
    extent = [REL_POS[0] - 0.5, REL_POS[-1] + 0.5, n_rows - 0.5, -0.5]
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, interpolation="nearest")
    ax.axvline(0, color="0.25", lw=0.7, ls=":")
    ax.set_xticks(REL_TICKS)
    return im


def create_delta_figure(out_path, layout="wide", dpi=300):
    cfg = LAYOUTS[layout]
    is_wide = (layout == "wide")

    data = {seq: build_matrices(seq) for seq in SEQ_TYPES}
    n_cond = data["donor"]["blend_delta"].shape[0]
    row_labels = data["donor"]["row_labels"]
    species_block = data["donor"]["species_block"]
    model_block = data["donor"]["model_block"]
    context_block = data["donor"]["context_block"]
    species_boundaries = _species_boundaries(species_block)
    edges = [-0.5] + species_boundaries + [n_cond - 0.5]
    species_order = list(dict.fromkeys(species_block))

    all_delta = np.concatenate([
        np.abs(data[s][k]).ravel()
        for s in SEQ_TYPES for k in ("blend_delta", "noblend_delta")
    ])
    dmax = max(round(float(np.nanpercentile(all_delta, 99)), 2), 0.05)

    # (key, title, cmap, vmin, vmax)
    quantities = [
        ("blend_delta", "Blend − Real", DELTA_CMAP, -dmax, dmax),
        ("noblend_delta", "No-Blend − Real", DELTA_CMAP, -dmax, dmax),
    ]
    seqs = list(SEQ_TYPES)

    # panel grid dimensions
    if is_wide:
        nP_rows, nP_cols = len(seqs), len(quantities)        # 2 x 2
    else:
        nP_rows, nP_cols = len(quantities), len(seqs)         # 2 x 2

    fig = plt.figure(figsize=cfg["figsize"])

    panel_w, panel_h = _panel_area_size_in(cfg, is_wide)
    if is_wide:
        panel_wspace = cfg["panel_wspace"]
        panel_hspace = _gridspec_gap_adjust(
            panel_h, nP_rows, cfg["panel_hspace"], DONOR_ACCEPTOR_GAP_CM,
        )
    else:
        panel_wspace = _gridspec_gap_adjust(
            panel_w, nP_cols, cfg["panel_wspace"], DONOR_ACCEPTOR_GAP_CM,
        )
        panel_hspace = _gridspec_gap_adjust(
            panel_h, nP_rows, cfg["panel_hspace"], -QUANTITY_ROW_GAP_REDUCE_CM,
        )

    # outer layout: panels + a colorbar strip (bottom for wide, right for tall)
    if is_wide:
        outer = gridspec.GridSpec(
            2, 1, height_ratios=[1.0, cfg["cbar_frac"]], hspace=cfg["cbar_gap"],
            left=cfg["left"], right=cfg["right"], top=cfg["top"], bottom=0.085,
        )
        panel_gs = gridspec.GridSpecFromSubplotSpec(
            nP_rows, nP_cols, subplot_spec=outer[0],
            wspace=panel_wspace, hspace=panel_hspace)
        cbar_gs = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=outer[1])
        cax_delta = fig.add_subplot(cbar_gs[0, 0])
        cbar_orient = "horizontal"
    else:
        outer = gridspec.GridSpec(
            1, 2, width_ratios=[1.0, cfg["cbar_frac"]], wspace=cfg["cbar_gap"],
            left=cfg["left"], right=cfg["right"], top=cfg["top"], bottom=0.07,
        )
        panel_gs = gridspec.GridSpecFromSubplotSpec(
            nP_rows, nP_cols, subplot_spec=outer[0],
            wspace=panel_wspace, hspace=panel_hspace)
        cbar_gs = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=outer[1])
        cax_delta = fig.add_subplot(cbar_gs[0, 0])
        cbar_orient = "vertical"

    delta_im = None
    panel_axes = {}

    for si, seq in enumerate(seqs):
        for qi, (key, title, cmap, vmin, vmax) in enumerate(quantities):
            pr, pc = (si, qi) if is_wide else (qi, si)
            ax = fig.add_subplot(panel_gs[pr, pc])
            panel_axes[(pr, pc)] = ax

            im = _draw_heatmap(ax, data[seq][key], n_cond, cmap, vmin, vmax)
            delta_im = im

            _draw_row_separators(
                ax, species_block, model_block, context_block,
                draw_species_lines=(pc != 0),
            )

            # column title (top panel row only): quantity in wide, seq in tall
            if pr == 0:
                ctitle = title if is_wide else SITE_LABEL[seq]
                ctitle_fs = cfg["title_fs"] if is_wide else cfg["species_fs"]
                ax.set_title(ctitle, fontsize=ctitle_fs,
                             fontweight="bold", pad=6)

            # x tick values on every row; axis title only on bottom row
            ax.tick_params(axis="x", labelsize=cfg["cond_fs"] + 1, labelbottom=True)
            if pr == nP_rows - 1 and not cfg["single_xlabel"]:
                ax.set_xlabel("Distance from splice site (bp)",
                              fontsize=cfg["title_fs"] - 1, labelpad=3)

            # y tick labels + group/species headers on leftmost panel col only
            ax.set_yticks(range(n_cond))
            if pc == 0:
                ax.set_yticklabels(row_labels, fontsize=cfg["cond_fs"])
                for k, sp in enumerate(species_order):
                    ymid = (edges[k] + edges[k + 1]) / 2
                    ax.annotate(
                        sp, xy=(cfg["species_x"], ymid),
                        xycoords=("axes fraction", "data"),
                        xytext=(SPECIES_X_OFFSET_PT, 0),
                        textcoords="offset points",
                        ha="center", va="center", rotation=90,
                        fontsize=cfg["species_fs"], fontweight="bold",
                        annotation_clip=False,
                    )
                # group header: seq in wide, quantity in tall
                ghead = SITE_LABEL[seq] if is_wide else title
                ghead_fs = cfg["species_fs"] if is_wide else cfg["group_fs"]
                ax.annotate(ghead, xy=(cfg["group_x"], (n_cond - 1) / 2),
                            xycoords=("axes fraction", "data"),
                            ha="center", va="center", rotation=90,
                            fontsize=ghead_fs, fontweight="bold",
                            annotation_clip=False)
            else:
                ax.tick_params(axis="y", labelleft=False)

    cb_delta = fig.colorbar(delta_im, cax=cax_delta, orientation=cbar_orient)
    cb_delta.set_label("Δ (synthetic − real)", fontsize=cfg["cbar_fs"])
    cb_delta.ax.tick_params(labelsize=cfg["cbar_fs"] - 0.5)

    for r in range(nP_rows):
        _draw_extended_species_lines(
            fig, panel_axes[(r, 0)], species_block, SPECIES_LINE_EXTEND_CM,
        )

    if cfg["single_xlabel"]:
        fig.canvas.draw()
        bottoms = [panel_axes[(nP_rows - 1, c)].get_position() for c in range(nP_cols)]
        cx = sum((b.x0 + b.x1) / 2 for b in bottoms) / len(bottoms)
        y0 = min(b.y0 for b in bottoms)
        fig.text(cx, y0 - 0.035, "Distance from splice site (bp)",
                 ha="center", va="top", fontsize=cfg["title_fs"] - 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out_path}  [{layout}]")


# --------------------------------------------------------------------------- #
# Demo harness: writes synthetic CSVs into the real directory structure and
# runs the production code path against them.
# --------------------------------------------------------------------------- #
def _write_demo_csvs(root):
    rng = np.random.default_rng(7)

    def real_profile(positions, center, sharp, peak=0.93, base=0.30):
        rel = positions - center
        bump = (peak - base) * np.exp(-(rel ** 2) / (2 * sharp ** 2))
        bump += 0.04 * np.exp(-((rel - 1) ** 2) / 2.0)
        return base + bump

    for context in CONTEXTS:
        length = {"402": 402, "2002": 2002}[context]
        center = MOTIF_POS[context]
        positions = np.arange(length)
        for species in SPECIES:
            sharp = {"arabidopsis": 7.0, "human": 9.0, "danio": 8.0}[species]
            for model in MODELS:
                severity = {"GAN": 0.45, "VAE": 0.25, "DIFFUSION": 0.15}[model]
                for seq_type in SEQ_TYPES:
                    real = real_profile(positions, center, sharp)
                    blend = np.clip(real + rng.normal(0, 0.015, length), 0, 1)
                    rel = positions - center
                    broaden = (real - 0.30) * (-severity) * np.exp(-(rel ** 2) / (2 * 14 ** 2))
                    flank = 0.10 * severity * np.exp(-((np.abs(rel) - 35) ** 2) / (2 * 8 ** 2))
                    noblend = np.clip(real + broaden + flank + rng.normal(0, 0.02, length), 0, 1)
                    d = os.path.join(root, context, model)
                    os.makedirs(d, exist_ok=True)
                    pd.DataFrame({
                        "Position": positions, "Real": real,
                        "Blend": blend, "No_Blend": noblend,
                    }).to_csv(os.path.join(d, f"{species}_{seq_type}_conservation.csv"),
                              index=False)


def main():
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description="Conservation fidelity heatmaps")
    parser.add_argument("--out",
                        default=os.path.join(OUTPUT_ROOT,
                                             "combined_conservation_delta_heatmap.png"))
    parser.add_argument("--layout", choices=["wide", "tall"], default="wide")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 9})

    if args.demo:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="cons_demo_")
        OUTPUT_ROOT = tmp
        _write_demo_csvs(tmp)
        create_delta_figure("/home/claude/delta_wide.png", layout="wide", dpi=args.dpi)
        create_delta_figure("/home/claude/delta_tall.png", layout="tall", dpi=args.dpi)
    else:
        create_delta_figure(args.out, layout=args.layout, dpi=args.dpi)


if __name__ == "__main__":
    main()
