#utils.py
"""
Auxiliary functions
"""

from scipy import linalg
import numpy as np
import matplotlib as mpl
import matplotsoccer as mps

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def add_ellips(ax, mean, covar, color=None, alpha=0.7):
    eigvals, eigvecs = linalg.eigh(covar)
    lengths = 2.0 * np.sqrt(2.0) * np.sqrt(eigvals)
    direction = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    width, height = max(lengths[0], 3), max(lengths[1], 3)

    ell = mpl.patches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,  # or edgecolor=color
        alpha=alpha
    )
    ax.add_patch(ell)
    return ax


def add_arrow(ax, x, y, dx, dy,
              arrowsize=2.5,
              linewidth=2.0,
              threshold=1.8,
              alpha=1.0,
              fc='grey',
              ec='grey'):
    """
    Draw an arrow only if its dx or dy exceed the threshold,
    with both facecolor and edgecolor set to grey by default.
    """
    if np.sqrt(dx ** 2 + dy ** 2) > threshold:
        return ax.arrow(
            x, y, dx, dy,
            head_width=arrowsize,
            head_length=arrowsize,
            linewidth=linewidth,
            fc=fc,
            ec=ec,
            length_includes_head=True,
            alpha=alpha,
            zorder=3,
        )

def consolidate(actions):
    # actions.fillna(0, inplace=True)

    # Consolidate corner_short and corner_crossed
    corner_idx = actions.type_name.str.contains("corner")
    actions["type_name"] = actions["type_name"].mask(corner_idx, "corner")

    # Consolidate freekick_short, freekick_crossed, and shot_freekick
    freekick_idx = actions.type_name.str.contains("freekick")
    actions["type_name"] = actions["type_name"].mask(freekick_idx, "freekick")

    # Consolidate keeper_claim, keeper_punch, keeper_save, keeper_pick_up
    keeper_idx = actions.type_name.str.contains("keeper")
    actions["type_name"] = actions["type_name"].mask(keeper_idx, "keeper_action")

    actions["start_x"] = actions["start_x"].mask(actions.type_name == "shot_penalty", 94.5)
    actions["start_y"] = actions["start_y"].mask(actions.type_name == "shot_penalty", 34)

    return actions


def add_noise(actions):
    # Start locations
    start_list = ["cross", "shot", "dribble", "pass", "keeper_action", "clearance", "goalkick"]
    mask = actions["type_name"].isin(start_list)
    noise = np.random.normal(0, 0.5, size=actions.loc[mask, ["start_x", "start_y"]].shape)
    actions.loc[mask, ["start_x", "start_y"]] += noise

    # End locations
    end_list = ["cross", "shot", "dribble", "pass", "keeper_action", "throw_in", "corner", "freekick", "shot_penalty"]
    mask = actions["type_name"].isin(end_list)
    noise = np.random.normal(0, 0.5, size=actions.loc[mask, ["end_x", "end_y"]].shape)
    actions.loc[mask, ["end_x", "end_y"]] += noise

    return actions

def remove_outliers(actions, verbose=False):
    X = actions[["start_x", "start_y", "end_x", "end_y"]].to_numpy(dtype=float)
    inliers = LocalOutlierFactor(contamination="auto").fit_predict(X)
    if verbose:
        print(f"Remove {(inliers == -1).sum()} out of {X.shape[0]} datapoints.")
    return actions[inliers == 1]