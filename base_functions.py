import itertools
import numpy as np
import pandas as pd
from dtw import dtw
import matplotlib.pyplot as plt
import math
import re

from ipywidgets import interact, widgets, fixed

def make_sections(dfs, section_names, params, skips=None):
    """
    Extract multivariate arrays from Excel sheets.

    Returns
    -------
    sections : dict
        {section_name: array of shape (n_points, n_params)}
    """
    if skips is None:
        skips = {}

    sections = {}

    for name in section_names:
        skip = skips.get(name, 0)

        arr = (
            dfs[name][params]
            .iloc[skip:]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        sections[name] = arr

    return sections

def make_weight_vector(params, weights=None):
    """
    Convert either a dict or a list/array of weights to a numpy vector.
    """
    if weights is None:
        return np.ones(len(params), dtype=float)

    if isinstance(weights, dict):
        return np.array([weights.get(p, 1.0) for p in params], dtype=float)

    weights = np.asarray(weights, dtype=float)

    if len(weights) != len(params):
        raise ValueError("weights must have the same length as params.")

    return weights

def make_circular_mask(params, circular_params=("Dgeo",)):
    """
    Mark circular variables, e.g. declination in degrees.
    """
    circular_params = set(circular_params)
    return np.array([p in circular_params for p in params], dtype=bool)

def compute_global_scales(
    sections,
    params=None,
    method="mad",
    scale_overrides=None,
    min_scale=1e-12,
):
    """
    Estimate one scale per parameter across all sections.

    method="mad" is robust.
    method="std" uses standard deviation.

    scale_overrides can be used for known measurement precision, e.g.
    {"Dgeo": 15.0, "Igeo": 10.0}
    """
    all_values = np.vstack(list(sections.values()))

    if method == "mad":
        center = np.nanmedian(all_values, axis=0)
        mad = np.nanmedian(np.abs(all_values - center), axis=0)
        scales = 1.4826 * mad
    elif method == "std":
        scales = np.nanstd(all_values, axis=0)
    else:
        raise ValueError("method must be 'mad' or 'std'.")

    scales = np.asarray(scales, dtype=float)
    scales[~np.isfinite(scales)] = 1.0
    scales[scales < min_scale] = 1.0

    if scale_overrides is not None:
        if params is None:
            raise ValueError("params must be provided when using scale_overrides.")

        for param_name, scale_value in scale_overrides.items():
            if param_name not in params:
                raise ValueError(f"{param_name!r} is not in params.")
            scales[params.index(param_name)] = float(scale_value)

    return scales

def circular_difference_degrees(a, b):
    """
    Smallest signed angular difference a - b for degrees.
    Result is in [-180, 180].
    """
    return (a - b + 180.0) % 360.0 - 180.0

def weighted_nan_dist(
    a,
    b,
    weights=None,
    scales=None,
    circular=None,
    metric="l1",
    missing_penalty=0.0,
    empty_penalty=1e6,
    nan_penalty=1.0,
):
    """
    Weighted multivariate local distance for DTW.

    This version follows the older working logic:
    - normal finite values are compared after scaling;
    - circular variables, e.g. Dgeo, are handled correctly;
    - missing values get a small finite penalty;
    - there is no huge empty-row penalty.

    missing_penalty and empty_penalty are kept only for compatibility
    with the rest of the current code.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = len(a)

    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if scales is None:
        scales = np.ones(n, dtype=float)
    else:
        scales = np.asarray(scales, dtype=float)

    if circular is None:
        circular = np.zeros(n, dtype=bool)
    else:
        circular = np.asarray(circular, dtype=bool)

    if not (len(b) == len(weights) == len(scales) == len(circular) == n):
        raise ValueError("a, b, weights, scales, and circular must have compatible lengths.")

    total = 0.0
    total_weight = 0.0

    for ak, bk, wk, sk, is_circular in zip(a, b, weights, scales, circular):
        if not np.isfinite(wk) or wk <= 0:
            continue

        if not np.isfinite(sk) or sk <= 0:
            sk = 1.0

        if not np.isfinite(ak) or not np.isfinite(bk):
            if metric == "l1":
                total += wk * nan_penalty
            elif metric == "l2":
                total += wk * nan_penalty**2
            else:
                raise ValueError("metric must be 'l1' or 'l2'.")
        else:
            if is_circular:
                diff = circular_difference_degrees(ak, bk)
            else:
                diff = ak - bk

            z = abs(diff) / sk

            if metric == "l1":
                total += wk * z
            elif metric == "l2":
                total += wk * z * z
            else:
                raise ValueError("metric must be 'l1' or 'l2'.")

        total_weight += wk

    if total_weight == 0:
        return float(nan_penalty)

    if metric == "l1":
        return float(total / total_weight)

    if metric == "l2":
        return float(np.sqrt(total / total_weight))

def prepare_anchors(anchors, n_x, n_y, add_endpoints=True):
    """
    Validate anchors for x and y.

    Anchors must be pairs of post-skip indices:
        [(x_index, y_index), ...]

    They must be monotonic in both x and y.
    """
    if anchors is None:
        anchors = []

    anchors = [(int(i), int(j)) for i, j in anchors]

    if add_endpoints:
        if (0, 0) not in anchors:
            anchors.append((0, 0))
        if (n_x - 1, n_y - 1) not in anchors:
            anchors.append((n_x - 1, n_y - 1))

    anchors = sorted(set(anchors), key=lambda t: (t[0], t[1]))

    for i, j in anchors:
        if i < 0 or i >= n_x or j < 0 or j >= n_y:
            raise ValueError(
                f"Anchor {(i, j)} is outside array sizes {(n_x, n_y)}."
            )

    for (i0, j0), (i1, j1) in zip(anchors[:-1], anchors[1:]):
        if i1 <= i0 or j1 <= j0:
            raise ValueError(
                "Anchors must be strictly increasing in both sections. "
                f"Problem: {(i0, j0)} -> {(i1, j1)}"
            )

    return anchors

def get_pair_anchors(name_x, name_y, n_x, n_y, anchors_by_pair=None):
    """
    Fetch anchors for a pair of sections.

    anchors_by_pair example:
    {
        ("C", "B"): [(0, 0), (4, 3), (15, 13)]
    }

    If a reversed key exists, anchors are automatically swapped.
    If no anchors are provided, only endpoints are used.
    """
    if anchors_by_pair is None:
        anchors = None
    elif (name_x, name_y) in anchors_by_pair:
        anchors = anchors_by_pair[(name_x, name_y)]
    elif (name_y, name_x) in anchors_by_pair:
        anchors = [(j, i) for i, j in anchors_by_pair[(name_y, name_x)]]
    else:
        anchors = None

    return prepare_anchors(anchors, n_x=n_x, n_y=n_y, add_endpoints=True)

def anchored_dtw_multi(
    x,
    y,
    anchors=None,
    weights=None,
    scales=None,
    circular=None,
    metric="l1",
    missing_penalty=0.0,
    empty_penalty=1e6,
    normalize_distance=True,
    **dtw_kwargs,
):
    """
    Anchored multivariate DTW between two sections.

    Returns
    -------
    result : dict
        index_x : indices in x along final warping path
        index_y : indices in y along final warping path
        distance : recomputed final path distance, without duplicated anchor costs
        normalized_distance : distance / path_length
        path_length : number of matched pairs in final path
        anchors : validated anchor list
        dtw_objective_distance : raw sum of DTW segment objective distances
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same number of parameters.")

    anchors = prepare_anchors(
        anchors,
        n_x=len(x),
        n_y=len(y),
        add_endpoints=True,
    )

    def dist_method(a, b):
        return weighted_nan_dist(
            a,
            b,
            weights=weights,
            scales=scales,
            circular=circular,
            metric=metric,
            missing_penalty=missing_penalty,
            empty_penalty=empty_penalty,
        )

    full_index_x = []
    full_index_y = []
    dtw_objective_distance = 0.0

    for (i_start, j_start), (i_end, j_end) in zip(anchors[:-1], anchors[1:]):
        x_seg = x[i_start : i_end + 1]
        y_seg = y[j_start : j_end + 1]

        alignment = dtw(
            x_seg,
            y_seg,
            dist_method=dist_method,
            keep_internals=True,
            **dtw_kwargs,
        )

        idx_x = np.asarray(alignment.index1, dtype=int) + i_start
        idx_y = np.asarray(alignment.index2, dtype=int) + j_start

        # Avoid duplicate anchor pair at segment boundary.
        if len(full_index_x) > 0:
            idx_x = idx_x[1:]
            idx_y = idx_y[1:]

        full_index_x.extend(idx_x)
        full_index_y.extend(idx_y)

        # Keep this only as diagnostic information.
        dtw_objective_distance += float(alignment.distance)

    full_index_x = np.asarray(full_index_x, dtype=int)
    full_index_y = np.asarray(full_index_y, dtype=int)

    path_length = len(full_index_x)

    # Recompute the actual path distance after duplicate anchor removal.
    path_costs = [
        dist_method(x[i], y[j])
        for i, j in zip(full_index_x, full_index_y)
    ]

    distance = float(np.sum(path_costs))

    if normalize_distance:
        normalized_distance = distance / max(path_length, 1)
    else:
        normalized_distance = distance

    return {
        "index_x": full_index_x,
        "index_y": full_index_y,
        "distance": distance,
        "normalized_distance": normalized_distance,
        "path_length": path_length,
        "anchors": anchors,
        "dtw_objective_distance": dtw_objective_distance,
    }

def pairwise_dtw_matrix(
    sections,
    params,
    weights=None,
    scales=None,
    anchors_by_pair=None,
    circular_params=("Dgeo",),
    metric="l1",
    missing_penalty=0.0,
    empty_penalty=1e6,
    **dtw_kwargs,
):
    """
    Compute all-vs-all multivariate DTW distances.

    Returns
    -------
    distance_matrix : pandas.DataFrame
        Symmetric matrix of normalized DTW distances.

    alignments : dict
        {(name_x, name_y): alignment_result}
    """
    names = list(sections.keys())

    weights = make_weight_vector(params, weights)

    if scales is None:
        scales = compute_global_scales(sections, params=params, method="mad")

    circular = make_circular_mask(params, circular_params=circular_params)

    distance_matrix = pd.DataFrame(
        np.zeros((len(names), len(names))),
        index=names,
        columns=names,
        dtype=float,
    )

    alignments = {}

    for name_x, name_y in itertools.combinations(names, 2):
        x = sections[name_x]
        y = sections[name_y]

        anchors = get_pair_anchors(
            name_x,
            name_y,
            n_x=len(x),
            n_y=len(y),
            anchors_by_pair=anchors_by_pair,
        )

        result = anchored_dtw_multi(
            x,
            y,
            anchors=anchors,
            weights=weights,
            scales=scales,
            circular=circular,
            metric=metric,
            missing_penalty=missing_penalty,
            empty_penalty=empty_penalty,
            **dtw_kwargs,
        )

        d = result["normalized_distance"]

        distance_matrix.loc[name_x, name_y] = d
        distance_matrix.loc[name_y, name_x] = d

        alignments[(name_x, name_y)] = result

    return distance_matrix, alignments

def choose_dtw_medoid(distance_matrix):
    """
    Choose the section with the smallest mean distance to all other sections.
    Does not modify the original matrix.
    """
    dm = distance_matrix.copy()

    # Safer than np.fill_diagonal(dm.values, np.nan)
    mask = np.eye(len(dm), dtype=bool)
    dm = dm.mask(mask)

    mean_distance = dm.mean(axis=1, skipna=True)
    medoid_name = mean_distance.idxmin()

    return medoid_name, mean_distance

def align_all_to_reference(
    sections,
    reference_name,
    params,
    weights=None,
    scales=None,
    anchors_by_pair=None,
    circular_params=("Dgeo",),
    metric="l1",
    missing_penalty=0.0,
    empty_penalty=1e6,
    **dtw_kwargs,
):
    """
    Align every section to one reference section.

    Output alignment for each non-reference section has:
    - index_x = indices in reference
    - index_y = indices in the aligned section
    """
    if reference_name not in sections:
        raise ValueError(f"{reference_name!r} is not in sections.")

    weights = make_weight_vector(params, weights)

    if scales is None:
        scales = compute_global_scales(sections, params=params, method="mad")

    circular = make_circular_mask(params, circular_params=circular_params)

    ref = sections[reference_name]
    alignments_to_ref = {}

    for name, arr in sections.items():
        if name == reference_name:
            continue

        anchors = get_pair_anchors(
            reference_name,
            name,
            n_x=len(ref),
            n_y=len(arr),
            anchors_by_pair=anchors_by_pair,
        )

        result = anchored_dtw_multi(
            ref,
            arr,
            anchors=anchors,
            weights=weights,
            scales=scales,
            circular=circular,
            metric=metric,
            missing_penalty=missing_penalty,
            empty_penalty=empty_penalty,
            **dtw_kwargs,
        )

        alignments_to_ref[name] = result

    return alignments_to_ref

def evaluate_weight_sets(
    sections,
    params,
    weight_sets,
    scales,
    anchors_by_pair=None,
    circular_params=("Dgeo",),
    metric="l1",
    step_pattern="symmetric2",
    **dtw_kwargs,
):
    """
    Evaluate DTW distance matrices and medoids under several parameter-weight sets.

    Parameters
    ----------
    sections : dict
        Section arrays, e.g. {"A": arr_A, "B": arr_B, ...}

    params : list
        Parameter names in the same order as section array columns.

    weight_sets : dict
        Dictionary of named weight dictionaries.
        Example:
        {
            "balanced": {"Dgeo": 0.25, "Igeo": 1.0, "ARMmax": 0.5},
            "directions_stronger": {"Dgeo": 0.5, "Igeo": 1.5, "ARMmax": 0.25},
        }

    scales : array-like
        Scale vector in the same order as params.

    Returns
    -------
    summary : pandas.DataFrame
        One row per weight set.

    mean_distances_table : pandas.DataFrame
        Rows = sections, columns = weight-set names.
        Values = mean DTW distance to all other sections.

    distance_matrices : dict
        {weight_set_name: distance_matrix}

    pair_alignments_by_weight : dict
        {weight_set_name: pair_alignments}
    """
    results = []
    mean_distances_by_weight = {}
    distance_matrices = {}
    pair_alignments_by_weight = {}

    for weight_name, weights in weight_sets.items():
        dm, pair_alignments = pairwise_dtw_matrix(
            sections=sections,
            params=params,
            weights=weights,
            scales=scales,
            anchors_by_pair=anchors_by_pair,
            circular_params=circular_params,
            metric=metric,
            step_pattern=step_pattern,
            **dtw_kwargs,
        )

        medoid_name, mean_distances = choose_dtw_medoid(dm)

        distance_matrices[weight_name] = dm
        pair_alignments_by_weight[weight_name] = pair_alignments
        mean_distances_by_weight[weight_name] = mean_distances

        sorted_means = mean_distances.sort_values()

        best = sorted_means.iloc[0]
        second = sorted_means.iloc[1]
        margin = second - best
        relative_margin = margin / best if best != 0 else np.nan

        results.append(
            {
                "weight_set": weight_name,
                "medoid": medoid_name,
                "best_mean_distance": best,
                "second_best_section": sorted_means.index[1],
                "second_best_mean_distance": second,
                "absolute_margin": margin,
                "relative_margin": relative_margin,
            }
        )

    summary = pd.DataFrame(results)

    mean_distances_table = pd.DataFrame(mean_distances_by_weight)

    return summary, mean_distances_table, distance_matrices, pair_alignments_by_weight

def test_pair_with_anchors(
    sections,
    params,
    section_x,
    section_y,
    anchors,
    weights,
    scales,
    circular_params=("Dgeo",),
    metric="l1",
    step_pattern="symmetric2",
):
    """
    Compare one pair with and without anchors.
    """
    weight_vec = make_weight_vector(params, weights)
    circular = make_circular_mask(params, circular_params=circular_params)

    no_anchor = anchored_dtw_multi(
        sections[section_x],
        sections[section_y],
        anchors=None,
        weights=weight_vec,
        scales=scales,
        circular=circular,
        metric=metric,
        step_pattern=step_pattern,
    )

    with_anchor = anchored_dtw_multi(
        sections[section_x],
        sections[section_y],
        anchors=anchors,
        weights=weight_vec,
        scales=scales,
        circular=circular,
        metric=metric,
        step_pattern=step_pattern,
    )

    print(f"{section_x} vs {section_y}")
    print("No anchors distance:  ", no_anchor["normalized_distance"])
    print("With anchors distance:", with_anchor["normalized_distance"])
    print("No anchors path length:  ", no_anchor["path_length"])
    print("With anchors path length:", with_anchor["path_length"])

    print("\nAnchor check:")
    for ax, ay in anchors:
        present = np.any(
            (with_anchor["index_x"] == ax)
            & (with_anchor["index_y"] == ay)
        )
        print((ax, ay), present)

    return no_anchor, with_anchor

def dynamic_draw(values, x_pos):

    values = np.asarray(values, dtype=float)

    point_y = values[x_pos]

    fig, ax = plt.subplots()
    ax.plot(values)

    if not np.isnan(point_y):
        ax.scatter(x_pos, point_y, label=f'{x_pos}')
        ax.legend(loc='upper right')
    else:
        ax.legend('N',loc='upper right')
        
    plt.show()

def dynamic_draw_sections(dfs, section_names, param_name, x_pos, skips=None, ncols=3):

    if skips is None:
        skips = {}

    idx = x_pos

    n_sections = len(section_names)
    nrows = math.ceil(n_sections / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False
    )

    axes = axes.ravel()

    for ax, section_name in zip(axes, section_names):

        skip = skips.get(section_name, 0)

        values = (
            dfs[section_name][param_name]
            .iloc[skip:]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        x = np.arange(len(values))

        ax.plot(x, values)
        ax.set_title(f"{section_name}: {param_name}")
        ax.set_xlabel("Index")
        ax.set_ylabel(param_name)

        if idx < len(values):

            point_y = values[idx]

            if not np.isnan(point_y):
                ax.scatter(
                    x_pos,
                    point_y,
                    label=f"{section_name}:{x_pos}",
                    zorder=5
                )
                ax.legend(loc="upper right")

            else:
                ax.plot([], [], ' ', label=f"{section_name}:N")
                ax.legend(loc="upper right")

        else:
            ax.plot([], [], ' ', label=f"{section_name}:N")
            ax.legend(loc="upper right")

    for ax in axes[n_sections:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def get_pair_alignment(pair_alignments, ref_name, other_name):
    """
    Return DTW indices as:
        ref_idx   = indices of ref_name
        other_idx = indices of other_name

    Works even if the stored pair is reversed.
    """

    if (ref_name, other_name) in pair_alignments:
        res = pair_alignments[(ref_name, other_name)]
        ref_idx = res["index_x"]
        other_idx = res["index_y"]

    elif (other_name, ref_name) in pair_alignments:
        res = pair_alignments[(other_name, ref_name)]
        ref_idx = res["index_y"]
        other_idx = res["index_x"]

    else:
        raise KeyError(f"No alignment found for {ref_name}-{other_name}")

    return ref_idx, other_idx, res

def parse_data(df, row_name,header_length = 1):
    data = []

    for i, row in df.iterrows():
        if i+1 > header_length:
            data.append([i,row[row_name]])

    return np.asarray(data)

def get_pair_alignment(pair_alignments, ref_name, other_name):
    """
    Return DTW indices in the direction:
        ref_idx   = indices of ref_name
        other_idx = indices of other_name

    Works even if the stored pair is reversed.
    """

    if (ref_name, other_name) in pair_alignments:
        res = pair_alignments[(ref_name, other_name)]
        ref_idx = np.asarray(res["index_x"], dtype=int)
        other_idx = np.asarray(res["index_y"], dtype=int)

    elif (other_name, ref_name) in pair_alignments:
        res = pair_alignments[(other_name, ref_name)]
        ref_idx = np.asarray(res["index_y"], dtype=int)
        other_idx = np.asarray(res["index_x"], dtype=int)

    else:
        raise KeyError(f"No alignment found for {ref_name}-{other_name}")

    return ref_idx, other_idx, res

def collapse_to_reference_grid(ref_idx, other_idx, other_values, n_ref, agg="mean"):
    """
    Collapse one DTW alignment onto the original reference index grid.

    For example, if several points from section B match one medoid point,
    they are aggregated into one value.
    """

    other_values = np.asarray(other_values, dtype=float)

    if other_values.ndim == 1:
        other_values = other_values[:, None]

    aligned = np.full((n_ref, other_values.shape[1]), np.nan)

    for i in range(n_ref):
        mask = ref_idx == i

        if not np.any(mask):
            continue

        vals = other_values[other_idx[mask]]

        if agg == "mean":
            aligned[i] = np.nanmean(vals, axis=0)
        elif agg == "median":
            aligned[i] = np.nanmedian(vals, axis=0)
        elif agg == "first":
            aligned[i] = vals[0]
        elif agg == "last":
            aligned[i] = vals[-1]
        else:
            raise ValueError("agg must be one of: 'mean', 'median', 'first', 'last'")

    return aligned

def plot_all_sections_to_medoid(
    sections,
    params,
    pair_alignments,
    medoid,
    section_names=None,
    param_names=None,
    skips=None,
    agg="mean",
    ncols=3,
    figsize_per_panel=(5, 3),
    linewidth=1.5,
    alpha=0.9,
    show_points=False,
):
    """
    Plot all sections aligned to the medoid reference grid.

    X-axis = original medoid point index.
    Y-axis = parameter value.

    Each non-medoid section is warped to the medoid and collapsed onto
    the medoid index grid.
    """

    if section_names is None:
        section_names = list(sections.keys())

    if param_names is None:
        param_names = params

    if skips is None:
        skips = {}

    n_ref = len(sections[medoid])
    medoid_skip = skips.get(medoid, 0)

    x_ref = np.arange(n_ref) + medoid_skip

    aligned_to_medoid = {
        medoid: np.asarray(sections[medoid], dtype=float)
    }

    for other in section_names:
        if other == medoid:
            continue

        ref_idx, other_idx, res = get_pair_alignment(
            pair_alignments=pair_alignments,
            ref_name=medoid,
            other_name=other,
        )

        aligned_to_medoid[other] = collapse_to_reference_grid(
            ref_idx=ref_idx,
            other_idx=other_idx,
            other_values=sections[other],
            n_ref=n_ref,
            agg=agg,
        )

    n_params = len(param_names)
    nrows = math.ceil(n_params / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True,
    )

    axes = np.asarray(axes).ravel()

    for ax, param_name in zip(axes, param_names):
        param_id = params.index(param_name)

        for name in section_names:
            y = aligned_to_medoid[name][:, param_id]

            if name == medoid:
                ax.plot(
                    x_ref,
                    y,
                    label=f"{name} medoid",
                    linewidth=linewidth + 0.8,
                    alpha=1.0,
                )
            else:
                ax.plot(
                    x_ref,
                    y,
                    label=name,
                    linewidth=linewidth,
                    alpha=alpha,
                )

            if show_points:
                ax.scatter(x_ref, y, s=12, alpha=alpha)

        ax.set_title(param_name)
        ax.set_xlabel(f"{medoid} index")
        ax.grid(True, alpha=0.3)

    for ax in axes[n_params:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(section_names), 6),
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout()
    plt.show()

    return aligned_to_medoid

def plot_all_warped_param(
    aligned_to_medoid,
    params,
    param_name="Dgeo",
    section_names=None,
    figsize=(12, 5),
    linewidth=1.5,
    alpha=0.8,
):
    param_id = list(params).index(param_name)

    if section_names is None:
        section_names = list(aligned_to_medoid.keys())

    plt.figure(figsize=figsize)

    for name in section_names:
        y = aligned_to_medoid[name][:, param_id]
        warp_axis = np.arange(len(y))

        plt.plot(
            warp_axis,
            y,
            label=name,
            linewidth=linewidth,
            alpha=alpha,
        )

    plt.xlabel("DTW warped coordinate")
    plt.ylabel(param_name)
    plt.title(f"All samples warped: {param_name}")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.25)
    plt.show()

def compare_anchor_effect(matrices_no_anchor, matrices_anchor, weight_set="balanced"):
    return matrices_anchor[weight_set] - matrices_no_anchor[weight_set]

def spherical_mean_dec_inc(dec, inc, weights=None, min_count=1):
    """
    Mean direction from declination/inclination in degrees.

    Parameters
    ----------
    dec, inc : array-like
        Declination and inclination in degrees.
    weights : array-like or None
        Optional weights for each direction.
    min_count : int
        Minimum number of valid directions required.

    Returns
    -------
    mean_dec, mean_inc, Rbar, n
        mean_dec : mean declination, degrees [0, 360)
        mean_inc : mean inclination, degrees
        Rbar     : mean resultant length, 0..1
        n        : number of valid directions
    """

    dec = np.asarray(dec, dtype=float)
    inc = np.asarray(inc, dtype=float)

    valid = np.isfinite(dec) & np.isfinite(inc)

    if weights is None:
        weights = np.ones_like(dec, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0)

    dec = dec[valid]
    inc = inc[valid]
    weights = weights[valid]

    n = len(dec)

    if n < min_count:
        return np.nan, np.nan, np.nan, n

    dec_rad = np.deg2rad(dec)
    inc_rad = np.deg2rad(inc)

    # spherical coordinates -> Cartesian unit vectors
    x = np.cos(inc_rad) * np.cos(dec_rad)
    y = np.cos(inc_rad) * np.sin(dec_rad)
    z = np.sin(inc_rad)

    # weighted vector mean
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    z_mean = np.average(z, weights=weights)

    Rbar = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    if Rbar == 0:
        return np.nan, np.nan, Rbar, n

    mean_dec = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360
    mean_inc = np.rad2deg(
        np.arctan2(z_mean, np.sqrt(x_mean**2 + y_mean**2))
    )

    return mean_dec, mean_inc, Rbar, n

def average_selected_aligned_directions(
    aligned_to_medoid,
    params,
    selected_sections=None,
    dec_param="Dgeo",
    inc_param="Igeo",
    min_count=1,
    section_weights=None,
):
    """
    Row-wise spherical average of Dgeo/Igeo for selected aligned sections.

    Parameters
    ----------
    aligned_to_medoid : dict
        Example:
        {
            "A": array,
            "B": array,
            ...
        }

    params : list
        Parameter names corresponding to array columns.

    selected_sections : list or None
        Sections to average. If None, uses all sections in aligned_to_medoid.

    dec_param, inc_param : str
        Names of declination and inclination columns.

    min_count : int
        Minimum number of valid directions required at each row.

    section_weights : dict or None
        Optional weights, e.g.
        {"A": 1.0, "B": 0.5, "C": 1.0}

    Returns
    -------
    pandas.DataFrame
        Columns:
        Dgeo_mean, Igeo_mean, Rbar, n
    """

    params = list(params)

    dec_id = params.index(dec_param)
    inc_id = params.index(inc_param)

    if selected_sections is None:
        selected_sections = list(aligned_to_medoid.keys())

    selected_sections = list(selected_sections)

    n_rows = len(aligned_to_medoid[selected_sections[0]])

    for name in selected_sections:
        if len(aligned_to_medoid[name]) != n_rows:
            raise ValueError(
                f"Section {name!r} has length {len(aligned_to_medoid[name])}, "
                f"but expected {n_rows}."
            )

    rows = []

    for i in range(n_rows):
        decs = []
        incs = []
        weights = []

        for name in selected_sections:
            arr = aligned_to_medoid[name]

            decs.append(arr[i, dec_id])
            incs.append(arr[i, inc_id])

            if section_weights is None:
                weights.append(1.0)
            else:
                weights.append(section_weights.get(name, 1.0))

        mean_dec, mean_inc, Rbar, n = spherical_mean_dec_inc(
            decs,
            incs,
            weights=weights,
            min_count=min_count,
        )

        rows.append({
            "index": i,
            f"{dec_param}_mean": mean_dec,
            f"{inc_param}_mean": mean_inc,
            "Rbar": Rbar,
            "n": n,
        })

    return pd.DataFrame(rows).set_index("index")

def spherical_mean_dec_inc(dec, inc, weights=None, min_count=1):
    """
    Mean direction from declination/inclination in degrees.

    Parameters
    ----------
    dec, inc : array-like
        Declination and inclination in degrees.
    weights : array-like or None
        Optional weights for each direction.
    min_count : int
        Minimum number of valid directions required.

    Returns
    -------
    mean_dec, mean_inc, Rbar, n
        mean_dec : mean declination, degrees [0, 360)
        mean_inc : mean inclination, degrees
        Rbar     : mean resultant length, 0..1
        n        : number of valid directions
    """

    dec = np.asarray(dec, dtype=float)
    inc = np.asarray(inc, dtype=float)

    valid = np.isfinite(dec) & np.isfinite(inc)

    if weights is None:
        weights = np.ones_like(dec, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        valid &= np.isfinite(weights) & (weights > 0)

    dec = dec[valid]
    inc = inc[valid]
    weights = weights[valid]

    n = len(dec)

    if n < min_count:
        return np.nan, np.nan, np.nan, n

    dec_rad = np.deg2rad(dec)
    inc_rad = np.deg2rad(inc)

    # spherical coordinates -> Cartesian unit vectors
    x = np.cos(inc_rad) * np.cos(dec_rad)
    y = np.cos(inc_rad) * np.sin(dec_rad)
    z = np.sin(inc_rad)

    # weighted vector mean
    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)
    z_mean = np.average(z, weights=weights)

    Rbar = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

    if Rbar == 0:
        return np.nan, np.nan, Rbar, n

    mean_dec = np.rad2deg(np.arctan2(y_mean, x_mean)) % 360
    mean_inc = np.rad2deg(
        np.arctan2(z_mean, np.sqrt(x_mean**2 + y_mean**2))
    )

    return mean_dec, mean_inc, Rbar, n

def safe_sheet_name(name, used_names):
    name = str(name)
    name = re.sub(r'[:\\/?*\[\]]', '_', name)
    name = name[:31]

    if name == "":
        name = "Sheet"

    base = name
    i = 1

    while name in used_names:
        suffix = f"_{i}"
        name = base[:31 - len(suffix)] + suffix
        i += 1

    used_names.add(name)
    return name

def nanmean_no_warning(rows):
    """
    Column-wise nanmean without RuntimeWarning when a whole column is NaN.
    """
    rows = np.asarray(rows, dtype=float)

    valid_count = np.sum(np.isfinite(rows), axis=0)
    total = np.nansum(rows, axis=0)

    out = np.full(rows.shape[1], np.nan, dtype=float)

    np.divide(
        total,
        valid_count,
        out=out,
        where=valid_count > 0,
    )

    return out

def collapse_to_medoid_grid(
    medoid_idx,
    values,
    n_medoid_rows,
    params,
    dec_param="Dgeo",
    inc_param="Igeo",
):
    """
    Collapse DTW path values to one row per medoid-grid point.
    Uses existing spherical_mean_dec_inc().
    """

    medoid_idx = np.asarray(medoid_idx, dtype=int)
    values = np.asarray(values, dtype=float)

    params = list(params)
    n_params = len(params)

    aligned = np.full((n_medoid_rows, n_params), np.nan)

    dec_id = params.index(dec_param) if dec_param in params else None
    inc_id = params.index(inc_param) if inc_param in params else None

    for i in range(n_medoid_rows):
        rows = values[medoid_idx == i]

        if len(rows) == 0:
            continue

        # Ordinary parameters
        aligned[i, :] = nanmean_no_warning(rows)

        # Directional parameters
        if dec_id is not None and inc_id is not None:
            mean_dec, mean_inc, *_ = spherical_mean_dec_inc(
                rows[:, dec_id],
                rows[:, inc_id],
            )

            aligned[i, dec_id] = mean_dec
            aligned[i, inc_id] = mean_inc

    return aligned

def export_res(
    fname,
    mean_directions,
    warped_to_medoid,
    section_names,
    params,
    sections=None,
    dec_param="Dgeo",
    inc_param="Igeo",
):
    section_names = list(section_names)
    params = list(params)

    if len(warped_to_medoid) == 0:
        raise ValueError("warped_to_medoid is empty. Nothing to export.")

    first_other = next(iter(warped_to_medoid))
    medoid_name = warped_to_medoid[first_other]["medoid_name"]

    if sections is not None:
        n_medoid_rows = len(sections[medoid_name])
    else:
        max_idx = max(
            np.max(warped_to_medoid[name]["medoid_idx"])
            for name in warped_to_medoid
        )
        n_medoid_rows = int(max_idx) + 1

    sheets = {}

    for name in section_names:

        if name == medoid_name:
            if sections is not None:
                data = np.asarray(sections[medoid_name], dtype=float)
            else:
                data = collapse_to_medoid_grid(
                    medoid_idx=warped_to_medoid[first_other]["medoid_idx"],
                    values=warped_to_medoid[first_other]["medoid_values"],
                    n_medoid_rows=n_medoid_rows,
                    params=params,
                    dec_param=dec_param,
                    inc_param=inc_param,
                )

        else:
            if name not in warped_to_medoid:
                print(f"Skipping {name}: not found in warped_to_medoid")
                continue

            data = collapse_to_medoid_grid(
                medoid_idx=warped_to_medoid[name]["medoid_idx"],
                values=warped_to_medoid[name]["other_values"],
                n_medoid_rows=n_medoid_rows,
                params=params,
                dec_param=dec_param,
                inc_param=inc_param,
            )

        df = pd.DataFrame(data, columns=params)
        df.insert(0, "medoid_grid_index", np.arange(n_medoid_rows))

        sheets[name] = df

    if mean_directions is not None:
        mean_df = mean_directions.copy()

        if "medoid_grid_index" not in mean_df.columns:
            mean_df.insert(0, "medoid_grid_index", mean_df.index)

        sheets["Mean"] = mean_df

    if len(sheets) == 0:
        raise ValueError("No sheets were prepared. Nothing to export.")

    used_sheet_names = set()

    with pd.ExcelWriter(fname, engine="openpyxl") as writer:
        for name, df in sheets.items():
            sheet_name = safe_sheet_name(name, used_sheet_names)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Exported {len(sheets)} sheets to {fname}")
    