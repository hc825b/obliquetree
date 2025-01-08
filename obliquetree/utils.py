from __future__ import annotations

from .src.utils import export_tree as _export_tree
from ._pywrap import BaseTree, Classifier, Regressor
import json
from typing import Optional, Dict, Any, List, Union
from io import BytesIO
import os


def load_tree(tree_data: Union[str, Dict]) -> Union[Classifier, Regressor]:
    """
    Load a decision tree model from a JSON file or dictionary representation.

    This function reconstructs a trained decision tree from its serialized form,
    either from a JSON file on disk or a dictionary containing the tree structure
    and parameters.

    Parameters
    ----------
    tree_data : Union[str, Dict]
        Either:
        - A string containing the file path to a JSON file containing the tree data
        - A dictionary containing the serialized tree structure and parameters

    Returns
    -------
    Union[Classifier, Regressor]
        A reconstructed decision tree object. The specific type (Classifier or
        Regressor) is determined by the 'task' parameter in the tree data.
    """
    # Handle input types
    if isinstance(tree_data, str):
        if not os.path.exists(tree_data):
            raise FileNotFoundError(f"The file {tree_data} does not exist")

        with open(tree_data, "r") as f:
            tree = json.load(f)
    elif isinstance(tree_data, dict):
        tree = tree_data
    else:
        raise ValueError("Input must be a JSON string, file path, or dictionary")

    # Validate tree structure
    if (
        not isinstance(tree, dict)
        or "params" not in tree
        or "task" not in tree["params"]
    ):
        raise ValueError("Invalid tree data structure")

    # Create appropriate object based on task
    if not tree["params"]["task"]:
        obj = Classifier.__new__(Classifier)
    else:
        obj = Regressor.__new__(Regressor)

    tree["_fit"] = True

    obj.__setstate__(tree)
    
    return obj


def export_tree(
    tree: Union[Classifier, Regressor], out_file: str = None
) -> Union[None, dict]:
    """
    Serialize a decision tree model to a dictionary or JSON file.

    This function converts a trained decision tree into a portable format that can
    be saved to disk or transmitted. The serialized format preserves all necessary
    information to reconstruct the tree using load_tree().

    Parameters
    ----------
    tree : Union[Classifier, Regressor]
        The trained decision tree model to export. Must be an instance of either
        Classifier or Regressor and must have been fitted.

    out_file : str, optional
        If provided, the path where the serialized tree should be saved as a JSON
        file. If None, the function returns the dictionary representation instead
        of saving to disk.

    Returns
    -------
    Union[None, dict]
        If out_file is None:
            Returns a dictionary containing the serialized tree structure and parameters
        If out_file is provided:
            Returns None after saving the tree to the specified JSON file
    """
    if not isinstance(tree, BaseTree):
        raise ValueError("`tree` must be an instance of `BaseTree`.")

    if not tree._fit:
        raise ValueError(
            "The tree has not been fitted yet. Please call the 'fit' method to train the tree before using this function."
        )

    tree_dict = _export_tree(tree)  # Assuming this function is implemented elsewhere.

    if out_file is not None:
        if isinstance(out_file, str):
            with open(out_file, "w") as f:
                json.dump(tree_dict, f, indent=4)
        else:
            raise ValueError("`out_file` must be a string if provided.")

    else:
        return tree_dict


def visualize_tree(
    tree: Union[Classifier, Regressor],
    feature_names: Optional[List[str]] = None,
    max_cat: Optional[int] = None,
    max_oblique: Optional[int] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    figsize: tuple = (20, 10),
    show_gini: bool = True,
    show_n_samples: bool = True,
    show_node_value: bool = True,
) -> None:
    """
    Generate a visual representation of a decision tree model.

    Creates a graphical visualization of the tree structure showing decision nodes,
    leaf nodes, and split criteria. The visualization can be customized to show
    various node statistics and can be displayed or saved to a file.

    Parameters
    ----------
    tree : Union[Classifier, Regressor]
        The trained decision tree model to visualize. Must be fitted before
        visualization.

    feature_names : List[str], optional
        Human-readable names for the features used in the tree. If provided,
        these names will be used in split conditions instead of generic feature
        indices (e.g., "age <= 30" instead of "f0 <= 30").

    max_cat : int, optional
        For categorical splits, limits the number of categories shown in the
        visualization. If there are more categories than this limit, they will
        be truncated with an ellipsis. Useful for splits with many categories.

    max_oblique : int, optional
        For oblique splits (those involving multiple features), limits the number
        of features shown in the split condition. Helps manage complex oblique
        splits in the visualization.

    save_path : str, optional
        If provided, saves the visualization to this file path. The file format
        is determined by the file extension (e.g., '.png', '.pdf').

    dpi : int, default=600
        The resolution (dots per inch) of the saved image. Only relevant if
        save_path is provided.

    figsize : tuple, default=(20, 10)
        The width and height of the figure in inches.

    show_gini : bool, default=True
        Whether to display Gini impurity values in the nodes.

    show_n_samples : bool, default=True
        Whether to display the number of samples that reach each node.

    show_node_value : bool, default=True
        Whether to display the predicted value/class distributions in each node.

    Returns
    -------
    None
        The function displays the visualization and optionally saves it to disk.
    """
    _check_visualize_tree_inputs(
        tree, feature_names, max_cat, max_oblique, save_path, dpi, figsize
    )

    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError(
            "graphviz is not installed. Please install it to use this function."
        )

    try:
        from matplotlib.pyplot import figure, imshow, imread, axis, savefig, show
    except:
        raise ImportError(
            "matplotlib is not installed. Please install it to use this function."
        )

    tree_dict = _export_tree(tree)  # Assuming this function is implemented elsewhere.

    node, params = tree_dict["tree"], tree_dict["params"]

    def _visualize_recursive(node, graph=None, parent=None, edge_label=""):
        if graph is None:
            graph = Digraph(format="png")
            graph.graph_attr.update(
                {
                    "rankdir": "TB",
                    "ranksep": "0.3",
                    "nodesep": "0.2",
                    "splines": "polyline",
                    "ordering": "out",
                }
            )
            graph.attr(
                "node",
                shape="box",
                style="filled",
                color="lightgrey",
                fontname="Helvetica",
                margin="0.2",
            )

        node_id = str(id(node))
        label_parts = []

        is_leaf = "left" not in node and "right" not in node

        if is_leaf:
            # For leaf nodes
            label_parts.append("leaf")
            label_parts.append(_format_value_str(node, params))

            # Add impurity for leaf nodes if requested and available
            if show_gini and "impurity" in node:
                label_parts.append(f"impurity: {_format_float(node['impurity'])}")

            # Add n_samples for leaf nodes if requested and available
            if show_n_samples and "n_samples" in node:
                label_parts.append(f"n_samples: {node['n_samples']}")

            graph.node(
                node_id,
                label="\n".join(label_parts),
                shape="box",
                style="filled",
                color="lightblue",
                fontname="Helvetica",
            )
        else:
            # For internal nodes
            # First add the split information
            if node.get("is_oblique", False):
                split_info = _create_oblique_expression(
                    node["features"],
                    node["weights"],
                    node["threshold"],
                    feature_names,
                    max_oblique,
                )
            elif "category_left" in node:
                categories = node["category_left"]
                cat_str = _format_categories(categories, max_cat)
                feature_label = (
                    feature_names[node["feature_idx"]]
                    if feature_names and node["feature_idx"] < len(feature_names)
                    else f"f{node['feature_idx']}"
                )
                split_info = f"{feature_label} in {cat_str}"
            else:
                threshold = (
                    _format_float(node["threshold"])
                    if isinstance(node["threshold"], float)
                    else node["threshold"]
                )
                feature_label = (
                    feature_names[node["feature_idx"]]
                    if feature_names and node["feature_idx"] < len(feature_names)
                    else f"f{node['feature_idx']}"
                )
                split_info = f"{feature_label} ≤ {threshold}"

            label_parts.append(split_info)

            if show_node_value:
                label_parts.append(_format_value_str(node, params))

            # Add Gini impurity if requested
            if show_gini and "impurity" in node:
                label_parts.append(f"impurity: {_format_float(node['impurity'])}")

            # Add n_samples if requested
            if show_n_samples and "n_samples" in node:
                label_parts.append(f"n_samples: {node['n_samples']}")

            graph.node(
                node_id,
                label="\n".join(label_parts),
                shape="box",
                style="filled",
                color="lightgrey",
                fontname="Helvetica",
            )

        if parent is not None:
            graph.edge(
                parent,
                node_id,
                label=edge_label,
                fontname="Helvetica",
                penwidth="1.0",
                minlen="1",
            )

        if "left" in node:
            _visualize_recursive(node["left"], graph, node_id, "Left")
        if "right" in node:
            _visualize_recursive(node["right"], graph, node_id, "Right")

        return graph

    graph = _visualize_recursive(node)
    png_data = graph.pipe(format="png")

    figure(figsize=figsize)
    imshow(imread(BytesIO(png_data)))
    axis("off")

    if save_path:
        savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    show()


def _format_float(value: float) -> str:
    """Format float value with 3 decimal places, return '0' for 0.0"""
    if value == 0.0:
        return "0"
    return f"{value:.2f}"


def _format_value_str(node: Dict[str, Any], params: Dict[str, Any]) -> str:
    """
    Format value string based on task type (regression vs classification) and number of classes

    Parameters:
    -----------
    node : Dict[str, Any]
        The tree node dictionary containing values or value
    params : Dict[str, Any]
        Tree parameters containing task and n_classes information
    """
    # Check if it's a regression task
    if params["task"]:
        value = (
            _format_float(node["value"])
            if isinstance(node["value"], float)
            else node["value"]
        )
        return f"Value: {value}"

    # Classification task
    else:
        if params["n_classes"] == 2:  # Binary classification
            # For binary case, node["values"] contains probability for positive class
            p = node["value"]
            return f"values: [{_format_float(1-p)}, {_format_float(p)}]"
        else:  # Multiclass (3 or more classes)
            values_str = ", ".join(
                _format_float(v) if isinstance(v, float) else str(v)
                for v in node.get("values", [0.0] * params["n_classes"])
            )
            return f"values: [{values_str}]"


def _create_oblique_expression(
    features: list,
    weights: list,
    threshold: float,
    feature_names: Optional[List[str]],
    max_oblique: Optional[int] = None,
) -> str:
    """Create mathematical expression for oblique split with line breaks after 5 terms"""
    terms = []

    # Sort features and weights by absolute weight value
    feature_weight_pairs = sorted(
        zip(features, weights), key=lambda x: abs(x[1]), reverse=True
    )

    # Apply max_oblique limit if specified
    if max_oblique is not None:
        feature_weight_pairs = feature_weight_pairs[:max_oblique]
        if len(features) > max_oblique:
            feature_weight_pairs.append(("...", 0))

    # Create terms with proper formatting
    lines = []
    current_line = []

    for i, (f, w) in enumerate(feature_weight_pairs):
        if f == "...":
            current_line.append("...")
            continue

        feature_label = (
            feature_names[f] if feature_names and f < len(feature_names) else f"f{f}"
        )

        if w == 1.0:
            term = feature_label  # Removed parentheses for coefficient 1
        elif w == -1.0:
            term = f"–{feature_label}"  # Removed parentheses for coefficient -1
        else:
            formatted_weight = _format_float(abs(w))
            term = f"{'– ' if w < 0 else ''}({formatted_weight} * {feature_label})"

        if i > 0:
            term = f"+ {term}" if w > 0 else f" {term}"

        current_line.append(term)

        # Start new line after every 5 terms
        if len(current_line) == 5 and i < len(feature_weight_pairs) - 1:
            lines.append(" ".join(current_line) + " +")
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    formatted_threshold = _format_float(threshold)
    expression = "\n".join(lines)
    return f"{expression} ≤ {formatted_threshold}"


def _format_categories(categories: list, max_cat: Optional[int] = None) -> str:
    """Format category list with line breaks after every 5 items"""
    if max_cat is not None and len(categories) > max_cat:
        shown_cats = categories[:max_cat]
        return f"[{', '.join(map(str, shown_cats))}, ...]"

    formatted_cats = []
    current_line = []

    for i, cat in enumerate(categories):
        current_line.append(str(cat))

        # Add line break after every 5 items or at the end
        if len(current_line) == 9 and i < len(categories) - 1:
            formatted_cats.append(", ".join(current_line) + ",")
            current_line = []

    if current_line:
        formatted_cats.append(", ".join(current_line))

    if len(formatted_cats) > 1:
        return "[" + "\n".join(formatted_cats) + "]"
    return f"[{formatted_cats[0]}]"


def _check_visualize_tree_inputs(
    tree: BaseTree,
    feature_names: Optional[List[str]] = None,
    max_cat: Optional[int] = None,
    max_oblique: Optional[int] = None,
    save_path: Optional[str] = None,
    dpi: int = 600,
    figsize: tuple = (20, 10),
) -> None:
    """
    Validate the inputs for the visualize_tree function.

    Parameters:
    -----------
    tree : object
        The tree object to be visualized, must have a certain expected structure.
    feature_names : Optional[List[str]]
        If provided, must be a list of strings matching the number of features in the tree.
    max_cat : Optional[int]
        If provided, must be a positive integer.
    max_oblique : Optional[int]
        If provided, must be a positive integer.
    save_path : Optional[str]
        If provided, must be a valid file path ending in a supported image format (e.g., '.png').
    dpi : int
        Must be a positive integer.
    figsize : tuple
        Must be a tuple of two positive numbers.
    """
    if not isinstance(tree, BaseTree):
        raise ValueError("`tree` must be an instance of `BaseTree`.")

    if not tree._fit:
        raise ValueError(
            "The tree has not been fitted yet. Please call the 'fit' method to train the tree before using this function."
        )

    if feature_names is not None:
        if not isinstance(feature_names, list) or not all(
            isinstance(f, str) for f in feature_names
        ):
            raise ValueError("feature_names must be a list of strings.")
        if len(feature_names) != tree.n_features:
            raise ValueError(
                f"feature_names must match the number of features in the tree ({tree.n_features})."
            )

    if max_cat is not None and (not isinstance(max_cat, int) or max_cat <= 0):
        raise ValueError("max_cat must be a positive integer.")

    if max_oblique is not None and (
        not isinstance(max_oblique, int) or max_oblique <= 0
    ):
        raise ValueError("max_oblique must be a positive integer.")

    if save_path is not None and not isinstance(save_path, str):
        raise ValueError("save_path must be a string representing a valid file path.")

    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")

    if (
        not isinstance(figsize, tuple)
        or len(figsize) != 2
        or not all(isinstance(dim, (int, float)) and dim > 0 for dim in figsize)
    ):
        raise ValueError("figsize must be a tuple of two positive numbers.")
