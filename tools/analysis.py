"""
Helper script to analyze and visualize metadata, CLIP, and classifier confidence data.
# Description: A module containing functions for analyzing and visualizing data.


# Functions:
    - getValidationSummary
    - getInputMetadata
    - load_metadata
    - getPairs
    - summarizeSimilarities
    - getGroupSummary
    - getSummaryByGroups
    - analyze_metadata
    - analyze_clip
    - analyze_classifier_confidence
    - viz_scatter
    - viz_hist
    - viz_scatter_prompts
    - viz_hist_zone
    - viz_hist_clf
    - analyze_confusion_zone
    - analyze_clf_stats
    - getStats
    - analyze_clip_trace
    - symmetrize_steps
    - computeSimilarityByInterpolation
    - plot_kde
    - getRegPlot
"""

import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from tools import classes


def getValidationSummary(val_classifier_json):
    """
    Retrieves the validation summary from a JSON file.

    Args:
        val_classifier_json (str): The path to the JSON file containing the validation summary.

    Returns:
        tuple: A tuple containing the following:
            - log_output (str): The formatted validation summary.
            - val_metrics_df (pandas.DataFrame): The validation metrics breakdown.
            - adv_metrics_df (pandas.DataFrame): The adversarial metrics breakdown.
            - val_output (dict): The original validation output dictionary.
            - adv_metrics (dict): The filtered adversarial metrics dictionary.
    """
    with open(val_classifier_json) as f:
        val_output = json.load(f)
    header_str = f"Validation Summary\nmodel: {val_output['eval_model']}\n"
    # overview validation metrics on validation set
    validation_metrics_str = f"""Validation Set:
    Top 1 Accuracy: {val_output['val_metrics']['acc1']}
    Top 5 Accuracy: {val_output['val_metrics']['acc5']}
    Loss: {val_output['val_metrics']['loss']}
    """
    adv_metrics = dict()
    for k, v in list(val_output["adv_metrics"].items()) + list(
        val_output["adv_val_metrics"].items()
    ):
        if k in ["acc1", "acc5", "RMSE-CE", "AURRA"]:
            adv_metrics[k] = v
    # overview validation metrics on adversarial set
    adversarial_metrics_str = f"""Adversarial Set:
    Top 1 Accuracy: {adv_metrics['acc1']}
    Top 5 Accuracy: {adv_metrics['acc5']}
    RMSE-CE: {adv_metrics['RMSE-CE']}
    AURRA: {adv_metrics['AURRA']}
    """
    # val class breakdown
    val_metrics_df = pd.DataFrame.from_dict(
        val_output["val_metrics_details"], orient="index"
    )
    val_metrics_df["acc1"] = val_metrics_df["val_metrics"].apply(lambda x: x["acc1"])
    val_metrics_df["acc5"] = val_metrics_df["val_metrics"].apply(lambda x: x["acc5"])
    val_metrics_df = (
        val_metrics_df.drop(columns=["val_metrics"])
        .reset_index()
        .rename(columns={"top_n_classes": "top_misclassified", "index": "class"})
    )
    # adv class breakdown
    adv_metrics_df = pd.DataFrame.from_dict(
        val_output["adv_metrics_details"], orient="index"
    )
    adv_metrics_df["acc1"] = adv_metrics_df["val_metrics"].apply(lambda x: x["acc1"])
    adv_metrics_df["acc5"] = adv_metrics_df["val_metrics"].apply(lambda x: x["acc5"])
    adv_metrics_df = (
        adv_metrics_df.drop(columns=["val_metrics"])
        .reset_index()
        .rename(columns={"top_n_classes": "top_misclassified", "index": "class"})
    )
    log_output = "".join(
        [header_str, "\n", validation_metrics_str, "\n", adversarial_metrics_str]
    )
    return log_output, val_metrics_df, adv_metrics_df, val_output, adv_metrics


def getInputMetadata(input_dir):
    """
    Retrieves metadata from JSON files in the specified input directory.

    Args:
        input_dir (str): The path to the input directory.

    Returns:
        dict: A dictionary containing the metadata extracted from the JSON files.
              The keys are the filenames without the extension, and the values are
              the corresponding metadata loaded from the JSON files.
    """
    input_metadata = {
        i.split(".")[0]: os.path.join(input_dir, i)
        for i in os.listdir(input_dir)
        if i.endswith(".json")
    }
    for k, v in input_metadata.items():
        with open(v, "r") as f:
            input_metadata[k] = json.load(f)
    return input_metadata


def load_metadata(input_metadata: dict):
    """
    Load metadata from a dictionary into a pandas DataFrame.

    Args:
        input_metadata (dict): A dictionary containing the metadata.

    Returns:
        pandas.DataFrame: The metadata loaded into a DataFrame.
    """
    df = pd.DataFrame.from_dict(input_metadata, orient="index")
    df = df.reset_index()
    df["class_pairs"] = df["class_pairs"].apply(lambda x: (x[0], x[1]))
    return df


def getPairs(df: pd.DataFrame, groups_list: list):
    """
    Get pairs of dataframes based on unique combinations of values in specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        groups_list (list): A list of column names to group by.

    Returns:
        tuple: A tuple containing the pair_prompt_group and pair_dict.
            - pair_prompt_group (numpy.ndarray): An array of unique combinations of values in the specified columns.
            - pair_dict (dict): A dictionary where the keys are the index of each pair in pair_prompt_group,
              and the values are the corresponding filtered DataFrames.

    """
    pair_prompt_group = df.drop_duplicates(subset=groups_list, keep="first")[
        groups_list
    ].values
    pair_dict = dict()
    for n_pair, pair_group in enumerate(pair_prompt_group):
        df_filtered = df.copy(deep=True)
        for n_cond, cond in enumerate(pair_group):
            df_filtered = df_filtered.loc[df_filtered[groups_list[n_cond]] == cond]
        pair_dict[n_pair] = df_filtered
    return pair_prompt_group, pair_dict


def summarizeSimilarities(x_dict, summary):
    """
    Summarizes the similarities in the given dictionary and updates the summary dictionary.

    Args:
        x_dict (dict): A dictionary containing the similarities.
        summary (dict): A dictionary to store the summary of similarities.

    Returns:
        dict: The updated summary dictionary.
    """
    for step, details in x_dict.items():
        keylist = details["similarity"].keys()
        if step not in summary:
            summary[step] = dict()
            for key in keylist:
                summary[step][key + "_total"] = 0
                summary[step][key + "_count"] = 0
        for key in keylist:
            summary[step][key + "_total"] += details["similarity"][key]
            summary[step][key + "_count"] += 1
    return summary


def getGroupSummary(pair_dict, embed_type="prompt_embed_similarity"):
    """
    Calculate the group summary for each group in the pair dictionary.

    Args:
        pair_dict (dict): A dictionary containing group indices as keys and dataframes as values.
        embed_type (str, optional): The column name in the dataframe to use for similarity calculation. Defaults to "prompt_embed_similarity".

    Returns:
        dict: A dictionary containing the group summary for each group.
    """
    pair_dict = pair_dict.copy()
    group_summary = dict()
    for group_i, df_i in pair_dict.items():
        group_i = str(group_i)
        group_summary["group_" + group_i] = dict()
        if embed_type in df_i.columns and df_i.shape[0] > 0:
            for i, row in df_i.iterrows():
                group_summary["group_" + group_i] = summarizeSimilarities(
                    row[embed_type], group_summary["group_" + group_i]
                )
    return group_summary


def getSummaryByGroups(group_summary, show_all=False):
    """
    Generate a summary of group data by calculating means for different classes and relative distances.

    Args:
        group_summary (dict): A dictionary containing group-wise summary data.
        show_all (bool, optional): Flag to indicate whether to include all columns in the summary dataframe.
                                   Defaults to False.

    Returns:
        dict: A modified version of the group_summary dictionary with means calculated and unnecessary columns removed.
    """
    group_summary = group_summary.copy()
    for group, details in group_summary.items():
        for step, step_details in details.items():
            if step_details["class1_count"] > 0:
                step_details["class1_mean"] = (
                    step_details["class1_total"] / step_details["class1_count"]
                )
            else:
                step_details["class1_mean"] = np.nan
            if step_details["class2_count"] > 0:
                step_details["class2_mean"] = (
                    step_details["class2_total"] / step_details["class2_count"]
                )
            else:
                step_details["class2_mean"] = np.nan
            if step_details["class1_relative_distance_count"] > 0:
                step_details["class1_relative_distance_mean"] = (
                    step_details["class1_relative_distance_total"]
                    / step_details["class1_relative_distance_count"]
                )
            else:
                step_details["class1_relative_distance_mean"] = np.nan
            if step_details["class2_relative_distance_count"] > 0:
                step_details["class2_relative_distance_mean"] = (
                    step_details["class2_relative_distance_total"]
                    / step_details["class2_relative_distance_count"]
                )
            else:
                step_details["class2_relative_distance_mean"] = np.nan
        group_summary[group] = (
            pd.DataFrame.from_dict(group_summary[group], orient="index")
            .reset_index()
            .rename(columns={"index": "interpolation_step"})
        )
        if not show_all:
            extra_cols = [
                "class1_mean",
                "class2_mean",
                "class1_relative_distance_mean",
                "class2_relative_distance_mean",
            ]
            group_summary[group] = group_summary[group][
                [col for col in group_summary[group].columns if col in extra_cols]
            ]
    return group_summary


def analyze_metadata(
    input_dir: str,
    groups_list=[
        "class_pairs",
        "prompt_text_0",
        "prompt_text_1",
        "negative_prompt_text_0",
        "negative_prompt_text_1",
    ],
):
    """
    Analyzes the metadata from the given input directory.

    Args:
        input_dir (str): The path to the input directory containing the metadata.
        groups_list (list, optional): A list of group names to analyze. Defaults to ["class_pairs", "prompt_text_0", "prompt_text_1", "negative_prompt_text_0", "negative_prompt_text_1"].

    Returns:
        tuple: A tuple containing the pair-prompt group and the summary by groups.
    """
    input_metadata = getInputMetadata(input_dir)
    df = load_metadata(input_metadata)
    pair_prompt_group, pair_dict = getPairs(df, groups_list)
    group_summary = getGroupSummary(pair_dict)
    summary_by_groups = getSummaryByGroups(group_summary)
    return pair_prompt_group, summary_by_groups


def analyze_clip(
    prompts,
    clip_dataset_dict,
    clip_model_path="openai/clip-vit-large-patch14",
    device="cuda",
):
    """
    Analyzes the given clip_dataset_dict using the CLIP model.

    Args:
        prompts (str or List[str]): The prompts to be used for analysis.
        clip_dataset_dict (dict): A dictionary containing the datasets to be analyzed.
            The keys represent the type of dataset, and the values represent the datasets themselves.
        clip_model_path (str, optional): The path or name of the CLIP model to be used.
            Defaults to "openai/clip-vit-large-patch14".
        device (str, optional): The device to run the analysis on.
            Defaults to "cuda".

    Returns:
        pandas.DataFrame: A DataFrame containing the analysis results.

    """
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_dict = {
        "dataset": list(),
        "type": list(),
        "file_path": list(),
        "file_name": list(),
        "actual_class": list(),
        "actual_classname": list(),
        "clip_pred_class": list(),
        "clip_pred_classname": list(),
        "cos_sim": list(),
        "delta_sim": list(),
        "sim_0": list(),
        "sim_1": list(),
    }
    with torch.inference_mode():
        for type_name, ds in tqdm(clip_dataset_dict.items()):
            class_to_idx = ds.class_to_idx
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            for i, (img, label) in enumerate(ds):
                file = ds.samples[i][0]
                file = os.path.normpath(file)
                folders = os.path.dirname(file).split(os.sep)
                dataset_type = "".join([folders[0], "/", folders[1]])
                inputs = clip_processor(
                    text=prompts, padding=True, images=img, return_tensors="pt"
                ).to(device)
                scores = (
                    clip_model(**inputs)
                    .logits_per_image.detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                pred = np.argmax(scores).item()
                actual_class = idx_to_class[label]
                pred_class = idx_to_class[pred]
                pred_score = scores[pred]
                pred_class_name = classes.IMAGENET2012_CLASSES[pred_class]

                clip_dict["dataset"].append(dataset_type)
                clip_dict["type"].append(type_name)
                clip_dict["file_path"].append(file)
                clip_dict["file_name"].append(os.path.basename(file))
                clip_dict["actual_class"].append(actual_class)
                clip_dict["actual_classname"].append(
                    classes.IMAGENET2012_CLASSES[actual_class]
                )
                clip_dict["clip_pred_class"].append(pred_class)
                clip_dict["clip_pred_classname"].append(pred_class_name)
                clip_dict["cos_sim"].append(pred_score.item())
                clip_dict["delta_sim"].append((scores)[0].item() - (scores)[1].item())
                clip_dict["sim_0"].append(scores[0].item())
                clip_dict["sim_1"].append(scores[1].item())
    return pd.DataFrame.from_dict(clip_dict)


def analyze_classifier_confidence(
    val_dataset_dict, model, base_class_idx, suffix="", device="cuda"
):
    """
    Analyzes the confidence of a classifier model on a dataset with the same transformation parameters used to load validation set.

    Args:
        val_dataset_dict (dict): A dictionary containing the train, val, synth, (etc.) datasets transformed with validation parameters.
        model: The classifier model to analyze.
        base_class_idx (int): The index of the base class for confidence analysis.
        suffix (str, optional): A suffix to append to the column names in the output dataframe. Defaults to "".
        device (str, optional): The device to use for inference. Defaults to "cuda".

    Returns:
        pandas.DataFrame: A dataframe containing the analysis results.

    """
    clf_dict = {
        "dataset": list(),
        "type": list(),
        "file_path": list(),
        "file_name": list(),
        "actual_class": list(),
        "actual_classname": list(),
        f"clf_pred_class{suffix}": list(),
        f"clf_pred_classname{suffix}": list(),
        f"y_conf{suffix}": list(),
        f"correct{suffix}": list(),
    }
    with torch.inference_mode():
        for type_name, ds in tqdm(val_dataset_dict.items()):
            class_to_idx = ds.class_to_idx
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            for i, (img, label) in enumerate(ds):
                file = ds.samples[i][0]
                file = os.path.normpath(file)
                folders = os.path.dirname(file).split(os.sep)
                dataset_type = "".join([folders[0], "/", folders[1]])
                output = model(img.unsqueeze(0).to(device))
                probs = torch.nn.functional.softmax(output, dim=1)

                scores = probs[0, base_class_idx].detach().cpu().numpy()
                # scores = scores / scores.sum()
                pred = np.argmax(scores)
                actual_class = idx_to_class[label]
                pred_class = idx_to_class[pred]
                pred_class_name = classes.IMAGENET2012_CLASSES[pred_class]

                clf_dict["dataset"].append(dataset_type)
                clf_dict["type"].append(type_name)
                clf_dict["file_path"].append(file)
                clf_dict["file_name"].append(os.path.basename(file))
                clf_dict["actual_class"].append(actual_class)
                clf_dict["actual_classname"].append(
                    classes.IMAGENET2012_CLASSES[actual_class]
                )
                clf_dict[f"clf_pred_class{suffix}"].append(pred_class)
                clf_dict[f"clf_pred_classname{suffix}"].append(pred_class_name)
                clf_dict[f"y_conf{suffix}"].append(scores.max())
                clf_dict[f"correct{suffix}"].append(1 if label == pred else 0)
    return pd.DataFrame.from_dict(clf_dict)


def viz_scatter(
    df,
    class_name_pairs,
    class_pairs,
    plot_type=None,
    xrange=None,
    sep_line_type=False,
    sep_line=True,
    save=None,
):
    """
    Create a scatter plot to visualize the data.

    Args:
        df (DataFrame): The input DataFrame containing the data.
        class_name_pairs (list): A list of class name pairs.
        class_pairs (list): A list of class pairs.
        plot_type (list, optional): A list of plot types. Defaults to None.
        xrange (tuple, optional): The range of x-axis values. Defaults to None.
        sep_line_type (bool, optional): Whether to plot separate lines for each type. Defaults to False.
        sep_line (bool, optional): Whether to plot a separate line for all data. Defaults to True.
        save (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        None
    """
    # A dictionary to map types to colors for easier customization
    colors = {
        ("train", 0): "royalblue",
        ("val", 0): "limegreen",
        ("synth", 0): "lightsteelblue",
        ("mix", 0): "dodgerblue",
        ("train", 1): "orangered",
        ("val", 1): "magenta",
        ("synth", 1): "lightsalmon",
        ("mix", 1): "tomato",
        ("train", 2): "dimgrey",
        ("synth", 2): "darkgrey",
        ("val", 2): "silver",
        ("mix", 2): "slategray",
    }
    plot_type = df["type"].unique() if plot_type is None else plot_type
    # Create a scatter plot
    plt.figure(figsize=(16, 8))  # Adjust figure size
    for i, class_id in enumerate(set(df["actual_class"])):
        id = 0 if class_id == class_pairs[0] else 1
        for j, type_ in enumerate(set(plot_type)):  # Iterate through unique types
            # Filter data for the current type
            x_type = df[(df["type"] == type_) & (df["actual_class"] == class_id)][
                "sim_0"
            ]
            y_type = df[(df["type"] == type_) & (df["actual_class"] == class_id)][
                "sim_1"
            ]
            # Plot the data
            plt.scatter(
                x_type,
                y_type,
                label=(type_, class_name_pairs[id]),
                color=colors[(type_, id)],
                alpha=0.5,
                marker=".",
            )
    if sep_line_type:
        for j, type_ in enumerate(set(plot_type)):
            clf = SVC(kernel="linear")
            df_type = df[df["type"] == type_]
            clf.fit(df_type[["sim_0", "sim_1"]], df_type["actual_class"])
            x_line = np.linspace(
                min(df["sim_0"].min(), df["sim_1"].min()),
                max(df["sim_0"].max(), df["sim_1"].max()),
                100,
            )
            y_line = (-1 * clf.intercept_ - clf.coef_[0][0] * x_line) / clf.coef_[0][1]
            plt.plot(
                x_line,
                y_line,
                color=colors[(type_, 2)],
                linestyle="--",
                label=f"SVM_{type_}",
            )
    if sep_line:
        clf = SVC(kernel="linear")
        clf.fit(df[["sim_0", "sim_1"]], df["actual_class"])
        x_line = np.linspace(
            min(df["sim_0"].min(), df["sim_1"].min()),
            max(df["sim_0"].max(), df["sim_1"].max()),
            100,
        )
        y_line = (-1 * clf.intercept_ - clf.coef_[0][0] * x_line) / clf.coef_[0][1]
        plt.plot(
            x_line,
            y_line,
            color="grey",
            linestyle="--",
            label="SVM_all",
        )
    if xrange is not None:
        plt.xticks(np.arange(xrange[0], xrange[1] + 0.1, 0.1))
    # Add labels and title
    plt.xlabel(f"Cosine similarity to {class_name_pairs[0]}")
    plt.ylabel(f"Cosine similarity to {class_name_pairs[1]}")
    plt.title(
        f"CLIP cosine similarity between {class_name_pairs[0].lower()} vs {class_name_pairs[1].lower()}"
    )
    # Add a legend
    plt.legend()
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    # Show the plot
    plt.show()
    return None


def viz_hist(
    df,
    class_name_pairs,
    class_pairs,
    plot_type=None,
    bins=100,
    xrange=None,
    separator_text=False,
    mix_line=False,
    save=None,
):
    """
    Visualizes the distribution of the 'delta_sim' values in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        class_name_pairs (list): A list of two class names.
        class_pairs (list): A list of two class IDs.
        plot_type (list, optional): A list of plot types. If None, all unique types in the DataFrame will be used. Defaults to None.
        bins (int, optional): The number of bins for the histogram. Defaults to 100.
        xrange (tuple, optional): The range of values for the x-axis. Defaults to None.
        separator_text (bool, optional): Whether to display the separator text. Defaults to False.
        mix_line (bool, optional): Whether to plot the decision tree threshold for the 'mix' type. Defaults to False.
        save (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        None
    """
    # plot distribution of delta_sim by type
    colors = {
        ("train", 0): "royalblue",
        ("val", 0): "limegreen",
        ("synth", 0): "lightsteelblue",
        ("mix", 0): "dodgerblue",
        ("train", 1): "orangered",
        ("val", 1): "magenta",
        ("synth", 1): "lightsalmon",
        ("mix", 1): "tomato",
        ("train", 2): "dimgrey",
        ("synth", 2): "darkgrey",
        ("val", 2): "silver",
        ("mix", 2): "slategray",
    }
    plot_type = df["type"].unique() if plot_type is None else plot_type
    plt.figure(figsize=(16, 8))
    legend_labels = list()
    _, bin_edges = np.histogram(df["delta_sim"], bins=bins)
    # overall median line
    for i, class_id in enumerate(set(df["actual_class"])):
        id = 0 if class_id == class_pairs[0] else 1
        for j, type_ in enumerate(set(plot_type)):
            x_type = df[(df["type"] == type_) & (df["actual_class"] == class_id)][
                "delta_sim"
            ]
            plt.hist(
                x_type,
                bins=bin_edges,
                alpha=0.5,
                label=(type_, class_name_pairs[id]),
                color=colors[(type_, id)],
            )
            legend_labels.append((type_, class_name_pairs[id]))
    if mix_line:
        df_mix = df[(df["type"] == "train") | (df["type"] == "synth")]
        df_mix = df_mix.copy()
        x_type = df_mix["delta_sim"]
        y_type = df_mix["actual_class"]
        y_type = [0 if y == class_pairs[0] else 1 for y in y_type]
        regressor = DecisionTreeRegressor(random_state=0, max_depth=1)
        regressor.fit(x_type.values.reshape(-1, 1), y_type)
        separator = regressor.tree_.threshold[0]
        plt.axvline(
            x=separator,
            color="rosybrown",
            linestyle="-.",
            alpha=0.6,
            label=f"Decision Tree Threshold = {separator:.3f} (mix)",
        )
        if separator_text:
            plt.text(
                separator * 1.005,
                5,
                f"{separator:.3f}",
                rotation=90,
                horizontalalignment="left",
                # fontsize=8,
            )
        legend_labels.append(f"Decision Tree Threshold = {separator:.3f} (mix)")
    for i, ds_type in enumerate(set(plot_type)):
        x_type = df[(df["type"] == ds_type)]["delta_sim"]
        y_type = df[(df["type"] == ds_type)]["actual_class"]
        y_type = [0 if y == class_pairs[0] else 1 for y in y_type]
        regressor = DecisionTreeRegressor(random_state=0, max_depth=1)
        regressor.fit(x_type.values.reshape(-1, 1), y_type)
        separator = regressor.tree_.threshold[0]
        plt.axvline(
            x=separator,
            color=colors[(ds_type, 2)],
            linestyle="--",
            alpha=0.6,
            label=f"Decision Tree Threshold = {separator:.3f} ({ds_type})",
        )
        if separator_text:
            plt.text(
                separator * 1.005,
                5,
                f"{separator:.3f}",
                rotation=90,
                horizontalalignment="left",
                # fontsize=8,
            )
        legend_labels.append(f"Decision Tree Threshold = {separator:.3f} ({ds_type})")
        for j, class_id in enumerate(set(df["actual_class"])):
            id = 0 if class_id == class_pairs[0] else 1
            x_type = df[(df["type"] == ds_type) & (df["actual_class"] == class_id)][
                "delta_sim"
            ]
            median = np.median(x_type)
            plt.axvline(
                x=median,
                color=colors[(ds_type, id)],
                linestyle="--",
                alpha=0.6,
                label=f"median = {median:.3f} ({ds_type}, {class_name_pairs[id]})",
            )
            if separator_text:
                plt.text(
                    median * 1.005,
                    5,
                    f"{median:.3f}",
                    rotation=90,
                    horizontalalignment="left",
                    # fontsize=8,
                )
            legend_labels.append(
                f"median = {median:.3f} ({ds_type}, {class_name_pairs[id]})"
            )
    if xrange is not None:
        plt.xticks(np.arange(xrange[0], xrange[1] + 0.1, 0.1))
    plt.title(
        f"Distribution of Inter-Class CLIP Score Difference between {class_name_pairs[0].lower()} vs {class_name_pairs[1].lower()}"
    )
    plt.xlabel("Inter-Class CLIP Score Difference")
    plt.ylabel("Frequency")
    plt.legend(legend_labels, prop={"size": 8})
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()
    return None


def viz_scatter_prompts(
    df,
    class_name_pairs,
    class_pairs,
    plot_type=None,
    xrange=None,
    prompts=None,
    sep_line=True,
    train_sep_line=False,
    mix_sep_line=False,
    save=None,
):
    """
    Visualizes scatter plots of cosine similarity between prompts using CLIP.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        class_name_pairs (list): A list of tuples representing the class name pairs.
        class_pairs (list): A list of tuples representing the class pairs.
        plot_type (list, optional): A list of plot types. Defaults to None.
        xrange (tuple, optional): A tuple representing the x-axis range. Defaults to None.
        prompts (list, optional): A list of prompts. Defaults to None.
        sep_line (bool, optional): Whether to plot a separation line. Defaults to True.
        train_sep_line (bool, optional): Whether to plot a separation line for the 'train' type. Defaults to False.
        mix_sep_line (bool, optional): Whether to plot a separation line for the 'synth' and 'train' types. Defaults to False.
        save (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        None
    """
    # Define a dictionary to map types to colors for easier customization
    colors = {
        ("train", 0): "royalblue",
        ("val", 0): "limegreen",
        ("synth", 0): "lightsteelblue",
        ("mix", 0): "dodgerblue",
        ("train", 1): "orangered",
        ("val", 1): "magenta",
        ("synth", 1): "lightsalmon",
        ("mix", 1): "tomato",
        ("train", 2): "dimgrey",
        ("synth", 2): "darkgrey",
        ("val", 2): "silver",
        ("mix", 2): "slategray",
    }
    plot_type = df["type"].unique() if plot_type is None else plot_type
    # Create a scatter plot
    plt.figure(figsize=(16, 8))  # Adjust figure size as desired
    for i, class_id in enumerate(set(df["actual_class"])):
        id = 0 if class_id == class_pairs[0] else 1
        for j, type_ in enumerate(set(plot_type)):  # Iterate through unique types
            # Filter data for the current type
            x_type = df[(df["type"] == type_) & (df["actual_class"] == class_id)][
                "sim_0"
            ]
            y_type = df[(df["type"] == type_) & (df["actual_class"] == class_id)][
                "sim_1"
            ]
            # Plot the data
            plt.scatter(
                x_type,
                y_type,
                label=(type_, class_name_pairs[id]),
                color=colors[(type_, id)],
                alpha=0.5,
                marker=".",
            )
    if train_sep_line and "train" in plot_type:
        df_train = df[df["type"] == "train"]
        clf = SVC(kernel="linear")
        clf.fit(df_train[["sim_0", "sim_1"]], df_train["actual_class"])
        x_line = np.linspace(
            min(df_train["sim_0"].min(), df_train["sim_1"].min()),
            max(df_train["sim_0"].max(), df_train["sim_1"].max()),
            100,
        )
        y_line = (-1 * clf.intercept_ - clf.coef_[0][0] * x_line) / clf.coef_[0][1]
        plt.plot(
            x_line,
            y_line,
            color="grey",
            linestyle="--",
            label="SVM_train",
        )
    if mix_sep_line and "synth" in plot_type and "train" in plot_type:
        df_mix = df[(df["type"] == "train") | (df["type"] == "synth")]
        df_mix = df_mix.copy()
        df_mix["type"] = "mix"
        clf = SVC(kernel="linear")
        clf.fit(df_mix[["sim_0", "sim_1"]], df_mix["actual_class"])
        x_line = np.linspace(
            min(df_mix["sim_0"].min(), df_mix["sim_1"].min()),
            max(df_mix["sim_0"].max(), df_mix["sim_1"].max()),
            100,
        )
        y_line = (-1 * clf.intercept_ - clf.coef_[0][0] * x_line) / clf.coef_[0][1]
        plt.plot(
            x_line,
            y_line,
            color="lightgrey",
            linestyle="--",
            label="SVM_mix",
        )
    if sep_line and (not train_sep_line and not mix_sep_line) and len(plot_type) == 1:
        df_type = df[df["type"] == plot_type[0]]
        clf = SVC(kernel="linear")
        clf.fit(df_type[["sim_0", "sim_1"]], df_type["actual_class"])
        x_line = np.linspace(
            min(df_type["sim_0"].min(), df_type["sim_1"].min()),
            max(df_type["sim_0"].max(), df_type["sim_1"].max()),
            100,
        )
        y_line = (-1 * clf.intercept_ - clf.coef_[0][0] * x_line) / clf.coef_[0][1]
        plt.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            label=f"SVM_{plot_type[0]}",
        )
    if xrange is not None:
        plt.xticks(np.arange(xrange[0], xrange[1] + 0.1, 0.1))
    if prompts is not None:
        axis_labels = prompts
    else:
        axis_labels = class_name_pairs
    # Add labels and title
    plt.xlabel(f"Cosine similarity to prompt: '{axis_labels[0]}'")
    plt.ylabel(f"Cosine similarity to prompt: '{axis_labels[1]}'")
    plt.title(
        f"CLIP cosine similarity between prompts: '{axis_labels[0].lower()}' vs '{axis_labels[1].lower()}'"
    )
    # Add a legend
    plt.legend()
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    # Show the plot
    plt.show()
    return None


def viz_hist_zone(
    df,
    class_name_pairs,
    class_pairs,
    plot_type=None,
    bins=100,
    xrange=None,
    manual_confusion_range=[],
    save=None,
    title="Distribution of Delta Similarity",
):
    """
    Visualizes the histogram of delta similarity values by type and class.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - class_name_pairs (list): A list of two class names.
    - class_pairs (list): A list of two class IDs.
    - plot_type (list, optional): A list of plot types. If None, all unique types in the dataframe will be used.
    - bins (int, optional): The number of bins for the histogram. Default is 100.
    - xrange (tuple, optional): The range of x-axis values. Default is None.
    - manual_confusion_range (list, optional): A list of three values representing the manual confusion range. Default is an empty list.
    - save (str, optional): The file path to save the plot. Default is None.
    - title (str, optional): The title of the plot. Default is "Distribution of Delta Similarity".

    Returns:
    - confusion_zone (dict): A dictionary containing the confusion zones for each plot type.

    """
    # plot distribution of delta_sim by type
    colors = {
        ("train", 0): "royalblue",
        ("val", 0): "limegreen",
        ("synth", 0): "lightsteelblue",
        ("mix", 0): "dodgerblue",
        ("train", 1): "orangered",
        ("val", 1): "magenta",
        ("synth", 1): "lightsalmon",
        ("mix", 1): "tomato",
        ("train", 2): "dimgrey",
        ("synth", 2): "darkgrey",
        ("val", 2): "silver",
        ("mix", 2): "slategray",
    }
    plot_type = df["type"].unique() if plot_type is None else plot_type
    plt.figure(figsize=(16, 8))
    legend_labels = list()
    _, bin_edges = np.histogram(df["delta_sim"], bins=bins)
    max_freq = 0
    for i, class_id in enumerate(set(df["actual_class"])):
        id = 0 if class_id == class_pairs[0] else 1
        for j, type_ in enumerate(set(plot_type)):
            x_type = df[(df["type"] == type_) & (df["actual_class"] == class_id)][
                "delta_sim"
            ]
            plt.hist(
                x_type,
                bins=bin_edges,
                alpha=0.5,
                label=(type_, class_name_pairs[id]),
                color=colors[(type_, id)],
            )
            legend_labels.append((type_, class_name_pairs[id]))
            freq, _ = np.histogram(x_type, bins=bin_edges)
            if max(freq) > max_freq:
                max_freq = max(freq)
    max_freq = max_freq + 10 - (max_freq % 10)
    confusion_zone = dict()
    for i, ds_type in enumerate(set(plot_type)):
        x_type = df[(df["type"] == ds_type)]["delta_sim"]
        y_type = df[(df["type"] == ds_type)]["actual_class"]
        y_type = [0 if y == class_pairs[0] else 1 for y in y_type]
        regressor = DecisionTreeRegressor(random_state=0, max_depth=1)
        regressor.fit(x_type.values.reshape(-1, 1), y_type)
        separator = regressor.tree_.threshold[0]
        if len(manual_confusion_range) < 3:
            min_confusion, max_confusion = None, None
            for j, class_id in enumerate(set(df["actual_class"])):
                id = 0 if class_id == class_pairs[0] else 1
                x_class_type = df[
                    (df["type"] == ds_type) & (df["actual_class"] == class_id)
                ]["delta_sim"]
                if id == 0:
                    confusions = [x for x in x_class_type if x < separator]
                else:
                    confusions = [x for x in x_class_type if x > separator]
                if len(confusions) != 0:
                    if min_confusion is None:
                        min_confusion = min(confusions)
                    elif min(confusions) < min_confusion:
                        min_confusion = min(confusions)
                    if max_confusion is None:
                        max_confusion = max(confusions)
                    elif max(confusions) > max_confusion:
                        max_confusion = max(confusions)
        else:
            separator = manual_confusion_range[1]
            min_confusion = manual_confusion_range[0]
            max_confusion = manual_confusion_range[2]
            confusions = []
        plt.axvline(
            x=separator,
            color=colors[(ds_type, 2)],
            linestyle="--",
            label=f"{ds_type} separator",
        )
        if len(confusions) != 0 or len(manual_confusion_range) != 0:
            plt.fill_betweenx(
                [0, max_freq],
                min_confusion,
                max_confusion,
                color=colors[(ds_type, 2)],
                alpha=0.2,
            )
        if len(confusions) != 0:
            confusion_zone[ds_type] = (min_confusion, separator, max_confusion)
            legend_labels.append(
                f"{ds_type} confusion zone: {min_confusion:.3f} - {max_confusion:.3f}"
            )
        else:
            confusion_zone[ds_type] = (None, separator, None)
        legend_labels.append(f"{ds_type} separator: {separator:.3f}")
    if xrange is not None:
        plt.xticks(np.arange(xrange[0], xrange[1] + 0.1, 0.1))
    plt.ylim(0, max_freq)
    plt.title(title)
    plt.xlabel(
        f"Delta Similarity (CLIP Score) between {class_name_pairs[0].lower()} vs {class_name_pairs[1].lower()}"
    )
    plt.ylabel("Frequency")
    plt.legend(legend_labels)
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()
    return confusion_zone


def viz_hist_clf(
    df,
    class_name_pairs,
    class_pairs,
    plot_type=None,
    bins=100,
    xrange=None,
    manual_confusion_range=[],
    suffix="",
    save=None,
):
    """
    Visualizes the histogram of the delta_sim values for different classes and types.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        class_name_pairs (list): A list of class name pairs.
        class_pairs (list): A list of class pairs.
        plot_type (list, optional): A list of plot types. If not provided, all unique types in the DataFrame will be used. Defaults to None.
        bins (int, optional): The number of bins for the histogram. Defaults to 100.
        xrange (tuple, optional): The range of values for the x-axis. Defaults to None.
        manual_confusion_range (list, optional): A list containing the manual confusion range. Defaults to an empty list.
        suffix (str, optional): A suffix to be added to the column name for correctness. Defaults to an empty string.
        save (str, optional): The file path to save the plot. Defaults to None.

    Returns:
        dict: A dictionary containing classification statistics for each type.
            The dictionary has the following structure:
            {
                type_1: (misclassified_stats, val_accuracy, num_confusions, num_misclassified, num_type_1),
                type_2: (misclassified_stats, val_accuracy, num_confusions, num_misclassified, num_type_2),
                ...
            }
            - misclassified_stats: A tuple containing the minimum, median, and maximum values of misclassified data.
            - val_accuracy: The validation accuracy for the type.
            - num_confusions: The number of misclassified data within the confusion zone.
            - num_misclassified: The total number of misclassified data.
            - num_type: The total number of data for the type.
    """
    # note: requires classifiers analysis
    # plot distribution of delta_sim by type
    colors = {
        ("train", 0): "royalblue",
        ("val", 0): "limegreen",
        ("synth", 0): "lightsteelblue",
        ("mix", 0): "dodgerblue",
        ("train", 1): "orangered",
        ("val", 1): "magenta",
        ("synth", 1): "lightsalmon",
        ("mix", 1): "tomato",
        ("train", 2): "dimgrey",
        ("synth", 2): "darkgrey",
        ("val", 2): "silver",
        ("mix", 2): "slategray",
    }
    plot_type = df["type"].unique() if plot_type is None else plot_type
    plt.figure(figsize=(16, 8))
    legend_labels = list()
    max_freq = 0
    _, bin_edges = np.histogram(df["delta_sim"], bins=bins)
    for i, class_id in enumerate(set(df["actual_class"])):
        id = 0 if class_id == class_pairs[0] else 1
        for j, type_ in enumerate(set(plot_type)):
            x_type = df[(df["type"] == type_) & (df["actual_class"] == class_id)][
                "delta_sim"
            ]
            plt.hist(
                x_type,
                bins=bin_edges,
                alpha=0.5,
                label=(type_, class_name_pairs[id]),
                color=colors[(type_, id)],
            )
            legend_labels.append((type_, class_name_pairs[id]))
            freq, _ = np.histogram(x_type, bins=bin_edges)
            if max(freq) > max_freq:
                max_freq = max(freq)
    max_freq = max_freq + 10 - (max_freq % 10)
    classification_stats = dict()
    for i, ds_type in enumerate(set(plot_type)):
        x_type = df[(df["type"] == ds_type)]["delta_sim"]
        y_type = df[(df["type"] == ds_type)]["actual_class"]
        y_type = [0 if y == class_pairs[0] else 1 for y in y_type]
        regressor = DecisionTreeRegressor(random_state=0, max_depth=1)
        regressor.fit(x_type.values.reshape(-1, 1), y_type)
        separator = regressor.tree_.threshold[0]
        if len(manual_confusion_range) < 3:
            min_confusion, max_confusion = None, None
            for j, class_id in enumerate(set(df["actual_class"])):
                id = 0 if class_id == class_pairs[0] else 1
                x_class_type = df[
                    (df["type"] == ds_type) & (df["actual_class"] == class_id)
                ]["delta_sim"]
                if id == 0:
                    confusions = [x for x in x_class_type if x < separator]
                else:
                    confusions = [x for x in x_class_type if x > separator]
                if len(confusions) != 0:
                    if min_confusion is None and len(confusions):
                        min_confusion = min(confusions)
                    elif min(confusions) < min_confusion:
                        min_confusion = min(confusions)
                    if max_confusion is None:
                        max_confusion = max(confusions)
                    elif max(confusions) > max_confusion:
                        max_confusion = max(confusions)
        else:
            separator = manual_confusion_range[1]
            min_confusion = manual_confusion_range[0]
            max_confusion = manual_confusion_range[2]
            confusions = []
        plt.axvline(
            x=separator,
            color=colors[(ds_type, 2)],
            linestyle="--",
            label=f"{ds_type} separator",
        )
        if len(confusions) != 0 or len(manual_confusion_range) != 0:
            plt.fill_betweenx(
                [0, max_freq],
                min_confusion,
                max_confusion,
                color=colors[(ds_type, 2)],
                alpha=0.2,
            )

        # plot clf incorrects as overlay
        x_misclassified = df[(df[f"correct{suffix}"] == 0) & (df["type"] == ds_type)][
            "delta_sim"
        ]
        if len(x_misclassified) != 0:
            misclassified_stats = (
                min(x_misclassified),
                np.median(x_misclassified),
                max(x_misclassified),
            )
        else:
            misclassified_stats = (None, None, None)
        # count of misclassified within confusion zone
        if len(confusions) != 0:
            x_confusion = [
                x for x in x_misclassified if min_confusion < x < max_confusion
            ]
            legend_labels.append(
                f"{ds_type} confusion zone: {min_confusion:.3f} - {max_confusion:.3f}"
            )
        else:
            x_confusion = []

        val_accuracy = (
            df[(df[f"correct{suffix}"] == 1) & (df["type"] == ds_type)].shape[0]
            / df[df["type"] == ds_type].shape[0]
        )

        plt.hist(
            x_misclassified,
            bins=bin_edges,
            alpha=1,
            label=(f"{ds_type} misclassified", "misclassified"),
            color="red",
        )
        classification_stats[ds_type] = (
            misclassified_stats,
            val_accuracy,
            len(x_confusion),
            len(x_misclassified),
            len(x_type),
        )
        legend_labels.append(f"{ds_type} separator: {separator:.3f}")

    if xrange is not None:
        plt.xticks(np.arange(xrange[0], xrange[1] + 0.1, 0.1))
    plt.ylim(0, max_freq)
    plt.title(
        f"Distribution of Inter-Class CLIP Score Difference between {class_name_pairs[0].lower()} vs {class_name_pairs[1].lower()}"
    )
    plt.xlabel("Inter-Class CLIP Score Difference")
    plt.ylabel("Frequency")
    plt.legend(legend_labels, prop={"size": 8})
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()
    return classification_stats


def analyze_confusion_zone(df, class_pairs=None, save_csv=None):
    """
    Analyzes the confusion zone for each data type in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        class_pairs (list, optional): The list of class pairs to consider. If not provided, all unique classes in the DataFrame will be considered. Defaults to None.
        save_csv (str, optional): The file path to save the results as a CSV file. If not provided, the results will be returned as a DataFrame. Defaults to None.

    Returns:
        pandas.DataFrame or None: If save_csv is None, returns a DataFrame containing the confusion zone for each data type. If save_csv is provided, saves the results as a CSV file and returns None.
    """
    confusion_zone = dict()
    if class_pairs is None:
        class_pairs = sorted(list(df["actual_class"].unique()))
    for i, ds_type in enumerate(df["type"].unique()):
        x_type = df[(df["type"] == ds_type)]["delta_sim"]
        y_type = df[(df["type"] == ds_type)]["actual_class"]
        y_type = [0 if y == class_pairs[0] else 1 for y in y_type]
        regressor = DecisionTreeRegressor(random_state=0, max_depth=1)
        regressor.fit(x_type.values.reshape(-1, 1), y_type)
        separator = regressor.tree_.threshold[0]

        min_confusion, max_confusion = None, None
        for j, class_id in enumerate(set(df["actual_class"])):
            id = 0 if class_id == class_pairs[0] else 1
            x_class_type = df[
                (df["type"] == ds_type) & (df["actual_class"] == class_id)
            ]["delta_sim"]
            if id == 0:
                confusions = [x for x in x_class_type if x < separator]
            else:
                confusions = [x for x in x_class_type if x > separator]
            if len(confusions) != 0:
                if min_confusion is None:
                    min_confusion = min(confusions)
                elif min(confusions) < min_confusion:
                    min_confusion = min(confusions)
                if max_confusion is None:
                    max_confusion = max(confusions)
                elif max(confusions) > max_confusion:
                    max_confusion = max(confusions)
        confusion_zone[ds_type] = (min_confusion, separator, max_confusion)
    if save_csv is None:
        return pd.DataFrame.from_dict(
            confusion_zone,
            orient="index",
            columns=["min_confusion", "separator", "max_confusion"],
        )
    else:
        pd.DataFrame.from_dict(
            confusion_zone,
            orient="index",
            columns=["min_confusion", "separator", "max_confusion"],
        ).to_csv(save_csv, index=True)


def analyze_clf_stats(df, class_pairs=None, save_csv=None):
    """
    Analyzes classifier statistics based on the input dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe containing classifier analysis data.
        class_pairs (list, optional): List of class pairs to consider. Defaults to None.
        save_csv (str, optional): Filepath to save the analysis results as a CSV file. Defaults to None.

    Returns:
        pandas.DataFrame: The analysis results as a dataframe if save_csv is None, otherwise None.

    """
    # requires classifier analysis in dataframe
    classification_stats = list()
    if class_pairs is None:
        class_pairs = sorted(list(df["actual_class"].unique()))
    for i, ds_type in enumerate(df["type"].unique()):
        x_type = df[(df["type"] == ds_type)]["delta_sim"]
        y_type = df[(df["type"] == ds_type)]["actual_class"]
        y_type = [0 if y == class_pairs[0] else 1 for y in y_type]
        regressor = DecisionTreeRegressor(random_state=0, max_depth=1)
        regressor.fit(x_type.values.reshape(-1, 1), y_type)
        separator = regressor.tree_.threshold[0]

        min_confusion, max_confusion = None, None
        for j, class_id in enumerate(set(df["actual_class"])):
            id = 0 if class_id == class_pairs[0] else 1
            x_class_type = df[
                (df["type"] == ds_type) & (df["actual_class"] == class_id)
            ]["delta_sim"]
            if id == 0:
                confusions = [x for x in x_class_type if x < separator]
            else:
                confusions = [x for x in x_class_type if x > separator]
            if len(confusions) != 0:
                if min_confusion is None:
                    min_confusion = min(confusions)
                elif min(confusions) < min_confusion:
                    min_confusion = min(confusions)
                if max_confusion is None:
                    max_confusion = max(confusions)
                elif max(confusions) > max_confusion:
                    max_confusion = max(confusions)
        if "correct_mix" in df.columns:
            suffixes = ("", "_mix")
        else:
            suffixes = [""]
        for suffix in suffixes:
            x_misclassified = df[
                (df[f"correct{suffix}"] == 0) & (df["type"] == ds_type)
            ]["delta_sim"]
            if len(x_misclassified) != 0:
                misclassified_stats = (
                    min(x_misclassified),
                    np.median(x_misclassified),
                    max(x_misclassified),
                )
            else:
                misclassified_stats = (None, None, None)
            # count of misclassified within confusion zone
            if len(confusions) != 0:
                x_confusion = [
                    x for x in x_misclassified if min_confusion < x < max_confusion
                ]
            else:
                x_confusion = []

            accuracy = (
                df[(df[f"correct{suffix}"] == 1) & (df["type"] == ds_type)].shape[0]
                / df[df["type"] == ds_type].shape[0]
            )
            if suffix == "":
                clf_type = "base"
            else:
                clf_type = "mix"
            classification_stats.append(
                (
                    ds_type,
                    clf_type,
                    misclassified_stats[0],
                    misclassified_stats[1],
                    misclassified_stats[2],
                    accuracy,
                    len(x_confusion),
                    len(x_misclassified),
                    len(x_type),
                    min_confusion,
                    separator,
                    max_confusion,
                )
            )
    if save_csv is None:
        return pd.DataFrame.from_records(
            classification_stats,
            columns=[
                "dataset",
                "classifier_type",
                "min_classified",
                "median_classified",
                "max_classified",
                "accuracy",
                "misclassified_in_confusion_count",
                "misclassified_count",
                "total_count",
                "min_confusion",
                "separator",
                "max_confusion",
            ],
        )
    else:
        pd.DataFrame.from_records(
            classification_stats,
            columns=[
                "dataset",
                "classifier_type",
                "min_classified",
                "median_classified",
                "max_classified",
                "accuracy",
                "misclassified_in_confusion_count",
                "misclassified_count",
                "total_count",
                "min_confusion",
                "separator",
                "max_confusion",
            ],
        ).to_csv(save_csv, index=True)


def getStats(df, metadata_path, synth_path):
    """
    Calculate and save statistics based on the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        metadata_path (str): The path to save the statistics files.
        synth_path (str): The synthetic path used for file naming.

    Raises:
        ValueError: If `metadata_path` or `synth_path` is None.

    Returns:
        None
    """
    if metadata_path is None or synth_path is None:
        raise ValueError("metadata_path and synth_path must be provided")
    df.groupby(["type", "actual_classname"])["delta_sim"].agg(
        ["mean", "median", "min", "max", "std", "count"]
    ).to_csv(f"{metadata_path}/stats_sim_by_class_{synth_path.replace('/','_')}.csv")
    df.groupby(["type"])["delta_sim"].agg(
        ["mean", "median", "min", "max", "std", "count"]
    ).to_csv(f"{metadata_path}/stats_sim_{synth_path.replace('/','_')}.csv")
    df.describe().to_csv(
        f"{metadata_path}/stats_general_{synth_path.replace('/','_')}.csv"
    )
    df_mix = df[(df["type"] == "train") | (df["type"] == "synth")].copy()
    df_mix.groupby(["actual_classname"])["delta_sim"].agg(
        ["mean", "median", "min", "max", "std", "count"]
    ).to_csv(
        f"{metadata_path}/stats_midelta_sim_by_class_{synth_path.replace('/','_')}.csv"
    )
    df_mix["delta_sim"].agg(
        ["mean", "median", "min", "max", "std", "count"]
    ).to_frame().to_csv(
        f"{metadata_path}/stats_midelta_sim_{synth_path.replace('/','_')}.csv"
    )
    print(f"Stats saved to {metadata_path}")


def analyze_clip_trace(
    prompts,
    df,
    class_pairs,
    clip_model_path="openai/clip-vit-large-patch14",
    device="cuda",
):
    """
    Analyzes the clip trace for a given set of prompts and dataframe.

    Args:
        prompts (str or List[str]): The prompts to be used for analysis.
        df (pandas.DataFrame): The dataframe containing the data to be analyzed.
        class_pairs (dict): A dictionary mapping class indices to class names.
        clip_model_path (str, optional): The path to the CLIP model. Defaults to "openai/clip-vit-large-patch14".
        device (str, optional): The device to run the analysis on. Defaults to "cuda".

    Returns:
        pandas.DataFrame: A dataframe containing the analysis results.
    """
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    clip_dict = {
        "dataset": list(),
        "type": list(),
        "file_path": list(),
        "file_name": list(),
        "actual_class": list(),
        "actual_classname": list(),
        "clip_pred_class": list(),
        "clip_pred_classname": list(),
        "cos_sim": list(),
        "delta_sim": list(),
        "sim_0": list(),
        "sim_1": list(),
    }
    with torch.inference_mode():
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            img_path = row["output_file_path"]
            # read image
            img = Image.open(img_path)
            img = T.PILToTensor()(img)
            inputs = clip_processor(
                text=prompts, padding=True, images=img, return_tensors="pt"
            ).to(device)
            scores = (
                clip_model(**inputs).logits_per_image.detach().cpu().numpy().squeeze()
            )
            pred = np.argmax(scores).item()
            actual_class = row["class_id"]
            pred_class = class_pairs[pred]
            pred_score = scores[pred]
            pred_class_name = classes.IMAGENET2012_CLASSES[pred_class]
            clip_dict["dataset"].append("data/trace")
            clip_dict["type"].append("synth")
            clip_dict["file_path"].append(img_path)
            clip_dict["file_name"].append(os.path.basename(img_path))
            clip_dict["actual_class"].append(actual_class)
            clip_dict["actual_classname"].append(
                classes.IMAGENET2012_CLASSES[actual_class]
            )
            clip_dict["clip_pred_class"].append(pred_class)
            clip_dict["clip_pred_classname"].append(pred_class_name)
            clip_dict["cos_sim"].append(pred_score.item())
            clip_dict["delta_sim"].append((scores)[0].item() - (scores)[1].item())
            clip_dict["sim_0"].append(scores[0].item())
            clip_dict["sim_1"].append(scores[1].item())
    return pd.DataFrame.from_dict(clip_dict)


def symmetrize_steps(df, steps=16):
    """
    Symmetrizes the steps in the given DataFrame.

    e.g. total interpolation steps: 16
    steps from 8 to 15 are 'shifted' and 'flipped' to 8 to 0

    in:  0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15
    out: 0 1 2 3 4 5 6 7 | 7 6  5  4  3  2  1  0

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the steps.

    Returns:
    pandas.DataFrame: The DataFrame with symmetrized steps.
    """

    if steps % 2 != 0:
        raise ValueError("steps must be an even number")

    middle = steps // 2

    dfr = df.copy()
    dfr["interpolation_step"] = dfr["interpolation_step"] - (steps - 1)
    dfr["interpolation_step"] = dfr["interpolation_step"].abs()
    dfr = dfr[dfr["interpolation_step"] < middle]

    dfl = df.copy()
    dfl = dfl[dfl["interpolation_step"] < middle]
    assert dfl.shape[0] + dfr.shape[0] - df.shape[0] == 0
    df = pd.concat([dfl, dfr], axis=0)
    return df


def computeSimilarityByInterpolation(
    df,
    clip_model_path="openai/clip-vit-large-patch14",
    device="cuda",
    ssim_resize=(256, 256),
    save=None,
):
    """
    Computes similarity scores between real and synthetic images using interpolation.

    Args:
        df (pandas.DataFrame): The input DataFrame containing image file paths and prompt texts.
        clip_model_path (str, optional): The path or name of the CLIP model to use. Defaults to "openai/clip-vit-large-patch14".
        device (str, optional): The device to run the computation on. Defaults to "cuda".
        ssim_resize (tuple, optional): The size to resize the images for SSIM calculation. Defaults to (256, 256).
        save (str, optional): The file path to save the resulting DataFrame. Defaults to None.

    Returns:
        pandas.DataFrame: The DataFrame with similarity scores computed and optionally saved.

    """
    df = df.copy()
    df["text_real_sim"] = None
    df["text_synth_sim"] = None
    df["image_real_synth_sim"] = None
    df["ssim"] = None
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    with torch.inference_mode():
        for i, rows in tqdm(df.iterrows(), total=df.shape[0]):
            # CLIP Similarity
            real_img = T.PILToTensor()(Image.open(rows["input_file_path"])).unsqueeze(0)
            # if real_img is grayscle, convert to RGB
            if real_img.shape[1] == 1:
                real_img = real_img.repeat(1, 3, 1, 1)
            prompts = rows["pos_prompt_text"]
            real_inputs = clip_processor(
                text=prompts, padding=True, images=real_img, return_tensors="pt"
            ).to(device)
            real_output = clip_model(**real_inputs)
            real_scores = real_output.logits_per_image.detach().cpu().numpy().squeeze()
            df.at[i, "text_real_sim"] = real_scores
            synth_img = T.PILToTensor()(Image.open(rows["output_file_path"])).unsqueeze(
                0
            )
            synth_inputs = clip_processor(
                text=prompts, padding=True, images=synth_img, return_tensors="pt"
            ).to(device)
            synth_output = clip_model(**synth_inputs)
            synth_scores = (
                synth_output.logits_per_image.detach().cpu().numpy().squeeze()
            )
            df.at[i, "text_synth_sim"] = synth_scores
            cos_sim_imgs = (
                cos_sim(real_output.image_embeds, synth_output.image_embeds)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
                * 100
            )
            df.at[i, "image_real_synth_sim"] = cos_sim_imgs

            # SSIM
            resize = ssim_resize
            real_img = T.Compose([T.ToTensor(), T.Resize(resize, antialias=True)])(
                Image.open(rows["input_file_path"])
            ).unsqueeze(0)
            real_img = torch.nn.functional.interpolate(real_img, size=(256, 256))
            # if real_img is grayscle, convert to RGB
            if real_img.shape[1] == 1:
                real_img = real_img.repeat(1, 3, 1, 1)
            synth_img = T.Compose([T.ToTensor(), T.Resize(resize, antialias=True)])(
                Image.open(rows["output_file_path"])
            ).unsqueeze(0)
            synth_img = torch.nn.functional.interpolate(synth_img, size=(256, 256))
            ssim_score = ssim(real_img, synth_img).item()
            df.at[i, "ssim"] = ssim_score
    df["ssim"] = df["ssim"].astype(float)
    df = symmetrize_steps(df)
    if save is not None:
        df.to_csv(save, index=False)
    return df


def plot_kde(df, ds_name, prompt_format, save=None, palette=None):
    """
    Plots a kernel density estimation (KDE) plot for CLIP scores.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        ds_name (str): The name of the dataset.
        prompt_format (str): The format of the prompt.
        save (str, optional): The file path to save the plot. Defaults to None.
        palette (str, optional): The color palette for the plot. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The KDE plot.

    """
    # get unique for each column
    subset = df.copy()
    # ds_name = "Imagenette Original"
    # prompt_format = "a photo of a <class_name>"
    classpairs = sorted(list(subset["actual_class"].unique()))
    classnames = dict()
    prompts = dict()
    for i in classpairs:
        # lookup i in actual_classname
        classname = subset[subset["actual_class"] == i]["actual_classname"].unique()
        classnames[i] = classname[0]
        prompts[i] = prompt_format.replace("<class_name>", classname[0])
    subset.rename({"actual_classname": "Class"}, axis=1, inplace=True)
    kdeplot = sns.kdeplot(
        data=subset,
        x="sim_0",
        y="sim_1",
        hue="Class",
        palette=palette,
        # fill=True,
    )
    kdeplot.set_title(
        f"CLIP Score for {ds_name}",
        wrap=True,
    )
    kdeplot.set_xlabel(f"Prompt: '{prompts[classpairs[0]]}'")
    kdeplot.set_ylabel(f"Prompt: '{prompts[classpairs[1]]}'")
    if save is not None:
        kdeplot.figure.savefig(save)
    return kdeplot


def getRegPlot(df, x, y, title, xlabel, ylabel, save):
    """
    Generate a regression plot using seaborn.

    Parameters:
    - df: DataFrame
        The input DataFrame containing the data.
    - x: str
        The name of the column to be plotted on the x-axis.
    - y: str
        The name of the column to be plotted on the y-axis.
    - title: str
        The title of the plot.
    - xlabel: str
        The label for the x-axis.
    - ylabel: str
        The label for the y-axis.
    - save: str, optional
        The file path to save the plot. If not provided, the plot will not be saved.

    Returns:
    - ax: matplotlib.axes.Axes
        The generated regression plot.

    """
    ax = sns.regplot(
        data=df,
        x=x,
        y=y,
        x_estimator=np.mean,
        order=1,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if save is not None:
        ax.get_figure().savefig(save)
    return ax
