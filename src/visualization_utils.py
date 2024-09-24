import numpy as np
import pandas as pd
from scipy.stats import skew, mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import seaborn as sns
from adjustText import adjust_text
import fastcluster  # Efficient clustering
from scipy.cluster.hierarchy import leaves_list
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines import KaplanMeierFitter


def save_svg(output_path):
    """
    Save the current matplotlib figure as an SVG file if an output path is provided.

    Args:
        output_path (str): Path to save the SVG file.

    Returns:
        None
    """
    if output_path is not None:
        plt.savefig(output_path, format="svg", dpi=300)


def plot_distribution(
    data,
    column,
    xlabel="",
    plot_type="auto",
    bins=30,
    axvline=None,
    axvline_text=None,
    text_offset=(0, 0),
    xticks=None,
    title=None,
    output_path=None,
):
    """
    Plot the distribution of a specified column using either histplot or countplot.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data to be plotted.
        column (str): The column name to be plotted.
        xlabel (str): Label for the x-axis.
        plot_type (str): 'auto', 'histplot', or 'countplot'. If 'auto', decides based on data type.
        bins (int): Number of bins for histplot (only used for continuous data).
        axvline (float, optional): Add a vertical line at this x-value.
        axvline_text (str, optional): Text to display near the vertical line.
        text_offset (tuple, optional): Position offset for the text relative to the axvline.
        xticks (list, optional): Labels for x-axis ticks.
        title (str, optional): Figure title.
        output_path (str, optional): Path to save the figure as an SVG file.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))

    # Decide plot type based on column data type if 'auto' is passed
    if plot_type == "auto":
        if pd.api.types.is_numeric_dtype(data[column]):
            plot_type = "histplot"
        else:
            plot_type = "countplot"

    # Create the appropriate plot
    if plot_type == "histplot":
        sns.histplot(data[column], kde=True, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
    elif plot_type == "countplot":
        sns.countplot(x=column, data=data)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        if xticks:
            plt.xticks(ticks=range(len(xticks)), labels=xticks)

    # Add axvline if specified
    if axvline is not None:
        plt.axvline(x=axvline, color="red", linestyle="--")

        # Add corresponding text
        if axvline_text:
            text = plt.text(
                axvline + text_offset[0],
                plt.ylim()[1] * 0.95 + text_offset[1],
                axvline_text,
                color="red",
                fontsize=10,
                ha="left",
            )
            adjust_text([text])  # Adjust text to avoid overlap

    plt.title(title)

    # Save plot as an SVG file if output_path is provided
    save_svg(output_path)

    plt.show()


def plot_volcano(
    biomarker_cols,
    log2_fold_changes,
    corrected_p_values,
    fc_threshold=1,
    pval_threshold=0.05,
    output_path=None,
):
    """
    Create a volcano plot to visualize biomarker associations.

    Args:
        biomarker_cols (list): List of biomarker names.
        log2_fold_changes (array-like): Log2 fold change values.
        corrected_p_values (array-like): P-values after correction.
        fc_threshold (float): Fold change threshold for significance.
        pval_threshold (float): P-value threshold for significance.
        output_path (str, optional): File path to save the plot as an SVG.

    Returns:
        None
    """
    neg_log_p_values = -np.log10(corrected_p_values)
    volcano_data = pd.DataFrame(
        {
            "biomarker": biomarker_cols,
            "log2FC": log2_fold_changes,
            "-log10(p-value)": neg_log_p_values,
        }
    )
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=volcano_data,
        x="log2FC",
        y="-log10(p-value)",
        color="grey",
        edgecolor=None,
        s=8,
        alpha=0.2,
    )
    plt.grid(False)
    pval_threshold_line = -np.log10(pval_threshold)
    sig = volcano_data[
        (volcano_data["-log10(p-value)"] > pval_threshold_line)
        & (abs(volcano_data["log2FC"]) > fc_threshold)
    ]
    sns.scatterplot(
        data=sig,
        x="log2FC",
        y="-log10(p-value)",
        color="red",
        edgecolor=None,
        s=8,
    )
    plt.axhline(y=pval_threshold_line, color="grey", linestyle="--", alpha=0.5)
    plt.axvline(x=fc_threshold, color="grey", linestyle="--", alpha=0.5)
    plt.axvline(x=-fc_threshold, color="grey", linestyle="--", alpha=0.5)
    texts = []
    for i in range(sig.shape[0]):
        texts.append(
            plt.text(
                sig["log2FC"].iloc[i],
                sig["-log10(p-value)"].iloc[i],
                sig["biomarker"].iloc[i],
                fontsize=8,
                ha="right",
                color="black",
            )
        )
    adjust_text(texts)
    plt.title("Volcano Plot: Biomarker Association with Incident Diabetes")
    plt.xlabel("Log2 Fold Change (Odds Ratio)")
    plt.ylabel("-Log10(p-value)")

    # Save plot as an SVG file if output_path is provided
    save_svg(output_path)

    plt.show()


def plot_multicollinearity_heatmap(df, output_path=None):
    """
    Generate a clustered correlation heatmap to assess multicollinearity among biomarkers.

    Args:
        df (pandas.DataFrame): Dataset containing biomarkers and predictors.
        output_path (str, optional): Path to save the figure as an SVG file.

    Returns:
        None
    """
    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Perform hierarchical clustering using Ward's method
    linkage_matrix = fastcluster.linkage_vector(
        correlation_matrix, method="ward"
    )

    # Reorder the correlation matrix based on the clustering order
    col_order = leaves_list(linkage_matrix)
    clustered_corr_matrix = correlation_matrix.iloc[col_order, col_order]

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Generate a heatmap with the clustered correlation matrix
    sns.heatmap(
        clustered_corr_matrix,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 6},
    )

    # Set a new title focusing on multicollinearity
    plt.title("Multicollinearity Analysis of Biomarkers", size=12)

    # Save the figure if a save path is provided
    save_svg(output_path)

    # Display the plot
    plt.show()


def plot_roc_curves(prediction_probs, y_test, output_path=None):
    """
    Plot ROC curve comparison for multiple models.

    Args:
        prediction_probs (dict): Dictionary of predicted probabilities for each model.
        y_test (array-like): True binary labels.
        output_path (str, optional): Path to save the figure as an SVG file.

    Returns:
        None
    """
    plt.figure(figsize=(8, 8))

    for model_name, y_pred_prob in prediction_probs.items():
        # Calculate AUC
        auc = roc_auc_score(y_test, y_pred_prob)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

    # Plot random baseline
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("ROC Curve Comparison of Models")
    plt.legend(loc="lower right")
    plt.grid(False)

    # Save plot as an SVG file if output_path is provided
    save_svg(output_path)

    plt.show()


def plot_feature_importance(
    selected_features,
    feature_importance,
    importance_metric=None,
    title=None,
    top_N=20,
    output_path=None,
):
    """
    Plot the top N feature importances based on the given importance metric.

    Args:
        selected_features (list): Features to be plotted.
        feature_importance (array-like): Importance values corresponding to the features.
        importance_metric (str, optional): The metric used for feature importance.
        title (str, optional): The title for the plot.
        top_N (int, optional): The number of top features to plot (default: 20).
        output_path (str, optional): Path to save the figure as an SVG file.

    Returns:
        None
    """
    importance_df = (
        pd.DataFrame(
            {"Feature": selected_features, "Importance": feature_importance}
        )
        .sort_values(by="Importance", ascending=False)
        .head(top_N)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(
        importance_df["Feature"], importance_df["Importance"], color="skyblue"
    )
    plt.xlabel(f"Importance ({importance_metric})")
    plt.ylabel("Features")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(False)

    # Save plot as an SVG file if output_path is provided
    save_svg(output_path)

    plt.show()


def plot_biomarker_skewness(dfs_dict, output_path=None):
    """
    Plot the skewness distribution of biomarkers for multiple DataFrames.

    Args:
        dfs_dict (dict): Dictionary where keys are labels (e.g., 'Raw Data') and values are DataFrames.
        output_path (str, optional): Path to save the figure as an SVG file.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))

    # Loop through the dictionary to calculate and plot skewness
    for label, df in dfs_dict.items():
        skewness = df.apply(lambda x: skew(x.dropna()))
        skewness_counts = len(skewness[abs(skewness) > 1])
        skewness_percentage = (skewness_counts / len(skewness)) * 100

        sns.kdeplot(
            skewness,
            label=f"{label}: {skewness_counts} ({skewness_percentage:.2f}%) skewed",
            alpha=0.6,
        )

    # Mark skewness boundaries
    plt.axvline(x=1, color="red", linestyle="--")
    plt.axvline(x=-1, color="red", linestyle="--")
    plt.text(
        3,
        plt.ylim()[1] * 0.95,
        "Skewness = ±1",
        color="red",
        fontsize=10,
        ha="left",
    )

    plt.title("Biomarker Skewness Distribution Across DataFrames")
    plt.xlabel("Skewness")
    plt.ylabel("Frequency")
    plt.legend()

    # Save plot as an SVG file if output_path is provided
    save_svg(output_path)

    plt.show()


def plot_cumulative_risk_curve(
    y_pred_prob,
    y_test,
    follow_up_time,
    group_labels,
    labels,
    xlim,
    target_cumulative=0.5,
    tick_width=2,
    ylim=1,
    ax=None,
):
    """
    Plot a Kaplan-Meier curve based on predicted probabilities, true labels,
    follow-up time, and group labels.

    Args:
        y_pred_prob (array-like): Predicted probabilities for survival.
        y_test (array-like): True binary outcomes (0: no event, 1: event).
        follow_up_time (array-like): Follow-up time until event or censoring.
        group_labels (array-like): Group labels for categorizing y_pred_prob into percentiles or other groups.
        labels (list): Labels for the custom percentiles.
        xlim (int): Limit for x-axis (time in years).
        target_cumulative (float, optional): Target cumulative percentage to highlight. Defaults to 0.5.
        tick_width (int, optional): Spacing between ticks on the x-axis. Defaults to 2.
        ylim (int, optional): Limit for y-axis (cumulative risk). Defaults to 1.
        ax (matplotlib.axes.Axes, optional): Axis on which to plot. If None, a new figure is created.

    Returns:
        None
    """

    # Define custom colormap
    colors = ["green", "#FDC010", "red"]
    cmap_name = "risk_spectrum"
    cm = mcolors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=len(labels)
    )

    # Convert input arrays into a dataframe for easier processing
    df = pd.DataFrame(
        {
            "y_pred_prob": y_pred_prob,
            "y_test": y_test,
            "follow_up_time": follow_up_time,
            "group_labels": group_labels,
        }
    )

    kmf = KaplanMeierFitter()

    # Prepare axis (no new figure here)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.grid(False)  # Disable the grid
    texts = []  # To store the text objects for cumulative risk

    # Loop through the custom-defined groups using the passed `labels` order
    for i, group in enumerate(labels):
        idx = df["group_labels"] == group
        if idx.sum() == 0:  # If no data for this group, skip it
            continue

        kmf.fit(
            df.loc[idx, "follow_up_time"],
            event_observed=df.loc[idx, "y_test"] == 1,
            label=group,
        )

        # Cumulative risk and confidence intervals
        cumulative_risk = 1 - kmf.survival_function_
        ci_upper = 1 - kmf.confidence_interval_[f"{group}_lower_0.95"]
        ci_lower = 1 - kmf.confidence_interval_[f"{group}_upper_0.95"]

        # Assign colors from colormap
        color = cm(i % cm.N)

        # Plot survival curve
        ax.step(
            cumulative_risk.index,
            cumulative_risk[group],
            where="post",
            label=group,
            color=color,
        )
        ax.fill_between(
            ci_upper.index,
            ci_lower,
            ci_upper,
            step="post",
            color=color,
            alpha=0.1,
        )

        # Annotate the cumulative risk at the end of the curve (right tail)
        last_time_point = cumulative_risk.index[-1]
        last_risk = cumulative_risk[group].iloc[-1]
        text = ax.text(
            last_time_point,
            last_risk + 0.01,
            f"{last_risk:.2f}",
            color=color,
            fontsize=8,
        )
        texts.append(text)

    # Set plot limits, labels, and ticks
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])
    ax.set_xlabel("Follow-up time (years)")
    ax.set_ylabel("Cumulative risk of event")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_width))

    # Ensure the legend respects the order of labels
    handles, legend_labels = ax.get_legend_handles_labels()
    ordered_handles = [
        handles[legend_labels.index(label)]
        for label in labels
        if label in legend_labels
    ]
    ax.legend(ordered_handles, labels, title="Percentile rankings", loc="best")

    # Remove unnecessary spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Adjust text to avoid overlapping
    adjust_text(texts, ax=ax, expand_points=(1.2, 1.6), expand_text=(1.2, 1.6))


def plot_rel_risk(
    y_test,
    y_pred,
    x_label="Predicted Risk Decile",
    y_label="Incidence of the Outcome (%)",
    ax=None,
    s=15,
    bins=10,
):
    """
    Generate a relative risk plot for any binary prediction problem with percentiles on the x-axis.

    Args:
        y_test (array-like): True binary outcomes (0: no event, 1: event).
        y_pred (array-like): Predicted probabilities for risk.
        x_label (str, optional): Label for the x-axis. Defaults to 'Predicted Risk Decile'.
        y_label (str, optional): Label for the y-axis. Defaults to 'Incidence of the Outcome (%)'.
        ax (matplotlib.axes.Axes, optional): Axis on which to plot. If None, a new figure is created.
        s (int, optional): Size of the scatter points. Defaults to 15.
        bins (int, optional): Number of bins to split the predicted probabilities into. Defaults to 10.

    Returns:
        None
    """

    # Create a DataFrame from y_test and y_pred arrays
    pred_df = pd.DataFrame({"y": y_test, "y_pred": y_pred})

    # Group data into deciles based on predicted probabilities
    pred_df["y_pred_bin"] = pd.qcut(
        pred_df["y_pred"].rank(method="first"), bins
    )

    # Aggregate data by decile (calculate the mean of y for each decile)
    grp_df = pred_df.groupby("y_pred_bin", observed=False).agg(
        y_mean=("y", "mean")
    )

    # Generate the percentiles for x-axis
    percentiles = np.linspace(0, bins, num=bins)

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(percentiles, grp_df["y_mean"], s=s)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim([-0.05, 1])
    ax.grid(False)

    plt.tight_layout()


def add_stat_annotation(ax, p_value, x1, x2, y, text_offset=0.05):
    """
    Add statistical annotation with p-value and significance stars on the boxplot.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axis object.
        p_value (float): The p-value to be displayed.
        x1 (float): The x-axis coordinate for the first group.
        x2 (float): The x-axis coordinate for the second group.
        y (float): The y-axis coordinate where the annotation should be placed.
        text_offset (float, optional): Offset for placing the annotation text slightly above the maximum y-value. Defaults to 0.05.

    Returns:
        None
    """
    # Determine significance level
    if p_value < 0.001:
        annotation = "***"
    elif p_value < 0.01:
        annotation = "**"
    elif p_value < 0.05:
        annotation = "*"
    else:
        annotation = "ns"  # Not significant

    # Add the annotation line and text
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + text_offset, y + text_offset, y],
        lw=1.5,
        c="grey",
        alpha=0.5,
    )
    ax.text(
        (x1 + x2) * 0.5,
        y + text_offset,
        f"{annotation} (p={p_value:.3g})",
        ha="center",
        va="bottom",
        color="grey",
    )


def plot_case_control_boxplot(y_test, y_pred, ax=None):
    """
    Create a boxplot showing the probability distribution of true cases and true controls,
    and annotate it with p-value and significance.

    Args:
        y_test (array-like): True binary outcomes (0: no event, 1: event).
        y_pred (array-like): Predicted probabilities for risk.
        ax (matplotlib.axes.Axes, optional): Axis on which to plot. If None, a new figure is created.

    Returns:
        None
    """
    # Convert arrays to DataFrame for easier plotting
    df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

    # Create a 'Group' column to differentiate between cases and controls
    df["Group"] = df["y_test"].map({0: "Control", 1: "Case"})

    # Create the boxplot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Use 'hue' for color assignment and disable the legend
    sns.boxplot(
        x="Group",
        y="y_pred",
        hue="Group",
        data=df,
        ax=ax,
        palette=["#AFCBE3", "#4F6A8A"],
    )
    ax.set_title("Probability Distribution of True Cases and Controls")
    ax.set_ylabel("Predicted Probability")
    ax.grid(False)

    # # Remove the legend since hue duplicates x
    # ax.legend_.remove()

    # Calculate p-value using the Mann-Whitney U test (non-parametric)
    control = df[df["Group"] == "Control"]["y_pred"]
    case = df[df["Group"] == "Case"]["y_pred"]
    stat, p_value = mannwhitneyu(control, case, alternative="two-sided")

    # Add p-value annotation (with stars for significance)
    y_max = max(
        df["y_pred"]
    )  # Maximum y value to place the annotation above the boxplots
    add_stat_annotation(ax, p_value, x1=0, x2=1, y=y_max * 1.05)


def plot_combined(
    y_pred_prob,
    y_test,
    follow_up_time,
    group_labels,
    labels,
    rel_risk_bins=10,
    ylim_cumulative_curve=1,
    output_path=None,
):
    """
    Create a figure with three panels: cumulative risk curve, relative risk, and boxplot.

    Args:
        y_pred_prob (array-like): Predicted probabilities for survival or risk.
        y_test (array-like): True binary outcomes (0: no event, 1: event).
        follow_up_time (array-like): Follow-up time for each subject until event or censoring.
        group_labels (array-like): Group labels for categorizing y_pred_prob into percentiles or other groups.
        labels (list): Labels for the custom percentiles.
        rel_risk_bins (int, optional): Number of bins to split the predicted probabilities into for the relative risk plot. Defaults to 10.
        ylim_cumulative_curve (float, optional): Limit for y-axis in the cumulative risk plot. Defaults to 1.
        output_path (str, optional): Path to save the figure as an SVG file.

    Returns:
        None
    """

    # Create a figure with 3 panels in one row
    fig, axes = plt.subplots(
        1, 3, figsize=(21, 6)
    )  # 21 inches wide for 3 plots

    # First panel: Boxplot
    plot_case_control_boxplot(y_test=y_test, y_pred=y_pred_prob, ax=axes[0])
    axes[0].set_title("Probability Distribution of Cases and Controls")

    # Second panel: Relative risk plot
    plot_rel_risk(
        y_test=y_test,
        y_pred=y_pred_prob,
        x_label="Predicted Risk Decile",
        y_label="Incidence of the Outcome (%)",
        ax=axes[1],
        s=15,
        bins=rel_risk_bins,
    )
    axes[1].set_title("Relative Risk per Decile")

    # Third panel: Cumulative risk curve
    plot_cumulative_risk_curve(
        y_pred_prob=y_pred_prob,
        y_test=y_test,
        follow_up_time=follow_up_time,
        group_labels=group_labels,
        labels=labels,
        xlim=15,
        target_cumulative=0.5,
        tick_width=1,
        ylim=ylim_cumulative_curve,
        ax=axes[2],
    )
    axes[2].set_title("Cumulative Risk Curve")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save plot as an SVG file with 300 DPI if output_path is provided
    save_svg(output_path)

    plt.show()
