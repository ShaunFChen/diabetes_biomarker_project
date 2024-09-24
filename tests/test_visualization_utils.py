import pytest

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

import matplotlib

matplotlib.use("Agg")
plt.set_loglevel("WARNING")

from src.visualization_utils import *


def test_save_svg(tmp_path):
    # Create a simple plot
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])

    # Define output path
    output_file = tmp_path / "test_plot.svg"

    # Call save_svg with the output path
    save_svg(output_file)

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close()


def test_save_svg_no_output():
    # Create a simple plot
    plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])

    # Call save_svg with None
    save_svg(None)

    # No exception should occur, and no file should be saved
    # Since we didn't provide a path, we cannot check for a file

    # Clean up
    plt.close()


def test_plot_distribution_numeric(tmp_path):
    data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    output_file = tmp_path / "histogram.svg"

    plot_distribution(
        data,
        column="value",
        xlabel="Value",
        plot_type="auto",
        title="Numeric Distribution",
        output_path=output_file,
    )

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")


def test_plot_distribution_categorical(tmp_path):
    data = pd.DataFrame({"category": ["A", "B", "A", "C", "B"]})
    output_file = tmp_path / "countplot.svg"

    plot_distribution(
        data,
        column="category",
        xlabel="Category",
        plot_type="auto",
        title="Categorical Distribution",
        output_path=output_file,
    )

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")


def test_plot_distribution_custom_parameters():
    data = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

    plot_distribution(
        data,
        column="value",
        xlabel="Value",
        plot_type="histplot",
        bins=5,
        axvline=3,
        axvline_text="Median",
        title="Custom Histogram",
    )

    # No exception should occur
    # Clean up
    plt.close("all")


def test_plot_volcano(tmp_path):
    biomarker_cols = ["Biomarker1", "Biomarker2", "Biomarker3"]
    log2_fold_changes = np.array([1.5, -2.0, 0.5])
    corrected_p_values = np.array([0.01, 0.05, 0.2])
    output_file = tmp_path / "volcano_plot.svg"

    plot_volcano(
        biomarker_cols,
        log2_fold_changes,
        corrected_p_values,
        fc_threshold=1,
        pval_threshold=0.05,
        output_path=output_file,
    )

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")


def test_plot_multicollinearity_heatmap(tmp_path):
    # Create a sample DataFrame
    data = {
        "Biomarker1": np.random.rand(10),
        "Biomarker2": np.random.rand(10),
        "Biomarker3": np.random.rand(10),
    }
    df = pd.DataFrame(data)
    output_file = tmp_path / "heatmap.svg"

    plot_multicollinearity_heatmap(df, output_path=output_file)

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")


def test_plot_roc_curves(tmp_path):
    y_test = np.array([0, 1, 0, 1])
    prediction_probs = {
        "Model1": np.array([0.2, 0.8, 0.3, 0.7]),
        "Model2": np.array([0.1, 0.9, 0.4, 0.6]),
    }
    output_file = tmp_path / "roc_curves.svg"

    plot_roc_curves(prediction_probs, y_test, output_path=output_file)

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")


def test_plot_feature_importance(tmp_path):
    selected_features = ["Feature1", "Feature2", "Feature3"]
    feature_importance = [0.8, 0.5, 0.3]
    output_file = tmp_path / "feature_importance.svg"

    plot_feature_importance(
        selected_features,
        feature_importance,
        importance_metric="Coefficient Magnitude",
        title="Feature Importance",
        top_N=3,
        output_path=output_file,
    )

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")


def test_plot_biomarker_skewness(tmp_path):
    df1 = pd.DataFrame({"Biomarker1": np.random.rand(10)})
    df2 = pd.DataFrame({"Biomarker1": np.random.rand(10)})
    dfs_dict = {"Dataset1": df1, "Dataset2": df2}
    output_file = tmp_path / "skewness.svg"

    plot_biomarker_skewness(dfs_dict, output_path=output_file)

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")


def test_plot_cumulative_risk_curve():
    y_pred_prob = np.array([0.2, 0.8, 0.5, 0.7])
    y_test = np.array([0, 1, 0, 1])
    follow_up_time = np.array([5, 10, 7, 8])
    group_labels = np.array(
        ["Low Risk", "High Risk", "Medium Risk", "High Risk"]
    )
    labels = ["Low Risk", "Medium Risk", "High Risk"]

    plot_cumulative_risk_curve(
        y_pred_prob,
        y_test,
        follow_up_time,
        group_labels,
        labels,
        xlim=12,
        ylim=1,
        ax=None,
    )

    # No exception should occur
    # Clean up
    plt.close("all")


def test_plot_rel_risk():
    y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.linspace(0.1, 0.9, 10)

    plot_rel_risk(
        y_test,
        y_pred,
        x_label="Predicted Risk Decile",
        y_label="Incidence of the Outcome (%)",
        ax=None,
        s=15,
        bins=5,
    )

    # No exception should occur
    # Clean up
    plt.close("all")


def test_plot_case_control_boxplot():
    y_test = np.array([0, 1, 0, 1])
    y_pred = np.array([0.2, 0.8, 0.3, 0.7])

    plot_case_control_boxplot(y_test, y_pred, ax=None)

    # No exception should occur
    # Clean up
    plt.close("all")


def test_plot_combined(tmp_path):
    y_pred_prob = np.array([0.2, 0.8, 0.5, 0.7])
    y_test = np.array([0, 1, 0, 1])
    follow_up_time = np.array([5, 10, 7, 8])
    group_labels = np.array(
        ["Low Risk", "High Risk", "Medium Risk", "High Risk"]
    )
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    output_file = tmp_path / "combined_plot.svg"

    plot_combined(
        y_pred_prob,
        y_test,
        follow_up_time,
        group_labels,
        labels,
        rel_risk_bins=3,
        ylim_cumulative_curve=1,
        output_path=output_file,
    )

    # Check that the file exists
    assert output_file.exists()

    # Clean up
    plt.close("all")
