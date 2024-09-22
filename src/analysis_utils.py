import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
from adjustText import adjust_text


# Data Preprocessing Functions
def clean_data(df, outcome_col):
    """
    Clean data by removing rows with missing values for the outcome column.
    """
    return df.dropna(subset=[outcome_col]).copy()

def calculate_propensity_scores(df, covariates, outcome_col):
    """
    Calculate propensity scores using logistic regression on covariates.
    """
    X_covariates = df[covariates]
    y = df[outcome_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_covariates)
    propensity_model = LogisticRegression(solver='liblinear')
    propensity_model.fit(X_scaled, y)
    df['propensity_score'] = propensity_model.predict_proba(X_scaled)[:, 1]
    return df


# Feature Engineering (Propensity Score Matching)
def match_propensity_scores(df, outcome_col):
    """
    Perform nearest neighbor matching based on propensity scores.
    """
    cases = df[df[outcome_col] == 1].copy()
    controls = df[df[outcome_col] == 0].copy()
    nbrs = NearestNeighbors(n_neighbors=1)
    nbrs.fit(controls[['propensity_score']])
    distances, indices = nbrs.kneighbors(cases[['propensity_score']])
    matched_controls = controls.iloc[indices.flatten()]
    matched_data = pd.concat([cases, matched_controls], ignore_index=True)
    return matched_data


# Logistic Regression for Biomarkers
def run_logistic_regression_on_biomarkers(df, covariates, outcome_col, biomarker_cols):
    """
    Run logistic regression on biomarkers and return log2 fold changes and p-values.
    """
    log2_fold_changes = []
    p_values = []
    for biomarker in biomarker_cols:
        X = df[[biomarker] + covariates]
        X = sm.add_constant(X)
        logit_model = sm.Logit(df[outcome_col], X)
        result = logit_model.fit(disp=0)
        odds_ratio = np.exp(result.params.iloc[1])
        log2_fold_changes.append(np.log2(odds_ratio))
        p_values.append(result.pvalues.iloc[1])
    return log2_fold_changes, p_values

def adjust_p_values(p_values):
    """
    Adjust p-values using Benjamini-Hochberg procedure.
    """
    _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')
    return corrected_p_values


# Visualization (Volcano Plot)
def plot_volcano(biomarker_cols, log2_fold_changes, corrected_p_values, fc_threshold=1, pval_threshold=0.05):
    """
    Create a volcano plot to visualize biomarker associations.
    """
    neg_log_p_values = -np.log10(corrected_p_values)
    volcano_data = pd.DataFrame({
        'biomarker': biomarker_cols,
        'log2FC': log2_fold_changes,
        '-log10(p-value)': neg_log_p_values
    })
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=volcano_data, x='log2FC', y='-log10(p-value)', color='grey', edgecolor=None, s=8, alpha=0.2)
    plt.grid(False)
    pval_threshold = -np.log10(pval_threshold)
    sig = volcano_data[(volcano_data['-log10(p-value)'] > pval_threshold) & (abs(volcano_data['log2FC']) > fc_threshold)]
    sns.scatterplot(data=sig, x='log2FC', y='-log10(p-value)', color='red', edgecolor=None, s=8)
    plt.axhline(y=pval_threshold, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(x=fc_threshold, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(x=-fc_threshold, color='grey', linestyle='--', alpha=0.5)
    texts = []
    for i in range(sig.shape[0]):
        texts.append(plt.text(sig['log2FC'].iloc[i], sig['-log10(p-value)'].iloc[i], sig['biomarker'].iloc[i], 
                              fontsize=8, ha='right', color='black'))
    adjust_text(texts)
    plt.title('Volcano Plot: Biomarker Association with Incident Diabetes')
    plt.xlabel('Log2 Fold Change (Odds Ratio)')
    plt.ylabel('-Log10(p-value)')
    plt.show()
