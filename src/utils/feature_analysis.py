import os

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt


def save_feature_analysis_plots(model, features_df: pd.DataFrame, log_folder: str, save_pred_every: int = None, model_type: str = 'auto'):
    if model_type == 'auto':
        explainer = shap.Explainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, features_df)
    else:
        print('[Warning] Invalid model_type, shap will not produce any feature plot')
        return

    shap_values = explainer(features_df)

    features_num = len(features_df.columns.to_list())
    shap.plots.beeswarm(shap_values, max_display=features_num, show=False)
    plt.savefig(os.path.join(log_folder, 'shap_impact_on_output.png'), bbox_inches='tight')
    plt.close()

    shap.plots.bar(shap_values, max_display=features_num, show=False)
    plt.savefig(os.path.join(log_folder, 'shap_importance.png'), bbox_inches='tight')
    plt.close()

    if save_pred_every is not None:
        assert save_pred_every > 0, 'save_pred_every must be a positive value'
        pred_path = os.path.join(log_folder, 'feature_pred_contributions')
        os.makedirs(pred_path, exist_ok=True)

        for i in range(len(shap_values) // save_pred_every):
            shap.plots.waterfall(shap_values[i * save_pred_every], max_display=features_num, show=False)
            plt.savefig(os.path.join(pred_path, f'shap_pred{i * save_pred_every}.png'), bbox_inches='tight')
            plt.close()

    print('Shap plots written successfully')


# inspired from: https://towardsdatascience.com/the-art-of-finding-the-best-features-for-machine-learning-a9074e2ca60d
def generate_dataset_correlation_heatmap(csv_path: str, save_folder: str, save_name: str = 'dataset_corr_heatmap.png'):
    dataset_df = pd.read_csv(csv_path).drop(columns=['data_augmented', 'exploration'], errors='ignore')

    # use the pands .corr() function to compute pairwise correlations for the dataframe
    corr = dataset_df.corr()

    # visualise the data with seaborn
    # upper triangular mask
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    sns.set_style(style='white')
    f, ax = plt.subplots(figsize=(12, 12))
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(corr, vmin=-1, vmax=1, mask=mask, cmap=cmap, square=True, linewidths=1, xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": .5}, ax=ax)  # type: plt.Axes
    f.savefig(os.path.join(save_folder, save_name), bbox_inches='tight')
    plt.close(f)
