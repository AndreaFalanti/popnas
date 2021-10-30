import os

import pandas as pd
import shap
from matplotlib import pyplot as plt


def save_feature_analysis_plots(model, features_df: pd.DataFrame, log_folder: str, save_pred_every: int = None, model_type: str = 'auto'):
    # TODO: shap automatic explainer selection doesn't work well, better use custom class directly if the model is not recognized
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

        for i in range((len(shap_values) // save_pred_every) + 1):
            shap.plots.waterfall(shap_values[i * save_pred_every], max_display=features_num, show=False)
            plt.savefig(os.path.join(pred_path, f'shap_pred{i * save_pred_every}.png'), bbox_inches='tight')
            plt.close()

    print('Shap plots written successfully')
