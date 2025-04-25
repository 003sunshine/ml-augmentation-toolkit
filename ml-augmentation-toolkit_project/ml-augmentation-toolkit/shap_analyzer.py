import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score


class SHAPAnalyzer:
    """
    使用XGBoost + SHAP进行特征重要性分析和交互作用分析。
    """

    def __init__(self, target_col, feature_name_mapping=None, random_state=42):
        self.target_col = target_col
        self.feature_name_mapping = feature_name_mapping or {}
        self.random_state = random_state

    def fit(self, train_data, test_data, model_params=None):
        self.train_data = train_data
        self.test_data = test_data

        self.X_train = self.train_data.drop(columns=[self.target_col])
        self.y_train = self.train_data[self.target_col]
        self.X_test = self.test_data.drop(columns=[self.target_col], errors='ignore')

        self.features = self.X_train.columns.tolist()
        self.feature_display_names = [self.feature_name_mapping.get(col, col) for col in self.features]

        self.model_params = model_params or {
            'colsample_bytree': 1.0,
            'gamma': 2.0,
            'learning_rate': 0.1,
            'max_depth': 10,
            'n_estimators': 50,
            'subsample': 0.7,
            'eval_metric': 'rmse',
            'n_jobs': -1,
            'random_state': self.random_state
        }

        xgb_model = XGBRegressor(**self.model_params)
        kf = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        y_pred = cross_val_predict(xgb_model, self.X_train, self.y_train, cv=kf)

        self.r2_score_cv = r2_score(self.y_train, y_pred)
        self.y_cv_pred = y_pred  # 保存交叉验证预测
        print(f"Cross-validated R²: {self.r2_score_cv:.4f}")

        self.final_model = xgb_model.fit(self.X_train, self.y_train)

        self.explainer = shap.TreeExplainer(self.final_model, feature_perturbation='tree_path_dependent')
        self.shap_values = self.explainer(self.X_test).values
        self.shap_interaction_values = self.explainer.shap_interaction_values(self.X_test)

    def save_feature_importance(self, path):
        xgb_importance = self.final_model.feature_importances_
        shap_importance = np.abs(self.shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'Feature': self.features,
            'DisplayName': self.feature_display_names,
            'XGBoost_Importance': xgb_importance,
            'SHAP_Importance': shap_importance
        }).sort_values('SHAP_Importance', ascending=False)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        importance_df.to_csv(path, index=False, float_format="%.6f")
        print(f"✅ 特征重要性保存到: {path}")

    def save_shap_values(self, path):
        shap_df = pd.DataFrame(self.shap_values, columns=self.features)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shap_df.to_csv(path, index=False, float_format="%.6f")
        print(f"✅ SHAP值保存到: {path}")

    def save_shap_summary_plot(self, path):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_test, feature_names=self.feature_display_names, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP蜂群图保存到: {path}")

    def save_interaction_heatmap(self, path):
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_interaction_values, self.X_test, plot_type="compact_dot", show=False)
        plt.title("SHAP Interaction Heatmap")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 交互热力图保存到: {path}")

    def save_interaction_strengths(self, path):
        strength = np.mean(np.abs(self.shap_interaction_values), axis=0)

        interaction_records = []
        for i in range(len(self.features)):
            for j in range(i+1, len(self.features)):
                interaction_records.append({
                    'Feature_A': self.features[i],
                    'Feature_B': self.features[j],
                    'Interaction_Strength': strength[i, j]
                })

        interaction_df = pd.DataFrame(interaction_records).sort_values('Interaction_Strength', ascending=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        interaction_df.to_csv(path, index=False, float_format="%.6f")
        print(f"✅ 全局交互强度保存到: {path}")

    def plot_dependence(self, feature, interaction_feature=None, path=None):
        shap.dependence_plot(
            feature,
            self.shap_values,
            self.X_test,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f"{feature} Interaction with {interaction_feature}")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 依赖图保存到: {path}")
        else:
            plt.show()
