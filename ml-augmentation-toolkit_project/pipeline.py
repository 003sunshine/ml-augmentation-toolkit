import os
import pandas as pd
from ml-augmentation-toolkit.mcmc_sampler import MCMCSampler
from ml-augmentation-toolkit.wgan_gp_generator import WGANGPRegressor
from ml-augmentation-toolkit.smogn_augmentor import SMOGNAugmentor
from ml-augmentation-toolkit.shap_analyzer import SHAPAnalyzer
from sklearn.preprocessing import StandardScaler


def main(config):
    # Step 0: 加载原始数据
    print("\n🔵 加载原始数据...")
    original_df = pd.read_csv(config["original_data_path"])

    # 初始化增强数据列表
    enhanced_datasets = []

    # Step 1: MCMC采样（可选）
    if config["use_mcmc"]:
        print("\n🚀 Step 1: MCMC Sampling...")
        mcmc_sampler = MCMCSampler(
            data_path=config["original_data_path"],
            trace_save_path=config["mcmc"]["trace_save_path"],
            sample_save_path=config["mcmc"]["sample_save_path"],
            draws=config["mcmc"]["draws"],
            chains=config["mcmc"]["chains"],
            cores=config["mcmc"]["cores"]
        )
        mcmc_samples, _ = mcmc_sampler.run(plot=False)
        enhanced_datasets.append(mcmc_samples)
        print("✅ MCMC采样完成。")

    # Step 2: WGAN-GP生成（可选）
    if config["use_wgan"]:
        print("\n🚀 Step 2: WGAN-GP Generation...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X = original_df.drop(columns=[config["target_col"]]).values
        y = original_df[config["target_col"]].values.reshape(-1, 1)

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        wgan_gp = WGANGPRegressor(latent_dim=config["wgan"]["latent_dim"])
        wgan_gp.fit(X_scaled, y_scaled, n_epochs=config["wgan"]["n_epochs"], batch_size=config["wgan"]["batch_size"])

        generated_scaled = wgan_gp.generate(y_scaled, n_samples=config["wgan"]["n_generated_samples"])
        generated_X = scaler_X.inverse_transform(generated_scaled)

        generated_df = pd.DataFrame(generated_X, columns=original_df.columns.drop(config["target_col"]))
        generated_df[config["target_col"]] = scaler_y.inverse_transform(y_scaled[:generated_df.shape[0]]).flatten()

        os.makedirs(os.path.dirname(config["wgan"]["save_path"]), exist_ok=True)
        generated_df.to_csv(config["wgan"]["save_path"], index=False)
        enhanced_datasets.append(generated_df)
        print("✅ WGAN-GP生成完成。")

    # Step 3: SMOGN增强（可选）
    if config["use_smogn"]:
        print("\n🚀 Step 3: SMOGN Data Augmentation...")
        smogn_augmentor = SMOGNAugmentor(
            target_col=config["target_col"],
            samp_method="balance",
            save_path=config["smogn"]["save_path"]
        )
        smogn_df = smogn_augmentor.fit_transform(original_df)
        smogn_augmentor.save()
        enhanced_datasets.append(smogn_df)
        print("✅ SMOGN增强完成。")

    # Step 4: 整合所有增强数据
    print("\n🔵 整合增强数据...")
    if enhanced_datasets:
        all_data = pd.concat(enhanced_datasets, axis=0).drop_duplicates().reset_index(drop=True)
    else:
        print("⚠️ 未选择任何数据增强方法，仅使用原始数据。")
        all_data = original_df.copy()

    print(f"总数据量: {all_data.shape[0]} 条")

    # Step 5: SHAP 可解释性分析
    print("\n🚀 Step 4: SHAP Analysis...")
    shap_analyzer = SHAPAnalyzer(
        target_col=config["target_col"],
        feature_name_mapping=config.get("feature_name_mapping", {}),
        random_state=42
    )
    test_data = pd.read_csv(config["shap"]["test_data_path"])

    shap_analyzer.fit(train_data=all_data, test_data=test_data)

    shap_analyzer.save_feature_importance(config["shap"]["feature_importance_path"])
    shap_analyzer.save_shap_values(config["shap"]["shap_values_path"])
    shap_analyzer.save_shap_summary_plot(config["shap"]["shap_summary_plot_path"])
    shap_analyzer.save_interaction_heatmap(config["shap"]["interaction_heatmap_path"])
    shap_analyzer.save_interaction_strengths(config["shap"]["interaction_strength_path"])
    shap_analyzer.plot_dependence(
        feature=config["shap"]["dependence_plot_feature"],
        interaction_feature=config["shap"]["dependence_plot_interaction"],
        path=config["shap"]["dependence_plot_path"]
    )

    print("\n🎯 Pipeline 完成!")


if __name__ == "__main__":
    config = {
        "original_data_path": "data/原始实验数据.csv",  # 原始实验数据路径

        "use_mcmc": True,   # 是否启用 MCMC
        "use_wgan": True,   # 是否启用 WGAN
        "use_smogn": False, # 是否启用 SMOGN

        "target_col": "Vol",

        "mcmc": {
            "trace_save_path": "outputs/mcmc_trace.csv",
            "sample_save_path": "outputs/mcmc_samples.csv",
            "draws": 4000,
            "chains": 4,
            "cores": 8
        },

        "wgan": {
            "latent_dim": 11,
            "n_epochs": 3000,
            "batch_size": 64,
            "n_generated_samples": 1000,
            "save_path": "outputs/wgan_generated.csv"
        },

        "smogn": {
            "save_path": "outputs/smogn_augmented.csv"
        },

        "shap": {
            "test_data_path": "data/原始实验数据.csv",
            "feature_importance_path": "outputs/shap_feature_importance.csv",
            "shap_values_path": "outputs/shap_values.csv",
            "shap_summary_plot_path": "outputs/shap_summary_plot.png",
            "interaction_heatmap_path": "outputs/interaction_heatmap.png",
            "interaction_strength_path": "outputs/global_interaction_strength.csv",
            "dependence_plot_feature": "Ti",
            "dependence_plot_interaction": "Ta",
            "dependence_plot_path": "outputs/Ti_Ta_dependence_plot.png"
        },

        "feature_name_mapping": {
            "Co": "Co", "Al": "Al", "W": "W", "Ta": "Ta", "Ti": "Ti", "Nb": "Nb", "Ni": "Ni", "Cr": "Cr", "V": "V", "Mo": "Mo",
            "Tage": r"$T_{\mathrm{age}}$", "tage": r"$t_{\mathrm{age}}$"
        }
    }

    main(config)
