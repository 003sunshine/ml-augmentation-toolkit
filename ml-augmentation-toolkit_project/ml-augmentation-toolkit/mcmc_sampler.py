import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns


class MCMCSampler:
    """
    使用 PyMC 对高温合金元素组成与温度进行 MCMC 采样。

    - 元素组成建模为 Dirichlet 分布（强约束：总和为100%）
    - 温度建模为 Truncated Normal 分布
    """

    def __init__(self,
                 data_path,
                 trace_save_path,
                 sample_save_path,
                 elements_cols=None,
                 t_col='T',
                 draws=4000,
                 tune=1000,
                 chains=4,
                 cores=4,
                 seed=42,
                 concentration=100):
        """
        初始化采样器

        Parameters:
            data_path (str): 原始CSV数据路径
            trace_save_path (str): 轨迹保存路径
            sample_save_path (str): 生成样本保存路径
            elements_cols (list): 元素列名（默认10种常见元素）
            t_col (str): 温度列名
            draws (int): 每条链的采样步数
            tune (int): 调优步数
            chains (int): 链数
            cores (int): 并行核数
            seed (int): 随机种子
            concentration (float): Dirichlet浓度参数
        """
        self.data_path = data_path
        self.trace_save_path = trace_save_path
        self.sample_save_path = sample_save_path
        self.elements_cols = elements_cols or ['Co', 'Al', 'W', 'Ta', 'Ti', 'Nb', 'Ni', 'Cr', 'V', 'Mo']
        self.t_col = t_col
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.seed = seed
        self.concentration = concentration
        self.EPSILON = 1e-6

    def load_data(self):
        """读取数据并检查列合法性"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到数据文件: {self.data_path}")
        self.data = pd.read_csv(self.data_path)

        for col in self.elements_cols + [self.t_col]:
            if col not in self.data.columns:
                raise ValueError(f"缺失列: {col}，请检查数据文件格式。")

        self.elements_data = self.data[self.elements_cols].replace(0, 1e-5)
        self.t_data = self.data[self.t_col]

    def _compute_dirichlet_alpha(self):
        """根据元素均值计算 Dirichlet 参数 α"""
        mean_props = self.elements_data.mean(axis=0) / 100.0
        alpha = np.maximum(mean_props * self.concentration, self.EPSILON)
        return alpha

    def build_model(self):
        """构建 PyMC 模型并进行采样"""
        alpha = self._compute_dirichlet_alpha()
        t_mu, t_sigma = self.t_data.mean(), self.t_data.std()
        t_min, t_max = self.t_data.min(), self.t_data.max()

        with pm.Model() as self.model:
            proportions = pm.Dirichlet("proportions", a=alpha, shape=(len(self.elements_cols),))
            elements_generated = pm.Deterministic("elements_generated", proportions * 100)
            t_prior = pm.TruncatedNormal("T_prior", mu=t_mu, sigma=t_sigma,
                                         lower=t_min - 10, upper=t_max + 10)

            self.trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=0.95,
                random_seed=self.seed,
                return_inferencedata=True
            )

    def check_convergence(self):
        """使用ArviZ进行收敛性诊断"""
        summary = az.summary(self.trace, var_names=["proportions", "T_prior"])
        if summary["r_hat"].max() > 1.05:
            print("⚠️ 警告：存在未收敛参数，建议增加采样步数或调整模型！")
        return summary

    def save_trace(self):
        """保存 MCMC 轨迹数据为 CSV"""
        proportions_trace = self.trace.posterior['proportions'].stack(sample=("chain", "draw")).values.transpose(1, 0)
        t_trace = self.trace.posterior['T_prior'].stack(sample=("chain", "draw")).values.flatten()
        trace_df = pd.DataFrame(proportions_trace, columns=[f"proportions_{el}" for el in self.elements_cols])
        trace_df["T_prior"] = t_trace

        os.makedirs(os.path.dirname(self.trace_save_path), exist_ok=True)
        trace_df.to_csv(self.trace_save_path, index=False)

    def extract_samples(self):
        """提取生成的后验样本"""
        posterior = self.trace.posterior
        self.samples_df = pd.DataFrame({
            col: posterior['elements_generated'][..., i].values.flatten()
            for i, col in enumerate(self.elements_cols)
        })
        self.samples_df['T'] = posterior['T_prior'].values.flatten()

    def save_samples(self):
        """保存后验样本"""
        os.makedirs(os.path.dirname(self.sample_save_path), exist_ok=True)
        self.samples_df.to_csv(self.sample_save_path, index=False)

    def plot_distributions(self, save_dir=None):
        """原始与生成数据分布对比图（可选保存）"""
        for col in self.elements_cols + ['T']:
            plt.figure(figsize=(8, 4))
            sns.kdeplot(self.data[col], label="原始数据", fill=True)
            sns.kdeplot(self.samples_df[col], label="生成数据", fill=True)
            plt.title(f"{col} 分布对比")
            plt.xlabel("值")
            plt.ylabel("密度")
            plt.legend()
            plt.tight_layout()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{col}_kde.png"))
            plt.show()

    def run(self, plot=True, save_plot_dir=None):
        """执行完整 MCMC 流程"""
        print("🔄 开始 MCMC 流程...")
        self.load_data()
        self.build_model()
        self.check_convergence()
        self.save_trace()
        self.extract_samples()
        self.save_samples()
        if plot:
            self.plot_distributions(save_dir=save_plot_dir)
        print("✅ MCMC流程完成！")
        return self.samples_df, self.trace

