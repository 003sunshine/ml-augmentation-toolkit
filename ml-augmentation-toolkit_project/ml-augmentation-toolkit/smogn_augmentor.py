import os
import smogn
import pandas as pd
import matplotlib.pyplot as plt


class SMOGNAugmentor:
    """
    使用 SMOGN 对回归数据进行增强，解决目标变量分布不平衡问题。
    """

    def __init__(self, target_col, samp_method="balance", save_path=None):
        """
        初始化增强器

        参数:
            target_col (str): 目标变量名（如 'K', 'Vol'）
            samp_method (str): 采样方法，可选 'balance' 或 'extreme'
            save_path (str): 增强后数据的保存路径（可选）
        """
        self.target_col = target_col
        self.samp_method = samp_method
        self.save_path = save_path
        self.original_df = None
        self.enhanced_df = None

    def fit_transform(self, df):
        """
        对输入 DataFrame 执行 SMOGN 增强

        参数:
            df (pd.DataFrame): 原始数据

        返回:
            pd.DataFrame: 增强后的数据
        """
        self.original_df = df.copy()
        self.enhanced_df = smogn.smoter(
            data=df,
            y=self.target_col,
            samp_method=self.samp_method
        )
        return self.enhanced_df

    def plot_distribution(self, bins=30):
        """
        可视化增强前后目标变量的分布对比图

        参数:
            bins (int): 直方图分箱数
        """
        if self.original_df is None or self.enhanced_df is None:
            raise ValueError("请先运行 fit_transform()")

        plt.figure(figsize=(8, 5))
        plt.hist(self.original_df[self.target_col], bins=bins, alpha=0.5, label="原始数据", edgecolor="black")
        plt.hist(self.enhanced_df[self.target_col], bins=bins, alpha=0.5, label="SMOGN 增强数据", edgecolor="black")
        plt.xlabel(self.target_col)
        plt.ylabel("频数")
        plt.title(f"SMOGN 增强前后 {self.target_col} 的分布对比")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save(self, path=None):
        """
        保存增强后的数据为 CSV

        参数:
            path (str): 指定保存路径；如为空则使用初始化时的 save_path
        """
        path = path or self.save_path
        if path is None:
            raise ValueError("未指定保存路径，请传入 path 或设置 save_path")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.enhanced_df.to_csv(path, index=False)
        print(f"✅ 增强后的数据已保存至：{path}")
