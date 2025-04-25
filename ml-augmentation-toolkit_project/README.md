# alloyxai

> **A modular machine learning pipeline for data augmentation and explainable modeling in superalloy design**  
> 面向高温合金设计的数据增强与可解释性建模一体化机器学习框架

---

## 🔬 Project Overview | 项目概述

**`alloyxai`** is a research-oriented Python toolkit that integrates *data generation*, *imbalance handling*, and *model interpretability* into a unified machine learning pipeline, specifically designed for **superalloy composition optimization and microstructure-performance prediction**.

该项目融合了多种数据增强手段（MCMC、WGAN-GP、SMOGN）与可解释性分析（SHAP），适用于**高温合金成分设计、相粗化行为建模及高温性能预测等典型材料科学问题**。

---

## 🧩 Core Modules | 核心模块

| 模块名            | 描述 |
|-------------------|------|
| `MCMCSampler`     | 基于贝叶斯推断的元素比例生成器（Dirichlet + TruncatedNormal） |
| `WGANGPRegressor` | 面向回归问题的小样本数据生成器，集成条件判别与梯度惩罚机制 |
| `SMOGNAugmentor`  | 用于不平衡目标分布的回归型过采样（适合长尾、高偏态分布） |
| `SHAPAnalyzer`    | 提供主效应、交互项、蜂群图与依赖图等多层次模型解释能力 |

---

## 🚀 Example Workflow | 示例工作流

```bash
# 安装依赖
pip install -r requirements.txt

# 运行主流程（默认启用 MCMC + WGAN + SHAP）
python pipeline.py
