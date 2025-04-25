import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def seed_everything(seed=42):
    """确保结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),  # 输入：噪声 + 条件变量
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim)  # 输出：回归特征
        )

    def forward(self, z, conditions):
        input_combined = torch.cat((z, conditions), dim=1)
        return self.model(input_combined)


class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # 输出：真实性分数
        )

    def forward(self, x, conditions):
        input_combined = torch.cat((x, conditions), dim=1)
        return self.model(input_combined)


class WGANGPRegressor:
    def __init__(self, latent_dim=11, lambda_gp=10, device=None):
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None
        self.discriminator = None

    def fit(self, X, conditions, n_epochs=3000, batch_size=32, n_critic=5, lr=1e-5, save_log_path=None):
        """
        训练 WGAN-GP 模型

        参数:
            X: np.ndarray, shape=(n_samples, n_features)，回归特征
            conditions: np.ndarray, shape=(n_samples, n_condition_features)，条件变量
        """
        seed_everything()

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        conditions = torch.tensor(conditions, dtype=torch.float32).to(self.device)

        self.output_dim = X.shape[1]
        self.condition_dim = conditions.shape[1]

        self.generator = Generator(self.latent_dim, self.condition_dim, self.output_dim).to(self.device)
        self.discriminator = Discriminator(self.output_dim, self.condition_dim).to(self.device)

        optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

        self.g_losses = []
        self.d_losses = []

        for epoch in range(n_epochs):
            for _ in range(n_critic):
                optimizer_D.zero_grad()
                idx = torch.randint(0, X.shape[0], (batch_size,))
                real_x = X[idx]
                real_c = conditions[idx]

                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_x = self.generator(z, real_c).detach()

                d_real = self.discriminator(real_x, real_c)
                d_fake = self.discriminator(fake_x, real_c)
                gp = self._gradient_penalty(real_x, fake_x, real_c)

                d_loss = -torch.mean(d_real) + torch.mean(d_fake) + gp
                d_loss.backward()
                optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_x = self.generator(z, real_c)
            d_fake = self.discriminator(fake_x, real_c)
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            optimizer_G.step()

            self.d_losses.append(d_loss.item())
            self.g_losses.append(g_loss.item())

            if epoch % 100 == 0:
                print(f"[{epoch}/{n_epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        # 日志保存
        if save_log_path:
            df_log = pd.DataFrame({'D_loss': self.d_losses, 'G_loss': self.g_losses})
            os.makedirs(os.path.dirname(save_log_path), exist_ok=True)
            df_log.to_csv(save_log_path, index=False)

    def _gradient_penalty(self, real_x, fake_x, condition):
        alpha = torch.rand(real_x.size(0), 1).to(self.device)
        interpolated = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
        d_interpolated = self.discriminator(interpolated, condition)
        gradients = torch.autograd.grad(outputs=d_interpolated,
                                        inputs=interpolated,
                                        grad_outputs=torch.ones_like(d_interpolated),
                                        create_graph=True, retain_graph=True)[0]
        grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        return self.lambda_gp * ((grad_norm - 1) ** 2).mean()

    def generate(self, condition_array, n_samples=None, z=None):
        """
        生成模拟数据

        参数:
            condition_array: np.ndarray, 条件变量数组
            n_samples: int, 要生成的样本数量（若 z 提供则可省略）
            z: torch.Tensor, 自定义潜变量张量

        返回:
            np.ndarray: 生成数据
        """
        self.generator.eval()
        condition_array = np.array(condition_array)

        if z is None:
            if n_samples is None:
                n_samples = condition_array.shape[0]
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
        else:
            z = z.to(self.device)
            n_samples = z.size(0)

        conditions = torch.tensor(condition_array, dtype=torch.float32).to(self.device)
        if conditions.shape[0] != n_samples:
            raise ValueError("生成样本数与条件变量数量不一致。")

        with torch.no_grad():
            fake_data = self.generator(z, conditions).cpu().numpy()
        return fake_data

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label="Discriminator Loss")
        plt.plot(self.g_losses, label="Generator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("WGAN-GP Training Loss")
        plt.tight_layout()
        plt.show()

    def save_model(self, path_prefix):
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        torch.save(self.generator.state_dict(), path_prefix + "_G.pth")
        torch.save(self.discriminator.state_dict(), path_prefix + "_D.pth")

    def load_model(self, path_prefix):
        self.generator.load_state_dict(torch.load(path_prefix + "_G.pth", map_location=self.device))
        self.discriminator.load_state_dict(torch.load(path_prefix + "_D.pth", map_location=self.device))
