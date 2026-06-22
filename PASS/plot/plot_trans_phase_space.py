import numpy as np
import matplotlib.pyplot as plt


def plot_twiss_ellipse(alpha, beta, emittance=1.0, xlim=None, ylim=None):
    """
    根据给定的 Twiss 参数绘制相椭圆，并可自定义坐标轴范围
    """
    gamma = (1 + alpha**2) / beta

    # 定义角度 theta 从 0 到 2*pi
    theta = np.linspace(0, 2 * np.pi, 500)

    # 椭圆参数化坐标计算
    x = np.sqrt(emittance * beta) * np.cos(theta)
    xp = -np.sqrt(emittance / beta) * (alpha * np.cos(theta) + np.sin(theta))

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(x, xp, label=rf'$\alpha={alpha}, \beta={beta}$')

    # 绘制坐标轴
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xlabel(r'$x$ (Position)')
    plt.ylabel(r"$x'$ (Divergence)")
    plt.title('Twiss Phase Ellipse')
    plt.legend()

    # 保持坐标轴等比例（重要：防止椭圆形状因坐标轴拉伸而变形）
    plt.axis()

    # --- 新增：设置坐标轴范围 ---
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.show()


# --- 使用示例 ---
# 传入 xlim 参数，例如将 X 轴限制在 [-3, 3] 范围内
plot_twiss_ellipse(alpha=-2.6143, beta=0.5, emittance=200e-6*9, xlim=(-0.04, 0.04), ylim=(-0.2, 0.2))
