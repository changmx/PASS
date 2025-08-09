import numpy as np
import matplotlib.pyplot as plt

e = 1.602176634e-19
epsilon0 = 8.8541878128e-12
pi = 3.141592653589793


def cal_Er_KV(x, y, a, Z, N):
    r = np.sqrt(x**2 + y**2)
    if r <= a:
        Er = Z * e * N * r / (2 * pi * epsilon0 * a**2)
    elif r > (0.1 - 1e-10):
        Er = 0
    else:
        Er = Z * e * N / (2 * pi * epsilon0 * r)
    # print(Er)
    return float(Er)


def cal_reletive_error(measured, theoretical, eps=1e-10):
    abs_error = np.abs(measured - theoretical)
    mask_zero = np.abs(theoretical) < eps
    rel_error = np.zeros_like(theoretical)
    # 非零区域计算相对误差
    rel_error[~mask_zero] = abs_error[~mask_zero] / np.abs(theoretical[~mask_zero])
    # 零值区域标记无穷大
    rel_error[mask_zero] = np.inf
    # 若测量值也为零，则误差为0
    mask_exact = mask_zero & (np.abs(measured) < eps)
    rel_error[mask_exact] = 0

    return rel_error * 100


def read_Er(filepath, Lx, Ly):
    data = np.loadtxt(filepath, delimiter=",")
    Ny, Nx = data.shape

    x_coord = np.arange(
        -(Nx - 1) / 2 * Lx,
        ((Nx - 1) / 2 + 1) * Lx,
        step=Lx,
    )
    y_coord = np.arange(
        -(Ny - 1) / 2 * Ly,
        ((Ny - 1) / 2 + 1) * Ly,
        step=Ly,
    )
    # print(x_coord)

    x, y = np.meshgrid(x_coord, y_coord, indexing="xy")

    vectorized_func = np.vectorize(cal_Er_KV)
    data_theory = vectorized_func(x, y, a=np.sqrt(200e-6 * 5) * 2, Z=1, N=1e11 / 2)

    np.savetxt(r"D:\PASS\test\electricField_r_theory.csv", data_theory, delimiter=",")

    diff = data - data_theory
    # print(data_theory)

    error = cal_reletive_error(data, data_theory)

    fig, ax = plt.subplots()

    # mesh = plt.pcolormesh(x, y, data, cmap="viridis", shading="auto")
    # mesh = plt.pcolormesh(x, y, data_theory, cmap="viridis", shading="auto")
    # mesh = plt.pcolormesh(x, y, diff, cmap="viridis", shading="auto")
    mesh = plt.pcolormesh(x, y, error, cmap="viridis", shading="auto")

    max_error = np.max(abs(diff))
    mean_squared_error = np.mean(diff**2)
    print(f"最大绝对误差: {max_error:.6f}")
    print(f"均方误差 (MSE): {mean_squared_error:.6f}")

    plt.colorbar(mesh, label="Z Value")  # 添加颜色条
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Z Values on 2D Grid")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    read_Er(r"D:\PASS\test\electricField_r_slice0_turn_1_.csv", Lx=2e-3, Ly=2e-3)
