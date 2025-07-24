import numpy as np
from scipy.sparse import csr_matrix
import csv
import pandas as pd


def generate_matrix(Nx, Ny, Lx, Ly):

    a = 1 / Lx / Lx
    b = 1 / Ly / Ly
    c = -2 * (a + b)

    boundary_vertices = [(0, 0), (0, Nx - 1), (Ny - 1, 0), (Ny - 1, Nx - 1)]

    boundary_top = [(0, j) for j in range(1, Nx - 1)]
    boundary_bottom = [(Ny - 1, j) for j in range(1, Nx - 1)]
    boundary_left = [(i, 0) for i in range(1, Ny - 1)]
    boundary_right = [(i, Nx - 1) for i in range(1, Ny - 1)]

    sub_boundary_vertices = [(1, 1), (1, Nx - 2), (Ny - 2, 1), (Ny - 2, Nx - 2)]
    sub_boundary_top = [(1, j) for j in range(2, Nx - 2)]
    sub_boundary_bottom = [(Ny - 2, j) for j in range(2, Nx - 2)]
    sub_boundary_left = [(i, 1) for i in range(2, Ny - 2)]
    sub_boundary_right = [(i, Nx - 2) for i in range(2, Ny - 2)]

    inner_points = [(i, j) for i in range(2, Ny - 2) for j in range(2, Nx - 2)]

    A = [0 for i in range(Nx * Ny * Nx * Ny)]

    for row, col in boundary_vertices:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
    for row, col in boundary_top:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
    for row, col in boundary_bottom:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
    for row, col in boundary_left:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
    for row, col in boundary_right:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c

    (row, col) = (1, 1)  # (1, Nx - 2), (Ny - 2, 1), (Ny - 2, Nx - 2)
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = c
    A[index + 1] = a
    A[index + Nx] = b

    (row, col) = (1, Nx - 2)  # , (Ny - 2, 1), (Ny - 2, Nx - 2)
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = c
    A[index - 1] = a
    A[index + Nx] = b

    (row, col) = (Ny - 2, 1)  # , , (Ny - 2, Nx - 2)
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = c
    A[index + 1] = a
    A[index - Nx] = b

    (row, col) = (Ny - 2, Nx - 2)  # , ,
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = c
    A[index - 1] = a
    A[index - Nx] = b

    for row, col in sub_boundary_top:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
        A[index + 1] = a
        A[index - 1] = a
        A[index + Nx] = b
    for row, col in sub_boundary_bottom:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
        A[index + 1] = a
        A[index - 1] = a
        A[index - Nx] = b
    for row, col in sub_boundary_left:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
        A[index + 1] = a
        A[index + Nx] = b
        A[index - Nx] = b
    for row, col in sub_boundary_right:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
        A[index - 1] = a
        A[index + Nx] = b
        A[index - Nx] = b

    for row, col in inner_points:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = c
        A[index + 1] = a
        A[index - 1] = a
        A[index + Nx] = b
        A[index - Nx] = b

    A_2d = np.array(A).reshape(Nx * Ny, Nx * Ny)

    # pprint(A_2d, width=20)
    # np.savetxt(
    #     r"D:\pythonTest\output.csv", A_2d, delimiter=",", fmt="%s", encoding="utf-8"
    # )
    sparse_csr = csr_matrix(A_2d)
    row_ptr = sparse_csr.indptr  # 行指针数组
    col_indices = sparse_csr.indices  # 列索引数组
    values = sparse_csr.data  # 非零值数组

    max_len = max(len(row_ptr), len(col_indices), len(values))

    with open(r"D:\pythonTest\output_csr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row_ptr)  # 第一行：row_ptr
        writer.writerow(col_indices)  # 第二行：col_indices
        writer.writerow(values)  # 第三行：values

    return A_2d


def generate_matrix2(Nx, Ny):

    boundary_vertices = [(0, 0), (0, Nx - 1), (Ny - 1, 0), (Ny - 1, Nx - 1)]

    boundary_top = [(0, j) for j in range(1, Nx - 1)]
    boundary_bottom = [(Ny - 1, j) for j in range(1, Nx - 1)]
    boundary_left = [(i, 0) for i in range(1, Ny - 1)]
    boundary_right = [(i, Nx - 1) for i in range(1, Ny - 1)]

    inner_points = [(i, j) for i in range(1, Ny - 1) for j in range(1, Nx - 1)]

    A = [0 for i in range(Nx * Ny * Nx * Ny)]

    (row, col) = (0, 0)  # [(0, 0), (0, Nx - 1), (Ny - 1, 0), (Ny - 1, Nx - 1)]
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = -4
    A[index + 1] = 1
    A[index + Nx] = 1

    (row, col) = (0, Nx - 1)  # , (Ny - 2, 1), (Ny - 2, Nx - 2)
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = -4
    A[index - 1] = 1
    A[index + Nx] = 1

    (row, col) = (Ny - 1, 0)  # , , (Ny - 2, Nx - 2)
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = -4
    A[index + 1] = 1
    A[index - Nx] = 1

    (row, col) = (Ny - 1, Nx - 1)  # , ,
    center = row * Nx + col
    index = center * (Nx * Ny) + center
    A[index] = -4
    A[index - 1] = 1
    A[index - Nx] = 1

    for row, col in boundary_top:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = -4
        A[index + 1] = 1
        A[index - 1] = 1
        A[index + Nx] = 1
    for row, col in boundary_bottom:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = -4
        A[index + 1] = 1
        A[index - 1] = 1
        A[index - Nx] = 1
    for row, col in boundary_left:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = -4
        A[index + 1] = 1
        A[index + Nx] = 1
        A[index - Nx] = 1
    for row, col in boundary_right:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = -4
        A[index - 1] = 1
        A[index + Nx] = 1
        A[index - Nx] = 1

    for row, col in inner_points:
        center = row * Nx + col
        index = center * (Nx * Ny) + center
        A[index] = -4
        A[index + 1] = 1
        A[index - 1] = 1
        A[index + Nx] = 1
        A[index - Nx] = 1

    A_2d = np.array(A).reshape(Nx * Ny, Nx * Ny)

    # pprint(A_2d, width=20)
    np.savetxt(
        r"D:\pythonTest\output.csv", A_2d, delimiter=",", fmt="%s", encoding="utf-8"
    )


def convert_CSR_to_dense(filename):
    # 读取CSV文件的三行数据
    row_ptr = np.loadtxt(filename, delimiter=",", max_rows=1, dtype=np.int32)
    col_indices = np.loadtxt(
        filename, delimiter=",", skiprows=1, max_rows=1, dtype=np.int32
    )
    values = np.loadtxt(filename, delimiter=",", skiprows=2, dtype=np.float32)

    # 校验数据一致性
    assert len(row_ptr) > 1, "row_ptr长度至少为2"
    assert len(col_indices) == len(values), "列索引与值数组长度需一致"

    # 计算矩阵维度
    n_rows = len(row_ptr) - 1
    n_cols = np.max(col_indices) + 1  # 最大列索引+1为列数

    # 创建CSR矩阵
    sparse_mat = csr_matrix((values, col_indices, row_ptr), shape=(n_rows, n_cols))

    # 转换为稠密矩阵（NumPy数组）
    dense_matrix = sparse_mat.toarray()  # 或 sparse_mat.todense()

    np.savetxt(
        r"D:\PASS\test\rebuilt_from_CSR.csv",
        dense_matrix,
        delimiter=",",
        fmt="%s",
        encoding="utf-8",
    )


def solve_matrix(Nx, Ny, Lx, Ly):
    A = generate_matrix(Nx, Ny, Lx, Ly)

    b = np.loadtxt(
        r"D:\PASS\test\charDensity.csv",
        delimiter=",",
        skiprows=0,
        unpack=False,
        usecols=(0, 1, 2),
    )

    x = np.linalg.solve(A, b)

    np.savetxt(
        r"D:\pythonTest\output_potential.csv",
        x,
        delimiter=",",
        fmt="%s",
        encoding="utf-8",
    )


if __name__ == "__main__":
    # generate_matrix(Nx=65, Ny=65, Lx=4e-3, Ly=2e-3)
    # generate_matrix2(5, 5)
    # convert_CSR_to_dense(r"D:\PASS\test\fd_matrix_csr.csv")
    solve_matrix(Nx=65, Ny=65, Lx=4e-3, Ly=2e-3)
