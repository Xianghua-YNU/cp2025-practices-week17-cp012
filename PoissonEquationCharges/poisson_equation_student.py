#!/usr/bin/env python3
"""
学生模板：求解正负电荷构成的泊松方程
文件：poisson_equation_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    使用松弛迭代法求解二维泊松方程

    参数:
        M (int): 每边的网格点数，默认100
        target (float): 收敛精度，默认1e-6
        max_iterations (int): 最大迭代次数，默认10000

    返回:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): 电势分布数组，形状为(M+1, M+1)
            iterations (int): 实际迭代次数
            converged (bool): 是否收敛

    物理背景:
        求解泊松方程 ∇²φ = -ρ/ε₀，其中：
        - φ 是电势
        - ρ 是电荷密度分布
        - 边界条件：四周电势为0
        - 正电荷位于 (60:80, 20:40)，密度 +1 C/m²
        - 负电荷位于 (20:40, 60:80)，密度 -1 C/m²

    数值方法:
        使用有限差分法离散化，迭代公式：
        φᵢⱼ = 0.25 * (φᵢ₊₁ⱼ + φᵢ₋₁ⱼ + φᵢⱼ₊₁ + φᵢⱼ₋₁ + h²ρᵢⱼ)
    """
    # 设置网格间距
    h = 1.0

    # 初始化电势数组，形状为(M+1, M+1)
    phi = np.zeros((M + 1, M + 1)，dtype=float)
    # 创建前一步的电势数组副本
    phi_prev = np.copy(phi)
    # 创建电荷密度数组
    rho = np.zeros((M + 1, M + 1),dtype=float)

    # 设置电荷分布
    pos_y1, pos_y2 = int(0.6*M), int(0.8*M)
    pos_x1, pos_x2 = int(0.2*M), int(0.4*M)
    neg_y1, neg_y2 = int(0.2*M), int(0.4*M)
    neg_x1, neg_x2 = int(0.6*M), int(0.8*M)
    
    rho[pos_y1:pos_y2, pos_x1:pos_x2] = 1.0   
    rho[neg_y1:neg_y2, neg_x1:neg_x2] = -1.0  
    # 初始化迭代变量
    delta = 1.0  # 用于存储最大变化量
    iterations = 0  # 迭代计数器
    converged = False  # 收敛标志



    # 主迭代循环
    while delta > target and iterations < max_iterations:
        # 使用有限差分公式更新内部网格点
        phi[1:-1, 1:-1] = 0.25 * (phi[2:, 1:-1] + phi[0:-2, 1:-1] +
                                  phi[1:-1, 2:] + phi[1:-1, :-2] +
                                  h ** 2 * rho[1:-1, 1:-1])

        # 计算最大变化量
        delta = np.max(np.abs(phi - phi_prev))

        # 更新前一步解
        phi_prev = np.copy(phi)

        # 增加迭代计数
        iterations += 1

    # 检查是否收敛
    converged = (delta <= target)

    return phi, iterations, converged


def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
    """
    可视化电势分布

    参数:
        phi (np.ndarray): 电势分布数组
        M (int): 网格大小
    """
    # 创建网格
    plt.figure(figsize=(10, 8))
    W= plt.imshow(phi, extent=[0, M, 0, M], origin='lower',
                    cmap='RdBu_r', interpolation='bilinear')
    cbar = plt.colorbar(W)
    cbar.set_label('Electric Potential (V)')
    plt.fill_between([20, 40], [60, 60], [80, 80], alpha=0.3, color='red', label='Positive Charge')
    plt.fill_between([60, 80], [20, 20], [40, 40], alpha=0.3, color='blue', label='Negative Charge')

    plt.xlabel('x (grid points)')
    plt.ylabel('y (grid points)')
    plt.title('Electric Potential Distribution Poisson Equation')
    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    分析并打印解的基本信息

    参数:
        phi (np.ndarray): 电势分布数组
        iterations (int): 迭代次数
        converged (bool): 收敛状态
    """
    print(f"迭代次数: {iterations}")
    print(f"是否收敛: {'是' if converged else '否'}")
    print(f"最大电势: {np.max(phi):.6f} V")
    print(f"最小电势: {np.min(phi):.6f} V")
    print(f"  Potential range: {np.max(phi) - np.min(phi):.6f} V")
    # 找到极值位置
    max_pos = np.unravel_index(np.argmax(phi), phi.shape)
    min_pos = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"最大电势位置: ({max_pos[0]}, {max_pos[1]})")
    print(f"最小电势位置: ({min_pos[0]}, {min_pos[1]})")


if __name__ == "__main__":
    # 测试代码区域
    print("开始求解二维泊松方程...")

    # 设置参数
    M = 100
    target = 1e-6
    max_iter = 10000

    # 调用求解函数
    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)

    # 分析结果
    analyze_solution(phi, iterations, converged)

    # 可视化结果
    visualize_solution(phi, M)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    center_y = M // 2
    plt.plot(phi[center_y, :], 'b-', linewidth=2)
    plt.xlabel('x (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along y = {center_y}')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    center_x = M // 2
    plt.plot(phi[:, center_x], 'r-', linewidth=2)
    plt.xlabel('y (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along x = {center_x}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
