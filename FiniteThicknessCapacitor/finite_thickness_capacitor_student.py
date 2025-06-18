#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using SOR method for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        np.ndarray: 2D electric potential distribution
    """
    # TODO: Implement SOR iteration for Laplace equation
    # 初始化电势网格
    potential = np.zeros((ny, nx))

    # 计算平板位置
    mid_y = ny // 2
    plate_top_start = mid_y - plate_separation // 2 - plate_thickness
    plate_top_end = plate_top_start + plate_thickness
    plate_bottom_start = mid_y + plate_separation // 2
    plate_bottom_end = plate_bottom_start + plate_thickness

    # 设置边界条件
    potential[plate_top_start:plate_top_end, :] = 100.0  # 上极板
    potential[plate_bottom_start:plate_bottom_end, :] = -100.0  # 下极板
    potential[:, 0] = 0.0  # 左边界
    potential[:, -1] = 0.0  # 右边界

    # 标记导体区域（不参与迭代）
    is_conductor = np.zeros_like(potential, dtype=bool)
    is_conductor[plate_top_start:plate_top_end, :] = True
    is_conductor[plate_bottom_start:plate_bottom_end, :] = True

    # SOR迭代
    for iter_count in range(max_iter):
        max_diff = 0.0
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                if not is_conductor[j, i]:
                    old_value = potential[j, i]
                    # SOR更新公式
                    potential[j, i] = (1 - omega) * old_value + omega * 0.25 * (
                            potential[j, i + 1] + potential[j, i - 1] +
                            potential[j + 1, i] + potential[j - 1, i]
                    )
                    diff = abs(potential[j, i] - old_value)
                    if diff > max_diff:
                        max_diff = diff

        # 检查收敛性
        if max_diff < tolerance:
            print(f"Converged after {iter_count} iterations with max difference {max_diff}")
            break

    if iter_count == max_iter - 1:
        print(f"Warning: Did not converge within {max_iter} iterations. Max difference: {max_diff}")

    return potential

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # TODO: Calculate charge density from potential
     # 使用中心差分计算拉普拉斯算子
    laplacian = np.zeros_like(potential_grid)

    # 内部点
    for j in range(1, potential_grid.shape[0] - 1):
        for i in range(1, potential_grid.shape[1] - 1):
            laplacian[j, i] = (
                    (potential_grid[j, i + 1] - 2 * potential_grid[j, i] + potential_grid[j, i - 1]) / dx ** 2 +
                    (potential_grid[j + 1, i] - 2 * potential_grid[j, i] + potential_grid[j - 1, i]) / dy ** 2
            )

    # 计算电荷密度 ρ = -∇²U / (4π)
    charge_density = -laplacian / (4 * np.pi)

    return charge_density

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
    """
    # TODO: Implement visualization
    plt.figure(figsize=(15, 6))

    # 电势分布等高线图
    plt.subplot(1, 2, 1)
    contour = plt.contourf(x_coords, y_coords, potential, levels=50, cmap='viridis')
    plt.colorbar(label='Electric Potential (V)')
    # 绘制电场线（电势梯度的负方向）
    dy, dx = np.gradient(potential)
    # 间隔采样以避免箭头过密
    skip = (slice(None, None, 5), slice(None, None, 5))

    # 创建网格以匹配梯度数组的维度
    X, Y = np.meshgrid(x_coords, y_coords)

    # 使用网格坐标和采样后的梯度绘制电场线
    plt.quiver(X[skip], Y[skip], -dx[skip], -dy[skip], color='white', scale=2000)
    plt.contour(x_coords, y_coords, potential, levels=10, colors='white', linewidths=0.5)
    plt.title('Electric Potential Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    # 电荷密度分布图
    plt.subplot(1, 2, 2)
    # 只显示导体表面附近的电荷密度
    charge_display = np.copy(charge_density)
    # 屏蔽非导体区域
    mask = np.abs(charge_density) < 1e-10
    charge_display[mask] = np.nan
    plt.imshow(charge_display, extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
               cmap='coolwarm', origin='lower')
    plt.colorbar(label='Charge Density')
    plt.title('Charge Density Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # TODO: Set simulation parameters and call functions
    # 设置模拟参数
    nx, ny = 100, 100  # 网格点数
    Lx, Ly = 1.0, 1.0  # 模拟区域大小（米）
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)  # 网格间距

    plate_thickness = 5  # 极板厚度（网格点）
    plate_separation = 30  # 极板间距（网格点）

    # 生成坐标数组
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)

    # 求解电势分布
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation)

    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)

    # 可视化结果
    plot_results(potential, charge_density, x_coords, y_coords)
