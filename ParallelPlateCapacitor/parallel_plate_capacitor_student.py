import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    
    说明:
        使用Jacobi迭代法求解拉普拉斯方程 (\(\nabla^2 V = 0\))，模拟平行板电容器的电势分布。
        初始化电势网格，设置边界条件（极板电势），迭代更新内部点的电势值直至收敛。
    """
    # 初始化电势网格，所有内部点初始电势为0
    u = np.zeros((ygrid, xgrid))
    
    # 计算极板位置
    xL = (xgrid - w) // 2  # 左侧极板x坐标
    xR = (xgrid + w) // 2  # 右侧极板x坐标（包含极板宽度w）
    yB = (ygrid - d) // 2  # 下极板y坐标
    yT = (ygrid + d) // 2  # 上极板y坐标（极板间距d）
    
    # 设置极板边界条件
    # 上极板电势+100V，下极板电势-100V
    u[yT, xL:xR + 1] = 100.0  
    u[yB, xL:xR + 1] = -100.0
    
    # 初始化迭代计数和收敛历史记录
    iterations = 0
    max_iter = 10000  # 最大迭代次数限制，防止无限循环
    convergence_history = []
    
    # 开始迭代过程
    while iterations < max_iter:
        # 保存当前电势分布用于后续收敛检查
        u_old = u.copy()
        
        # Jacobi迭代更新内部点的电势值
        # 使用矢量化操作提高效率，更新所有内部点（不包括边界）
        u[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] +  # 上下点
            u[1:-1, 2:] + u[1:-1, :-2]   # 左右点
        )
        
        # 重新应用极板边界条件（迭代可能影响极板附近的点）
        u[yT, xL:xR + 1] = 100.0
        u[yB, xL:xR + 1] = -100.0
        
        # 计算电势变化的最大值作为收敛指标
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        # 检查是否达到收敛条件
        iterations += 1
        if max_change < tol:
            break
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    使用Gauss-Seidel SOR迭代法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子（推荐范围1.0-2.0）
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    
    说明:
        使用Gauss-Seidel SOR迭代法求解拉普拉斯方程 (\(\nabla^2 V = 0\))，
        在每次迭代中，新计算的电势值立即用于后续的计算，加速收敛。
    """
    # 初始化电势网格，所有内部点初始电势为0
    u = np.zeros((ygrid, xgrid))
    
    # 计算极板位置
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # 设置极板边界条件
    u[yT, xL:xR + 1] = 100.0
    u[yB, xL:xR + 1] = -100.0
    
    convergence_history = []
    
    for iteration in range(Niter):
        u_old = u.copy()
        
        # 逐点进行Gauss-Seidel SOR迭代
        for i in range(1, ygrid - 1):
            for j in range(1, xgrid - 1):
                # 跳过极板位置（确保不修改极板电势）
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue
                
                # 计算邻点电势的平均值
                r_ij = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])
                
                # 应用SOR公式更新电势值
                u[i, j] = (1 - omega) * u[i, j] + omega * r_ij
        
        # 重新应用极板边界条件
        u[yT, xL:xR + 1] = 100.0
        u[yB, xL:xR + 1] = -100.0
        
        # 计算电势变化的最大值作为收敛指标
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        # 检查是否达到收敛条件
        if max_change < tol:
            break
    
    return u, iteration + 1, convergence_history

def plot_results(x, y, u, method_name):
    """
    可视化电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    
    说明:
        绘制3D电势分布图和等势线投影图，并计算电场线进行可视化。
        电场线通过电势的梯度计算得到。
    """
    fig = plt.figure(figsize=(10, 5))
    
    # 创建3D电势分布图
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax1.plot_wireframe(X, Y, u, alpha=0.7)
    
    # 绘制等势线投影
    levels = np.linspace(u.min(), u.max(), 20)
    ax1.contour(x, y, u, zdir='z', offset=u.min(), levels=levels)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('电势 (V)')
    ax1.set_title(f'3D电势分布\n({method_name})')
    
    # 创建等势线和电场线图
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 20)
    contour = ax2.contour(X, Y, u, levels=levels, colors='red', linestyles='dashed', linewidths=0.8)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
    
    # 计算电场线（电场是电势的负梯度）
    EY, EX = np.gradient(-u, 1)  # np.gradient返回y方向梯度，然后是x方向梯度
    
    # 绘制电场线流线图
    ax2.streamplot(X, Y, EX, EY, density=1.5, color='blue', linewidth=1, arrowsize=1.5, arrowstyle='->')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'等势线与电场线\n({method_name})')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 模拟参数设置
    xgrid, ygrid = 50, 50  # 网格尺寸
    w, d = 20, 20          # 平行板宽度和间距
    tol = 1e-3             # 收敛容差
    
    # 创建坐标网格
    x = np.linspace(0, xgrid - 1, xgrid)
    y = np.linspace(0, ygrid - 1, ygrid)
    
    print("求解平行板电容器的拉普拉斯方程...")
    print(f"网格尺寸: {xgrid} x {ygrid}")
    print(f"极板宽度: {w}, 极板间距: {d}")
    print(f"收敛容差: {tol}")
    
    # 使用Jacobi方法求解
    print("1. 使用Jacobi迭代法:")
    start_time = time.time()
    u_jacobi, iter_jacobi, conv_history_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d, tol=tol)
    time_jacobi = time.time() - start_time
    print(f"   在{iter_jacobi}次迭代后收敛")
    print(f"   用时: {time_jacobi:.3f}秒")
    
    # 使用SOR方法求解
    print("2. 使用Gauss-Seidel SOR迭代法:")
    start_time = time.time()
    u_sor, iter_sor, conv_history_sor = solve_laplace_sor(xgrid, ygrid, w, d, tol=tol)
    time_sor = time.time() - start_time
    print(f"   在{iter_sor}次迭代后收敛")
    print(f"   用时: {time_sor:.3f}秒")
    
    # 比较两种方法性能
    print("\n3. 性能比较:")
    print(f"   Jacobi方法: {iter_jacobi}次迭代, {time_jacobi:.3f}秒")
    print(f"   SOR方法:    {iter_sor}次迭代, {time_sor:.3f}秒")
    print(f"   迭代次数加速比: {iter_jacobi/iter_sor:.1f}倍")
    print(f"   时间加速比: {time_jacobi/time_sor:.2f}倍")
    
    # 绘制结果
    plot_results(x, y, u_jacobi, "Jacobi迭代法")
    plot_results(x, y, u_sor, "SOR迭代法")
    
    # 绘制收敛历史比较图
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(conv_history_jacobi)), conv_history_jacobi, 'r-', label='Jacobi迭代法')
    plt.semilogy(range(len(conv_history_sor)), conv_history_sor, 'b-', label='SOR迭代法')
    plt.xlabel('迭代次数')
    plt.ylabel('最大电势变化量')
    plt.title('收敛速度比较')
    plt.grid(True)
    plt.legend()
    plt.show()
