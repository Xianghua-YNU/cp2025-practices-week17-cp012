"""学生模板：ParallelPlateCapacitor
文件：parallel_plate_capacitor_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    使用Jacobi迭代法求解拉普拉斯方程
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        tol (float): 收敛容差
    
    返回:
        tuple: (potential_array, iterations, convergence_history)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用Jacobi迭代法，通过反复迭代更新每个网格点的电势值，直至收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点的电势更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    u_new = u.copy()
    
    # 设置平行板电势
    xL = (xgrid - w) // 2
    xR = xL + w
    yB = (ygrid - d) // 2
    yT = yB + d
    
    u[yT:yT+1, xL:xR] = 100
    u[yB-1:yB, xL:xR] = -100
    
    iterations = 0
    convergence_history = []
    
    start_time = time.time()
    
    while True:
        # 在每次迭代中保持旧网格的副本
        u_old = u.copy()
        
        # 更新内部点的电势
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板的位置
                if (yB <= i < yT) and (xL <= j < xR):
                    continue
                
                u_new[i, j] = 0.25 * (u_old[i+1, j] + u_old[i-1, j] + 
                                     u_old[i, j+1] + u_old[i, j-1])
        
        # 更新u为新的值
        u = u_new.copy()
        
        iterations += 1
        
        # 记录最大变化量
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        # 检查收敛条件
        if max_change < tol:
            break
    
    elapsed_time = time.time() - start_time
    print(f"Jacobi方法: 迭代次数={iterations}, 计算时间={elapsed_time:.4f}秒")
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    实现SOR算法求解平行板电容器的电势分布
    
    参数:
        xgrid (int): x方向网格点数
        ygrid (int): y方向网格点数
        w (int): 平行板宽度
        d (int): 平行板间距
        omega (float): 松弛因子
        Niter (int): 最大迭代次数
        tol (float): 收敛容差
    返回:
        tuple: (电势分布数组, 迭代次数, 收敛历史)
    
    物理背景: 求解平行板电容器内部的电势分布，满足拉普拉斯方程 \(\nabla^2 V = 0\)。
    数值方法: 使用逐次超松弛（SOR）迭代法，通过引入松弛因子加速收敛。
    
    实现步骤:
    1. 初始化电势网格，设置边界条件（极板电势）。
    2. 循环迭代，每次迭代根据周围点和松弛因子更新当前点的电势。
    3. 记录每次迭代的最大变化量，用于收敛历史分析。
    4. 检查收敛条件，如果最大变化量小于容差，则停止迭代。
    5. 返回最终的电势分布、迭代次数和收敛历史。
    """
    # 初始化电势网格
    u = np.zeros((ygrid, xgrid))
    
    # 设置平行板电势
    xL = (xgrid - w) // 2
    xR = xL + w
    yB = (ygrid - d) // 2
    yT = yB + d
    
    u[yT:yT+1, xL:xR] = 100
    u[yB-1:yB, xL:xR] = -100
    
    iterations = 0
    convergence_history = []
    
    start_time = time.time()
    
    for _ in range(Niter):
        u_old = u.copy()
        
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # 跳过平行板的位置
                if (yB <= i < yT) and (xL <= j < xR):
                    continue
                
                # Gauss-Seidel SOR 迭代公式
                r_ij = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
                u[i, j] = (1 - omega) * u[i, j] + omega * r_ij
        
        iterations += 1
        
        # 记录最大变化量
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)
        
        # 检查收敛条件
        if max_change < tol:
            break
    
    elapsed_time = time.time() - start_time
    print(f"SOR方法 (ω={omega}): 迭代次数={iterations}, 计算时间={elapsed_time:.4f}秒")
    
    return u, iterations, convergence_history

def plot_results(x, y, u, method_name):
    """
    绘制三维电势分布、等势线和电场线
    
    参数:
        x (array): X坐标数组
        y (array): Y坐标数组
        u (array): 电势分布数组
        method_name (str): 方法名称
    
    实现步骤:
    1. 创建包含两个子图的图形。
    2. 在第一个子图中绘制三维线框图显示电势分布以及在z方向的投影等势线。
    3. 在第二个子图中绘制等势线和电场线流线图。
    4. 设置图表标题、标签和显示(注意不要出现乱码)。
    """
    plt.style.use('seaborn-v0_8-dark-palette')
    fig = plt.figure(figsize=(15, 6))
    
    # 三维电势分布
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_wireframe(X, Y, u, rstride=2, cstride=2, color='blue', linewidth=0.5)
    ax1.set_title(f"{method_name}方法: 电势分布", fontsize=12)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('电势 (V)', fontsize=10)
    
    # 等势线和电场线
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, u, 20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.1fV')
    
    # 计算电场
    Ey, Ex = np.gradient(-u)  # 计算电场分量
    
    # 绘制电场线
    ax2.quiver(X[::3, ::3], Y[::3, ::3], Ex[::3, ::3], Ey[::3, ::3], 
               scale=50, color='red', width=0.003)
    
    ax2.set_title(f"{method_name}方法: 等势线和电场线", fontsize=12)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 示例用法
    xgrid = 101  # x方向网格点数
    ygrid = 101  # y方向网格点数
    w = 40       # 平行板宽度
    d = 20       # 平行板间距

    x = np.linspace(0, xgrid-1, xgrid)
    y = np.linspace(0, ygrid-1, ygrid)

    # 使用Jacobi方法
    print("运行Jacobi方法...")
    u_jacobi, iter_jacobi, _ = solve_laplace_jacobi(xgrid, ygrid, w, d)
    plot_results(x, y, u_jacobi, "Jacobi")

    # 使用SOR方法
    print("运行SOR方法...")
    u_sor, iter_sor, _ = solve_laplace_sor(xgrid, ygrid, w, d, omega=1.5)
    plot_results(x, y, u_sor, "SOR")

    print(f"Jacobi方法迭代次数: {iter_jacobi}")
    print(f"SOR方法迭代次数: {iter_sor}")
if __name__ == "__main__":
    # 示例用法（学生可以取消注释并在此处测试他们的实现）
    pass
