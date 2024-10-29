import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def f(x):
    """
    目标函数 f(x) = e^{x1 + 3x2 -0.1} + e^{x1 - 3x2 -0.1} + e^{-x1 -0.1}
    
    参数:
    - x: numpy数组，形状为 (2,)
    
    返回:
    - 标量 f(x)
    """
    x1, x2 = x
    try:
        term1 = np.exp(x1 + 3 * x2 - 0.1)
        term2 = np.exp(x1 - 3 * x2 - 0.1)
        term3 = np.exp(-x1 - 0.1)
        return term1 + term2 + term3
    except OverflowError:
        return np.inf  # 当指数溢出时返回无穷大

def grad_f(x):
    """
    目标函数的梯度 ∇f(x)
    
    参数:
    - x: numpy数组，形状为 (2,)
    
    返回:
    - numpy数组，形状为 (2,)
    """
    x1, x2 = x
    try:
        df_dx1 = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1)
        df_dx2 = 3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
        return np.array([df_dx1, df_dx2])
    except OverflowError:
        return np.array([np.inf, np.inf])  # 当指数溢出时返回无穷大

def hessian_f(x):
    """
    目标函数的Hessian矩阵 ∇²f(x)
    
    参数:
    - x: numpy数组，形状为 (2,)
    
    返回:
    - numpy数组，形状为 (2, 2)
    """
    x1, x2 = x
    try:
        d2f_dx1dx1 = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)
        d2f_dx1dx2 = 3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
        d2f_dx2dx2 = 9 * np.exp(x1 + 3 * x2 - 0.1) + 9 * np.exp(x1 - 3 * x2 - 0.1)
        return np.array([[d2f_dx1dx1, d2f_dx1dx2],
                         [d2f_dx1dx2, d2f_dx2dx2]])
    except OverflowError:
        return np.array([[np.inf, np.inf],
                         [np.inf, np.inf]])  # 当指数溢出时返回无穷大

def backtracking_line_search(x, d, alpha=0.1, beta=0.7):
    """
    回溯直线搜索，寻找满足Armijo条件的步长 t
    
    参数:
    - x: 当前点 (numpy数组，形状为 (2,))
    - d: 下降方向 (numpy数组，形状为 (2,))
    - alpha: Armijo条件的参数 (0 < alpha < 0.5)
    - beta: 步长缩小因子 (0 < beta < 1)
    
    返回:
    - t: 合适的步长
    - iter_count: 回溯迭代次数
    """
    t = 1.0
    iter_count = 0
    fx = f(x)
    grad_fx = grad_f(x)
    while True:
        x_new = x + t * d
        fx_new = f(x_new)
        if fx_new <= fx + alpha * t * np.dot(grad_fx, d):
            break
        t *= beta
        iter_count += 1
        if iter_count > 100:  # 防止无限循环
            print("回溯线搜索未找到合适的步长")
            break
    return t, iter_count

def newton_method(x_init, max_iters=50, tol=1e-6, alpha=0.1, beta=0.7):
    """
    牛顿方法进行优化
    
    参数:
    - x_init: 初始点 (numpy数组，形状为 (2,))
    - max_iters: 最大迭代次数
    - tol: 收敛阈值（梯度范数）
    - alpha: 回溯线搜索的alpha
    - beta: 回溯线搜索的beta
    
    返回:
    - x_opt: 优化后的点
    - history: 优化过程中所有点的列表
    - line_search_iters: 每次回溯搜索的迭代次数的列表
    """
    x = x_init.copy()
    history = [x.copy()]
    line_search_iters = []
    for i in range(max_iters):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"在第 {i} 次迭代时收敛。")
            break
        hess = hessian_f(x)
        if np.any(np.isinf(hess)):
            print("Hessian矩阵中存在无穷大，优化无法继续。")
            break
        try:
            # 计算牛顿方向
            d = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessian矩阵不可逆，优化无法继续。")
            break
        # 回溯直线搜索
        t, iter_count = backtracking_line_search(x, d, alpha, beta)
        # 更新点
        x = x + t * d
        history.append(x.copy())
        line_search_iters.append(iter_count)
        print(f"迭代 {i + 1}: x = {x}, f(x) = {f(x):.4f}, t = {t:.4f}, 回溯迭代次数 = {iter_count}")
    return x, history, line_search_iters

def plot_optimization(history, func):
    """
    绘制优化路径和函数等高线
    
    参数:
    - history: 优化过程中所有点的列表
    - func: 目标函数
    """
    history = np.array(history)
    x_vals = history[:, 0]
    y_vals = history[:, 1]
    
    """
    绘制 y_vals 的折线图

    参数:
    - history: 优化过程中所有点的列表
    """
    
    fy = []
    for i in range(history.shape[0]):
        fy.append(f(history[i]))

    plt.figure(figsize=(10, 6))
    plt.plot(fy, marker='o', linestyle='-', color='b')
    plt.title('newton method optimization alpha=0.1, beta=0.7')
    plt.xlabel('iteration')
    plt.ylabel('y_vals')
    plt.grid()
    plt.show()

    # 创建等高线图的网格
    x1 = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 400)
    x2 = np.linspace(min(y_vals) - 1, max(y_vals) + 1, 400)
    X1, X2 = np.meshgrid(x1, x2)
    # 计算Z值，处理溢出
    with np.errstate(over='ignore'):
        Z = np.exp(X1 + 3 * X2 - 0.1) + np.exp(X1 - 3 * X2 - 0.1) + np.exp(-X1 - 0.1)
    Z[np.isinf(Z)] = np.nan  # 将无穷大替换为NaN，避免绘图错误
    
    plt.figure(figsize=(12, 8))
    
    # 绘制等高线，使用LogNorm进行对数归一化
    contour_levels = np.logspace(-1, 3, 20)
    cp = plt.contour(X1, X2, Z, levels=contour_levels, norm=LogNorm(), cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)
    
    # 绘制优化路径
    plt.plot(x_vals, y_vals, 'ro-', label='Optimization Path')
    plt.plot(x_vals[0], y_vals[0], 'go', label='START')
    plt.plot(x_vals[-1], y_vals[-1], 'bo', label='END')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Newton Method Optimization alpha=0.1, beta=0.7')
    plt.legend()
    plt.colorbar(cp, label='$f(x)$')
    plt.grid(True)
    plt.show()

def main():
    # 定义初始点
    x_init = np.array([1.0, 1.0])  # 初始点可以更改
    print("初始点 x:", x_init)
    
    # 执行牛顿方法优化
    x_opt, history, line_search_iters = newton_method(x_init, max_iters=50, tol=1e-6, alpha=0.1, beta=0.7)
    
    print("\n优化后的点 x_opt:", x_opt)
    print("f(x_opt):", f(x_opt))
    
    # 可视化优化过程
    plot_optimization(history, f)

if __name__ == "__main__":
    main()
