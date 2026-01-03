import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

def gradient_descent():
    def f(x):
        return (x - 2)**2 + 4

    def grad_f(x):
        return 2*(x - 2)

    def run_gradient_descent(x0=-4.0, lr=0.2, steps=12):
        xs = [x0]
        for _ in range(steps):
            x = xs[-1]
            xs.append(x - lr * grad_f(x))
        return np.array(xs)

    x_plot = np.linspace(-5, 5, 400)
    xs = run_gradient_descent()
    ys = f(xs)

    x_final = xs[-1]
    y_final = f(x_final)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(x_plot, f(x_plot))
    axes[0].set_title("f(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")

    axes[1].axhline(0, linewidth=1, color="red")
    axes[1].plot(x_plot, grad_f(x_plot))
    axes[1].set_title("f'(x) (gradient)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f'(x)")

    axes[2].plot(x_plot, f(x_plot), label="f(x)")
    axes[2].scatter(xs, ys, color="red")

    for i in range(len(xs) - 1):
        axes[2].annotate(
            "",
            xy=(xs[i+1], ys[i+1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->")
        )
    
    axes[2].annotate(
        f"x_final = {x_final:.4f}",
        xy=(x_final, y_final),
        xytext=(x_final , y_final + 10),
    )

    axes[2].set_title("Trajectoire du gradient descent")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("f(x)")

    plt.tight_layout()
    plt.show()

def gradient_descent_2D():
    def f(x, y):
        return (x - 2)**2 + 4 + (y - 2)**2 + 4

    def grad_f(x, y):
        return 2*(x - 2), 2*(y - 2)

    def run_gradient_descent(x0, y0, lr=0.2, steps=12):
        xs = [x0]
        ys = [y0]

        for _ in range(steps):
            gx, gy = grad_f(xs[-1], ys[-1])
            x = xs[-1]
            y = ys[-1]
            xs.append(x - lr * gx)
            ys.append(y - lr * gy)

        return np.array(xs), np.array(ys)

    x0, y0 = -4.0, 4.0
    xs, ys = run_gradient_descent(x0, y0)
    x_final, y_final = xs[-1], ys[-1]

    X, Y = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    Z = f(X, Y)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].contour(X, Y, Z, levels=20)
    axes[0].set_title("f(x, y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    GX, GY = grad_f(X, Y)
    axes[1].quiver(X[::10, ::10], Y[::10, ::10], -GX[::10, ::10], -GY[::10, ::10])
    axes[1].set_title("f'(x,y) (gradient)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    axes[2].contour(X, Y, Z, levels=20)
    axes[2].scatter(xs, ys, color="red")

    for i in range(len(xs) - 1):
        axes[2].annotate(
            "",
            xy=(xs[i+1], ys[i+1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->")
        )

    axes[2].annotate(
        f"(x*, y*) = ({x_final:.2f}, {y_final:.2f})",
        xy=(x_final, y_final),
        xytext=(x_final + 0.5, y_final + 0.5)
    )

    axes[2].set_title("Trajectoire du gradient descent")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    plt.tight_layout()
    plt.show()