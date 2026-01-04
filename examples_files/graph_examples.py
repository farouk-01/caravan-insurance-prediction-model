import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

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

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

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
        
    axes[2].set_title("Trajectoire du gradient descent")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    plt.tight_layout()
    plt.show()

def good_bad_PCA():
    np.random.seed(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # --- continue ---
    n = 150
    x_cont = np.linspace(0, 10, n)
    y_cont = 0.5 * x_cont + np.random.normal(0, 0.5, size=n)

    X_cont = np.column_stack([
        x_cont,
        0.5 * x_cont + np.random.normal(0, 1, size=n)
    ])

    Z_cont = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X_cont))

    sc0 = axes[0].scatter(Z_cont[:, 0], Z_cont[:, 1], c=y_cont, cmap="viridis", alpha=0.7)
    axes[0].set_title("PCA variable continue")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    plt.colorbar(sc0, ax=axes[0]).set_label("Effet")

    #--- ordinale ---
    x_ord = np.random.choice([0, 5, 6], size=n)
    x_ord2_arr = [0,1,2,3,4]
    x_ord2 = np.random.choice(x_ord2_arr, size=n)
    y_ord = (
        (x_ord == 6) * np.random.binomial(1, 0.6, size=n) * 6
        + np.random.normal(0, 0.3, size=n)
    )
    z = np.random.normal(0, 1.0, size=n)
    z_corr = z * 0.8 + np.random.normal(0, 1, size=n)
    scaler = StandardScaler()

    x = np.column_stack([
        x_ord,
        x_ord2,
        z,
        z_corr
    ])

    x[:, -2:] = scaler.fit_transform(x[:, -2:])

    Z_ord = PCA(n_components=2).fit_transform(x)

    mask_tp = y_ord > 3
    mask_fn = ~mask_tp

    axes[1].scatter(Z_ord[mask_tp, 0], Z_ord[mask_tp, 1],
                    c="#1f77b4", s=20, alpha=0.7, label="TP")

    axes[1].scatter(Z_ord[mask_fn, 0], Z_ord[mask_fn, 1],
                    c="orange", s=20, alpha=0.7, label="FN")
    axes[1].set_title("PCA variable ordinale")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    df_to_r = pd.DataFrame({
        'x_ord': x_ord,
        'x_ord2':x_ord2,
        'z': z,
        'z_corr': z_corr,
        'y_ord': y_ord
    })

    #--- ordinale one-hot ---
    levels = np.array([0, 5, 6])
    levels2 = np.array(x_ord2_arr)
    x_ohe = (x_ord[:, None] == levels).astype(int)
    x_ohe2 = (x_ord2[:, None] == levels2).astype(int)
    y_ord = (
        (x_ohe[:, 2] == 1) * np.random.binomial(1, 0.6, size=n) * 6
        + np.random.normal(0, 0.3, size=n)
    )
    x = np.column_stack([
        x_ohe,
        x_ohe2,
        z,
        z_corr          
    ])

    x[:, -2:] = scaler.fit_transform(x[:, -2:])

    Z_ohe = PCA(n_components=2).fit_transform(x)

    mask_tp = y_ord > 3
    mask_fn = ~mask_tp

    axes[2].scatter(Z_ohe[mask_tp, 0], Z_ohe[mask_tp, 1],
                    c="#1f77b4", s=20, alpha=0.7, label="TP")

    axes[2].scatter(Z_ohe[mask_fn, 0], Z_ohe[mask_fn, 1],
                    c="orange", s=20, alpha=0.7, label="FN")
    axes[2].set_title("PCA variable ordinale one-hot")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    return df_to_r