import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from poly_regression import PolynomialRegression

df = pd.read_csv("podaci.csv")
X = df[["X"]]
y = df["Y"]
X_new = np.array([600, 900, 1300, 1600])
xx = np.linspace(500, 1600, 400)
lambdas = [0, 0.1, 10, 1000]

fig, axes = plt.subplots(1, len(lambdas), figsize=(16, 4), sharey=True)
for ax, lam in zip(axes, lambdas):
    print(f"===== lambda = {lam} =====")
    for M in [1, 2, 3, 5, 9]:
        m = PolynomialRegression(degree=M, alpha=lam).fit(X, y)
        print(f"M={M}  Err={m.error(X, y):.4g}  MSE={m.score(X, y):.4g}  pred={m.predict(X_new).round(2)}")
        ax.plot(xx, m.predict(xx), label=f"M={M}")
    ax.scatter(df.X, df.Y, c="k", zorder=5)
    ax.set_title(f"lambda = {lam}")
    ax.set_ylim(0, 8)
    ax.set_xlabel("X (mm)")
axes[0].set_ylabel("Y (t/ha)")
axes[0].legend()
plt.tight_layout()
plt.show()