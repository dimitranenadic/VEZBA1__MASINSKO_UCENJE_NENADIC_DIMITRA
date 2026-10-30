import numpy as np, pandas as pd, matplotlib.pyplot as plt
from poly_regression import PolynomialRegression

df = pd.read_csv("podaci.csv"); X = df[["X"]]; y = df["Y"]
X_new = np.array([600, 900, 1300, 1600])
xx = np.linspace(500, 1600, 400)

for M in [1, 2, 3, 5, 9]:
    m = PolynomialRegression(degree=M).fit(X, y)
    print(f"M={M}  w={m.coef_}")
    print(f"     Err={m.error(X, y):.4g}  MSE={m.score(X, y):.4g}  pred={m.predict(X_new).round(2)}")
    plt.plot(xx, m.predict(xx), label=f"M={M}")

plt.scatter(df.X, df.Y, c="k", zorder=5)
plt.ylim(0, 8); plt.xlabel("X (mm)"); plt.ylabel("Y (t/ha)"); plt.legend(); plt.show()