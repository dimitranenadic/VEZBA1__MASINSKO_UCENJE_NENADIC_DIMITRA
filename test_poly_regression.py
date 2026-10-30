import pandas as pd
from poly_regression import PolynomialRegression

df = pd.read_csv("podaci.csv")     # kolone X, Y
X = df[["X"]]
y = df["Y"]

model = PolynomialRegression(degree=2)
model.fit(X, y)
print("w   =", model.coef_)
print("y^  =", model.predict(X))
print("MSE =", model.score(X, y))