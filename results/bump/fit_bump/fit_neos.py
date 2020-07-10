import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.loadtxt("neos_bump_ratio.csv", delimiter=",")

x = data[:, 0]
# x was from 0 to 8, but it should have been 1 to 8
x = 1+7./8.*x
y = data[:, 1]

x_fit = x[np.logical_and(0<x,x<8.)]
y_fit = y[np.logical_and(0<x,x<8.)]

def f(x, a0, b0, mu, sig):
    return a0 + b0*np.exp(-(x-mu)**2/ (2*sig**2))
f = np.vectorize(f)

res = curve_fit(f, x, y, [1., 0.1, 4.5, 0.5])

plt.figure()
plt.plot(x, y, "k-", label="NEOS Ratio")
plt.plot(x_fit, f(x_fit, *res[0]), label="Gaussian Fit")
plt.title("Fit Res: a0=%.3f, b0=%.3f, mu=%.2f, sig=%.3f"%tuple(res[0]))
plt.xlabel("Prompt Energy (MeV)")
plt.ylabel("Ratio to Prediction")
plt.legend()
plt.savefig("neos_fit.png")
print("Fit results: %s"%res[0])
