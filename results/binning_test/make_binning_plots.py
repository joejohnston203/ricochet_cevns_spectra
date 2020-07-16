import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    bins =           np.array([1,    10,    20,    40,   80,   160])
    bump_res_ge =    np.array([1.e3, 164.2, 46.6,  37.8, 36.1, 35.1])
    bump_res_cawo4 = np.array([1.e3, 298.1, 271.7, 28.8, 17.9, 15.8])

    plt.figure()
    plt.semilogx(bins, bump_res_ge, 'k-', label="Ge")
    plt.semilogx(bins, bump_res_cawo4, 'r--', label=r'CaWO$_4$')
    plt.xlabel("Number of Energy Bins")
    plt.ylabel(r"Required Exposure for 5$\sigma$ Significance (kg*years)")
    plt.title("Bump Results (Best Case) vs Binning")
    plt.ylim(0., 400)
    plt.legend()
    plt.savefig("plots/bump_results_vs_bins.png")

    plt.figure()
    bins =          np.array([1,    3,   6,   10,  20,  40,  80,  160])
    lowenu_res_ge = np.array([1.e2, 20., 1.5, 0.6, 0.3, 0.2, 0.2, 0.2])
    plt.semilogx(bins, lowenu_res_ge, 'k-', label="Ge")
    plt.xlabel("Number of Energy Bins")
    plt.ylabel(r"Required Exposure for 5$\sigma$ Significance (kg*years)")
    plt.ylim(0., 10.)
    plt.title("Low E Nu Results (Best Case) vs Binning")
    plt.legend()
    plt.savefig("plots/lowenu_results_vs_bins.png")
