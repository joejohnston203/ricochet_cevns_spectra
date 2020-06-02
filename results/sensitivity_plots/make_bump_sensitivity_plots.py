import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from scipy.interpolate import make_interp_spline, BSpline

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # Make plots for Commercial Reactor,5 MeV Bump

    fig1, (a00, a01, a10) = plt.subplots(3, 1)
    fig1.patch.set_facecolor('white')
    fig1.set_figwidth(5.2)
    fig1.set_figheight(10.6)
    #plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.4, left=0.15, right=0.9, bottom=0.075, top=0.95)

    #x_thr      = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0063, 0.0126, 0.0250, 0.0500]
    #y_thr      = [4.07e5, 4.62e5, 5.09e5, 3.79e5, 8.80e5, 2.57e5, 3.80e5, 4.13e5, 1.75e5, 7.80e5]
    x_thr     = [0.0001, 0.00015, 0.0002, 0.0003, 0.0004, 0.0006, 0.0008, 0.0012, 0.0016, 0.0024, 0.0032, 0.0048, 0.0063, 0.01,   0.0126, 0.02,   0.025,  0.04,   0.050]
    y_thr     = [3.80e5, 7.04e5,  3.02e5, 4.49e5, 4.33e5, 6.02e5, 3.67e5, 5.80e5, 3.89e5, 3.32e5, 4.29e5, 4.65e5, 1.98e5, 4.50e5, 4.32e5, 3.56e5, 2.24e5, 1.33e5, 7.86e4]

    spl_thr   = make_interp_spline(np.log10(x_thr), np.log10(y_thr), k=1)
    x_thr_new = np.logspace(np.log10(x_thr[0]), np.log10(x_thr[-1]))
    y_thr_new = np.power(10., spl_thr(np.log10(x_thr_new)))

    y_avg = 0
    for y in y_thr:
        y_avg += y
    y_avg = y_avg/float(len(y_thr))

    #a00.plot(1.e3*x_thr_new, y_thr_new, color="k", linestyle="-")
    a00.plot(1.e3*x_thr_new, 0.*x_thr_new+y_avg, color="k", linestyle="-")
    a00.plot(1.e3*np.array(x_thr), y_thr, ".", color="k")
    #a00.plot(x_thr, y_thr, "r:")
    #a00.fill_between(x_thr, y_low_thr, y_high_thr,
    #                 facecolor='c', color='c', alpha=0.5)
    a00.axvline(1, color='k', linestyle=":")
    a00.axvline(10, color='k', linestyle=":")
    a00.axvline(50, color='k', linestyle=":")
    a00.set(xlabel="Threshold (eV)", ylabel="Required Exposure (kg*years)",
            title="5 MeV Bump")
    a00.set_xscale('log')
    a00.set_yscale('log')
    a00.set_ylim(5.e4, 5.e6)

    x_a       = [28.1,  65.4,  72.6]
    y_a_worst = [1.1e5, 2.7e5, 6.9e5]
    #y_a_med   = [0.28,  0.14,  0.13]
    y_a_best  = [1.0e6, 3.0e6, 2.8e6]

    a01.plot(x_a, y_a_worst, linestyle="--", color="#984ea3", label="Worst Case")
    a01.plot(x_a, y_a_worst, ".", color="#984ea3")
    #a01.plot(x_a, y_a_med, linestyle=":", color="#ff7f00", label="Medium Case")
    #a01.plot(x_a, y_a_med, ".", color="#ff7f00")
    a01.plot(x_a, y_a_best, linestyle="-.", color="#a65628", label="Best Case")
    a01.plot(x_a, y_a_best, ".", color="#a65628")
    #a01.fill_between(x_a, y_low_a, y_high_a,
    #                 facecolor='c', color='c', alpha=0.5)
    a01.set(xlabel="A", ylabel="Required Exposure (kg*years)")
    a01.legend()
    a01.set_yscale('log')

    #x_b       = [1.,     1.7,    2.8,    4.6,    7.7,    12.9,   21.5,   36.,    60.,    100.]
    #y_b_worst = []
    #y_b_med   = []
    #y_b_ge  = [4.17e5, 5.57e4, 1.72e4, 5.71e4, 1.01e6, 5.07e5, 6.67e5, 8.04e5, 7.04e5, 4.90e5]

    x_b      = [1.,    1.17,  1.37,  1.61,  1.89,  2.21,  2.59,  3.04,  3.56,  4.18,  4.89,  5.74,  6.72,  7.88,  9.24,  10.8,  12.7,  14.9,  17.4,  20.4,  24.0,  28.1,  32.9,  38.6,  45.2,  53.0,  62.1,  72.8,  85.3,  100.]
    y_b_ge   = [4.5e4, 3.8e5, 1.8e5, 1.3e5, 1.1e5, 3.9e4, 2.9e4, 1.5e4, 9.2e3, 3.7e4, 6.3e4, 2.1e5, 1.2e5, 4.2e5, 9.1e5, 6.1e5, 5.6e5, 5.7e5, 5.7e5, 6.0e5, 6.3e5, 7.6e5, 6.8e5, 6.5e5, 7.7e5, 6.8e5, 6.2e5, 5.8e5, 5.3e5, 5.0e5]
    y_b_zn   = [1.3e5, 5.5e5, 9.1e4, 6.4e5, 4.4e4, 2.0e4, 1.9e4, 1.2e4, 6.3e4, 4.4e4, 9.1e4, 2.3e5, 2.6e5, 6.7e5, 9.5e5, 5.5e5, 6.4e5, 6.2e5, 6.6e5, 6.8e5, 7.7e5, 8.2e5, 8.5e5, 9.1e5, 9.1e5, 9.4e5, 7.6e5, 6.9e5, 4.4e5, 1.4e6]

    #spl_b_worst = make_interp_spline(np.log10(x_b), np.log10(y_b_worst), k=3)
    spl_b_zn = make_interp_spline(np.log10(x_b), np.log10(y_b_zn), k=3)
    spl_b_ge = make_interp_spline(np.log10(x_b), np.log10(y_b_ge), k=3)

    x_b_new = np.logspace(np.log10(x_b[0]), np.log10(x_b[-1]))
    #y_b_worst_new = np.power(10., spl_b_worst(np.log10(x_b_new)))
    y_b_zn_new = np.power(10., spl_b_zn(np.log10(x_b_new)))
    y_b_ge_new = np.power(10., spl_b_ge(np.log10(x_b_new)))

    #a10.plot(x_b_new, y_b_worst_new, color="#e41a1c", linestyle="--", label="Worst Shape")
    #a10.plot(x_b, y_b_worst, ".", color="#e41a1c")
    a10.plot(x_b_new[7:], y_b_zn_new[7:], color="#377eb8", linestyle=":", label="Zn")
    a10.plot(x_b[7:], y_b_zn[7:], ".", color="#377eb8")
    a10.plot(x_b_new[7:], y_b_ge_new[7:], color="#4daf4a", linestyle="-.", label="Ge")
    a10.plot(x_b[7:], y_b_ge[7:], ".", color="#4daf4a")
    #a10.fill_between(x_b, y_low_b, y_high_b,
    #                 facecolor='c', color='c', alpha=0.5)
    a10.set(xlabel="Background Level (evts/kg/day)", ylabel="Required Exposure (kg*years)")
    a10.set_xscale('log')
    a10.set_yscale('log')
    a10.legend()

    plt.savefig('plots/bump_sensitivity_plots.png')
