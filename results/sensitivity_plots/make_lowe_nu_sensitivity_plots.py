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

    # Make plots for Commercial Reactor, All Enu<1.8 MeV

    fig1, (a00, a01, a10) = plt.subplots(3, 1)
    fig1.patch.set_facecolor('white')
    fig1.set_figwidth(5.2)
    fig1.set_figheight(10.6)
    #plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.4)

    x_thr      = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0063, 0.0126, 0.0250, 0.0500]
    y_thr      = [0.0324, 0.0328, 0.0337, 0.0370, 0.0414, 0.0523, 0.0771, 0.152,  0.656,  16.5]

    spl_thr = make_interp_spline(np.log10(x_thr), np.log10(y_thr), k=3)
    x_thr_new = np.logspace(np.log10(x_thr[0]), np.log10(x_thr[-1]))
    y_thr_new = np.power(10., spl_thr(np.log10(x_thr_new)))

    a00.plot(1.e3*x_thr_new, y_thr_new, color="k", linestyle="-")
    a00.plot(1.e3*np.array(x_thr), y_thr, ".", color="k")
    #a00.plot(x_thr, y_thr, "r:")
    #a00.fill_between(x_thr, y_low_thr, y_high_thr,
    #                 facecolor='c', color='c', alpha=0.5)
    a00.axvline(1, color='k', linestyle=":")
    a00.axvline(10, color='k', linestyle=":")
    a00.axvline(50, color='k', linestyle=":")
    a00.set(xlabel="Threshold (eV)", ylabel="Required Exposure (kg*years)")
    a00.set_xscale('log')
    a00.set_yscale('log')

    x_a       = [28.1,  65.4,  72.6]
    y_a_worst = [14.,   14.,   25.]
    y_a_med   = [0.28,  0.14,  0.13]
    y_a_best  = [0.081, 0.038, 0.034]

    a01.plot(x_a, y_a_worst, linestyle="--", color="#984ea3", label="Worst Case")
    a01.plot(x_a, y_a_worst, ".", color="#984ea3")
    a01.plot(x_a, y_a_med, linestyle=":", color="#ff7f00", label="Medium Case")
    a01.plot(x_a, y_a_med, ".", color="#ff7f00")
    a01.plot(x_a, y_a_best, linestyle="-.", color="#a65628", label="Best Case")
    a01.plot(x_a, y_a_best, ".", color="#a65628")
    #a01.fill_between(x_a, y_low_a, y_high_a,
    #                 facecolor='c', color='c', alpha=0.5)
    a01.set(xlabel="A", ylabel="Required Exposure (kg*years)")
    a01.legend()
    a01.set_yscale('log')

    x_b       = [1.,    1.7,   2.8,   4.6,   7.7,   12.9,  21.5,  36.,   60.,   100.,  250.,  500., 1000.]
    y_b_worst = [0.095, 0.103, 0.104, 0.107, 0.114, 0.149, 0.186, 0.275, 0.353, 0.471, 1.10,  1.95, 3.52]
    y_b_med   = [0.095, 0.101, 0.103, 0.107, 0.112, 0.139, 0.157, 0.203, 0.248, 0.309, 0.601, 1.00, 2.10]
    y_b_best  = [0.095, 0.100, 0.102, 0.105, 0.108, 0.132, 0.143, 0.165, 0.171, 0.179, 0.273, 0.57, 1.20]

    spl_b_worst = make_interp_spline(np.log10(x_b), np.log10(y_b_worst), k=3)
    spl_b_med = make_interp_spline(np.log10(x_b), np.log10(y_b_med), k=3)
    spl_b_best = make_interp_spline(np.log10(x_b), np.log10(y_b_best), k=3)

    x_b_new = np.logspace(np.log10(x_b[0]), np.log10(x_b[-1]))
    y_b_worst_new = np.power(10., spl_b_worst(np.log10(x_b_new)))
    y_b_med_new = np.power(10., spl_b_med(np.log10(x_b_new)))
    y_b_best_new = np.power(10., spl_b_best(np.log10(x_b_new)))

    a10.plot(x_b_new, y_b_worst_new, color="#e41a1c", linestyle="--", label="Worst Shape")
    a10.plot(x_b, y_b_worst, ".", color="#e41a1c")
    a10.plot(x_b_new, y_b_med_new, color="#377eb8", linestyle=":", label="Medium Shape")
    a10.plot(x_b, y_b_med, ".", color="#377eb8")
    a10.plot(x_b_new, y_b_best_new, color="#4daf4a", linestyle="-.", label="Optimistic Shape")
    a10.plot(x_b, y_b_best, ".", color="#4daf4a")
    #a10.fill_between(x_b, y_low_b, y_high_b,
    #                 facecolor='c', color='c', alpha=0.5)
    a10.set(xlabel="Background Level (evts/kg/day)", ylabel="Required Exposure (kg*years)")
    a10.set_xscale('log')
    a10.set_yscale('log')
    a10.legend()

    plt.savefig('plots/lowe_nu_sensitivity_plots.png')
