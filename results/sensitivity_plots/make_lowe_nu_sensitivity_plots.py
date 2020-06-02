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

    fig1, (a00, a10) = plt.subplots(2, 1)
    fig1.patch.set_facecolor('white')
    fig1.set_figwidth(5.2)
    fig1.set_figheight(7.6)
    #plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.4, left=0.15, right=0.9, bottom=0.1, top=0.95)

    # Si
    x_thr      = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0063, 0.0126, 0.0250, 0.0500]
    y_thr      = [0.141,  0.141,  0.144,  0.151,  0.160,  0.177,  0.208,  0.287,  0.567,  0.946]
    spl_thr = make_interp_spline(np.log10(x_thr), np.log10(y_thr), k=3)
    x_thr_new = np.logspace(np.log10(x_thr[0]), np.log10(x_thr[-1]))
    y_thr_new = np.power(10., spl_thr(np.log10(x_thr_new)))
    a00.plot(1.e3*x_thr_new, y_thr_new, color="#984ea3", linestyle="--", label="Si (A=28.1)")
    a00.plot(1.e3*np.array(x_thr), y_thr, ".", color="#984ea3")

    # Zn
    x_thr      = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0063, 0.0126, 0.0250, 0.0500]
    y_thr      = [0.0380, 0.0387, 0.0395, 0.0436, 0.0477, 0.0599, 0.0876, 0.151,  0.619,  6.86]
    spl_thr = make_interp_spline(np.log10(x_thr), np.log10(y_thr), k=3)
    x_thr_new = np.logspace(np.log10(x_thr[0]), np.log10(x_thr[-1]))
    y_thr_new = np.power(10., spl_thr(np.log10(x_thr_new)))
    a00.plot(1.e3*x_thr_new, y_thr_new, color="#ff7f00", linestyle=":", label="Zn (A=65.4))")
    a00.plot(1.e3*np.array(x_thr), y_thr, ".", color="#ff7f00")

    # Ge
    x_thr      = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016, 0.0032, 0.0063, 0.0126, 0.0250, 0.0500]
    y_thr      = [0.0324, 0.0328, 0.0337, 0.0370, 0.0414, 0.0523, 0.0771, 0.152,  0.656,  16.5]
    spl_thr = make_interp_spline(np.log10(x_thr), np.log10(y_thr), k=3)
    x_thr_new = np.logspace(np.log10(x_thr[0]), np.log10(x_thr[-1]))
    y_thr_new = np.power(10., spl_thr(np.log10(x_thr_new)))
    a00.plot(1.e3*x_thr_new, y_thr_new, color="#a65628", linestyle="-.", label="Ge (A=72.6)")
    a00.plot(1.e3*np.array(x_thr), y_thr, ".", color="#a65628")

    a00.legend()
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

    '''a01.plot(x_a, y_a_worst, linestyle="--", color="#984ea3", label="Worst Case")
    a01.plot(x_a, y_a_worst, ".", color="#984ea3")
    a01.plot(x_a, y_a_med, linestyle=":", color="#ff7f00", label="Medium Case")
    a01.plot(x_a, y_a_med, ".", color="#ff7f00")
    a01.plot(x_a, y_a_best, linestyle="-.", color="#a65628", label="Best Case")
    a01.plot(x_a, y_a_best, ".", color="#a65628")
    #a01.fill_between(x_a, y_low_a, y_high_a,
    #                 facecolor='c', color='c', alpha=0.5)
    a01.set(xlabel="A", ylabel="Required Exposure (kg*years)")
    a01.legend()
    a01.set_yscale('log')'''

    #x_b       = [1.,    1.7,   2.8,   4.6,   7.7,   12.9,  21.5,  36.,   60.,   100.,  250.,  500., 1000.]
    #y_b_worst = [0.095, 0.103, 0.104, 0.107, 0.114, 0.149, 0.186, 0.275, 0.353, 0.471, 1.10,  1.95, 3.52]
    #y_b_med   = [0.095, 0.101, 0.103, 0.107, 0.112, 0.139, 0.157, 0.203, 0.248, 0.309, 0.601, 1.00, 2.10]
    #y_b_best  = [0.095, 0.100, 0.102, 0.105, 0.108, 0.132, 0.143, 0.165, 0.171, 0.179, 0.273, 0.57, 1.20]

    x_b       = [1.,    1.7,   2.8,   4.6,   7.7,   12.9,  21.5,  36.,   60.,   100.,  250.,  500., 1000.]
    y_b_worst = [0.095, 0.103, 0.104, 0.107, 0.114, 0.149, 0.186, 0.260, 0.353, 0.485, 1.10,  1.95, 3.52]
    y_b_med   = [0.095, 0.101, 0.103, 0.107, 0.112, 0.139, 0.160, 0.203, 0.248, 0.309, 0.601, 1.00, 2.05]
    y_b_best  = [0.095, 0.101, 0.102, 0.104, 0.107, 0.125, 0.143, 0.176, 0.200, 0.247, 0.370, 0.57, 1.1]

    #x_b       = [1.,    1.27,  1.61,  2.04,  2.59,  3.29,  4.18,  5.30,  6.72,  8.53,  10.8,  13.7,  17.4,  22.1,  28.1,  35.6,  45.2,  57.4,  72.8,  92.4,  117.,  149.,  189.,  240.,  304.,  386.,  489.,  621.,  788.,  1000.]
    #y_b_worst = [0.096, 0.096, 0.097, 0.098, 0.103, 0.105, 0.107, 0.111, 0.113, 0.117, 0.124, 0.132, 0.144, 0.161, 0.212, 0.240, 0.273, 0.305, 0.352, 0.450, 0.519, 0.600, 0.850, 0.881, 1.123, 1.524, 1.989, 2.402, 2.904, 3.500]
    #y_b_med   = [0.095, 0.096, 0.096, 0.097, 0.102, 0.104, 0.106, 0.107, 0.111, 0.113, 0.116, 0.121, 0.127, 0.131, 0.169, 0.182, 0.198, 0.214, 0.236, 0.299, 0.341, 0.381, 0.453, 0.449, 0.577, 0.757, 1.026, 1.293, 1.682, 2.079]
    #y_b_best  = [0.095, 0.095, 0.096, 0.096, 0.103, 0.102, 0.104, 0.104, 0.106, 0.108, 0.111, 0.113, 0.116, 0.122, 0.151, 0.157, 0.168, 0.180, 0.196, 0.239, 0.266, 0.298, 0.292, 0.256, 0.243, 0.423, 0.540, 0.724, 0.938, 1.253]

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
