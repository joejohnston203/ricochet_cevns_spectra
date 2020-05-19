import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # Make plots for Commercial Reactor, U-238 n Capture

    fig1, ((a00, a01), (a10, a11)) = plt.subplots(2, 2)
    fig1.patch.set_facecolor('white')
    fig1.set_figwidth(9.6)
    fig1.set_figheight(7.2)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.4)

    x_thr      = [1.,   10.,  50.]
    y_thr      = [5.6,  61.8, 1.0e9]
    y_low_thr  = [3.7,  25.0, 1.1e7]
    y_high_thr = [18.2, 62.0, 1.0e14]
    a00.plot(x_thr, y_thr)
    a00.fill_between(x_thr, y_low_thr, y_high_thr,
                     facecolor='c', color='c', alpha=0.5)
    a00.set(xlabel="Threshold (eV)", ylabel="Required Exposure (kg*years)")
    a00.set_xscale('log')
    a00.set_yscale('log')

    x_a      = [28.1, 65.4, 72.6]
    y_a      = [30.1, 27.4, 56.6]
    y_low_a  = [12.8, 11.9, 25.0]
    y_high_a = [53.2, 50.3, 61.8]
    a01.plot(x_a, y_a)
    a01.fill_between(x_a, y_low_a, y_high_a,
                     facecolor='c', color='c', alpha=0.5)
    a01.set(xlabel="A", ylabel="Required Exposure (kg*years)")
    #a01.set_yscale('log')

    x_b      = [1.,   10.,  100.]
    y_b      = [47.9, 61.8, 122.]
    y_low_b  = [19.3, 25.0, 55.7]
    y_high_b = [78.5, 62.0, 328.]
    a10.plot(x_b, y_b)
    a10.fill_between(x_b, y_low_b, y_high_b,
                     facecolor='c', color='c', alpha=0.5)
    a10.set(xlabel="Background Level (evts/kg/day)", ylabel="Required Exposure (kg*years)")
    a10.set_xscale('log')
    #a10.set_yscale('log')

    x_sh      = [1,    2,    3,  ]
    y_sh      = [50.0, 55.0, 50.0]
    y_low_sh  = [22.6, 25.0, 21.8]
    y_high_sh = [55.5, 61.8, 54.4]
    a11.plot(x_sh, y_sh)
    a11.fill_between(x_sh, y_low_sh, y_high_sh,
                     facecolor='c', color='c', alpha=0.5)
    a11.set(xlabel="Background Shape", ylabel="Required Exposure (kg*years)")
    a11.xaxis.set_major_locator(MaxNLocator(integer=True))
    #11.set_yscale('log')

    plt.savefig('plots/lowe_nu_sensitivity_plots.png')
