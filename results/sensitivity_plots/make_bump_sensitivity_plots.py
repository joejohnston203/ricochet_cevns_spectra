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

    x_thr      = [1.,    10.,   50.]
    y_thr      = [2.5e7, 3.3e7, 8.5e7]
    y_low_thr  = [2.9e2, 9.6e3, 1.0e4]
    y_high_thr = [1.6e9, 2.0e9, 3.5e9]
    a00.plot(x_thr, y_thr)
    a00.fill_between(x_thr, y_low_thr, y_high_thr,
                     facecolor='c', color='c', alpha=0.5)
    a00.set(xlabel="Threshold (eV)", ylabel="Required Exposure (kg*years)")
    a00.set_xscale('log')
    a00.set_yscale('log')

    #x_a      = [28.1, 65.4, 72.6]
    x_a      = [72.6]
    y_a      = [1.0e5]
    y_low_a  = [4.9e3]
    y_high_a = [5.9e6]
    a01.plot(x_a, y_a)
    a01.fill_between(x_a, y_low_a, y_high_a,
                     facecolor='c', color='c', alpha=0.5)
    a01.set(xlabel="A", ylabel="Required Exposure (kg*years)")
    #a01.set_yscale('log')

    x_b      = [1.,    10.,   100.]
    y_b      = [1.0e5, 4.8e4, 3.2e5]
    y_low_b  = [4.9e3, 2.7e1, 6.3e1]
    y_high_b = [5.9e6, 1.3e7, 1.9e7]
    a10.plot(x_b, y_b)
    a10.fill_between(x_b, y_low_b, y_high_b,
                     facecolor='c', color='c', alpha=0.5)
    a10.set(xlabel="Background Level (evts/kg/day)", ylabel="Required Exposure (kg*years)")
    a10.set_xscale('log')
    a10.set_yscale('log')

    x_sh      = [1,    2,    3,  ]
    y_sh      = [2.7e7, 2.5e7, 1.0e5]
    y_low_sh  = [1.1e4, 2.9e2, 4.8e3]
    y_high_sh = [1.0e10, 2.6e9, 5.9e6]
    a11.plot(x_sh, y_sh)
    a11.fill_between(x_sh, y_low_sh, y_high_sh,
                     facecolor='c', color='c', alpha=0.5)
    a11.set(xlabel="Background Shape", ylabel="Required Exposure (kg*years)")
    a11.xaxis.set_major_locator(MaxNLocator(integer=True))
    a11.set_yscale('log')

    #plt.savefig('plots/bump_sensitivity_plots.png')
    plt.show()
