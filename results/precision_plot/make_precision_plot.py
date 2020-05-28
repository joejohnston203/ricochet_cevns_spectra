import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # PAPER PLOT- Medium case, 5% signal uncertainty and 0% uncertainty
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    data_var = np.loadtxt("lowe_nu_commercial_allLowE_precision_varying/ge_bkgd_medium_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, 'r-', label="5% CEvNS Uncertainty")

    data_fix = np.loadtxt("lowe_nu_commercial_allLowE_precision_fixed/ge_bkgd_medium_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, 'b-.', label="No CEvNS Uncertainty")

    plt.axhline(y=1., color='k', linestyle=":")
    plt.axhline(y=5., color='k', linestyle=":")
    
    plt.legend(prop={'size':11})
    plt.gca().set_xlim(1.e-1, 100.)
    plt.gca().set_ylim(0., 20.)
    plt.xlabel("Exposure (kg*years)")
    plt.ylabel("Precision (%)")

    plt.savefig('plots/lowenu_comm_precision.png')


    # DEBUG PLOT: Varying low e envelope
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    data_var = np.loadtxt("lowe_nu_commercial_allLowE_precision_varying/ge_bkgd_optimistic_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=100, Optimistic")

    data_var = np.loadtxt("lowe_nu_commercial_allLowE_precision_varying/ge_bkgd_medium_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=10, Medium")

    data_var = np.loadtxt("lowe_nu_commercial_allLowE_precision_varying/ge_bkgd_optimistic_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=10, Optimistic")

    plt.axhline(y=1., color='k', linestyle=":")
    plt.axhline(y=5., color='k', linestyle=":")
    
    plt.legend(prop={'size':11})
    plt.gca().set_xlim(1.e-1, 100.)
    plt.gca().set_ylim(0., 20.)
    plt.xlabel("Exposure (kg*years)")
    plt.ylabel("Precision (%)")
    plt.title("Precision with 5% Uncertainty on Enu<1.8 MeV")

    plt.savefig('plots/lowenu_comm_precision_debug_5.png')


    # DEBUG PLOT: Fixed low e envelope
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    data_fix = np.loadtxt("lowe_nu_commercial_allLowE_precision_fixed/ge_bkgd_medium_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='-.', label="B=100, Medium")

    data_fix = np.loadtxt("lowe_nu_commercial_allLowE_precision_fixed/ge_bkgd_optimistic_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='-.', label="B=100, Optimistic")

    data_fix = np.loadtxt("lowe_nu_commercial_allLowE_precision_fixed/ge_bkgd_medium_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='-.', label="B=10, Medium")

    data_fix = np.loadtxt("lowe_nu_commercial_allLowE_precision_fixed/ge_bkgd_optimistic_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='-.', label="B=10, Optimistic")

    plt.axhline(y=1., color='k', linestyle=":")
    plt.axhline(y=5., color='k', linestyle=":")
    
    plt.legend(prop={'size':11})
    plt.gca().set_xlim(1.e-1, 100.)
    plt.gca().set_ylim(0., 20.)
    plt.xlabel("Exposure (kg*years)")
    plt.ylabel("Precision (%)")
    plt.title("Precision with No Uncertainty on Enu<1.8 MeV")

    plt.savefig('plots/lowenu_comm_precision_debug_0.png')

    
