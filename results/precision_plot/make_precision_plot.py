import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline, BSpline

def comparison_plots(shape="medium", bkgd="1.0e+02", label="medium"):
    # PAPER PLOT- Medium case, 5% signal uncertainty and 0% uncertainty
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    data_var = np.loadtxt("lowe_nu_commercial_allLowE_precision_varying/ge_bkgd_"+shape+"_"+bkgd+"_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    spl_var = make_interp_spline(np.log10(x_var), y_var, k=3)
    x_var_new = np.logspace(np.log10(x_var[0]), np.log10(x_var[-1]))
    y_var_new = spl_var(np.log10(x_var_new))
    plt.semilogx(x_var_new, y_var_new, 'r-', label="5% Uncertainty, One Shape")

    data_fix = np.loadtxt("lowe_nu_commercial_allLowE_precision_fixed/ge_bkgd_"+shape+"_"+bkgd+"_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    spl_fix = make_interp_spline(np.log10(x_fix), y_fix, k=3)
    x_fix_new = np.logspace(np.log10(x_fix[0]), np.log10(x_fix[-1]))
    y_fix_new = spl_fix(np.log10(x_fix_new))
    plt.semilogx(x_fix_new, y_fix_new, 'b-.', label="No Uncertainty, One Shape")

    plt.axhline(y=1., color='k', linestyle=":")
    plt.axhline(y=5., color='k', linestyle=":")

    plt.legend(prop={'size':11})
    plt.gca().set_xlim(1.e-1, 100.)
    plt.gca().set_ylim(0., 20.)
    plt.xlabel("Exposure (kg*years)")
    plt.ylabel("Precision (%)")
    plt.title("Precision vs Exposure, %s Shape, Bkgd %s"%(shape, bkgd))

    plt.savefig('plots/lowenu_comm_precision_'+label+'.png')

    data_var = np.loadtxt("lowe_nu_commercial_allLowE_precision_summed_shapes_varying/ge_bkgd_"+shape+"_"+bkgd+"_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    spl_var = make_interp_spline(np.log10(x_var), y_var, k=3)
    x_var_new = np.logspace(np.log10(x_var[0]), np.log10(x_var[-1]))
    y_var_new = spl_var(np.log10(x_var_new))
    plt.semilogx(x_var_new, y_var_new, 'k--', label="5% Uncertainty, Summed Shapes")

    data_fix = np.loadtxt("lowe_nu_commercial_allLowE_precision_summed_shapes_fixed/ge_bkgd_"+shape+"_"+bkgd+"_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    spl_fix = make_interp_spline(np.log10(x_fix), y_fix, k=3)
    x_fix_new = np.logspace(np.log10(x_fix[0]), np.log10(x_fix[-1]))
    y_fix_new = spl_fix(np.log10(x_fix_new))
    plt.semilogx(x_fix_new, y_fix_new, 'c:', label="No Uncertainty, Summed Shapes")
    
    plt.legend(prop={'size':11})
    plt.savefig('plots/lowenu_comm_precision_summed_shapes_'+label+'.png')


def debug_plots(dir_prefix="lowe_nu_commercial_allLowE_precision_",
                label="one_shape"):
    # DEBUG PLOT: Varying low e envelope
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_conservative_1.0e+03_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='-', label="B=1000, Conservative")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_medium_1.0e+03_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='-', label="B=1000, Medium")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_optimistic_1.0e+03_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='-', label="B=1000, Optimistic")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_conservative_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=100, Conservative")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_medium_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=100, Medium")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_optimistic_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=100, Optimistic")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_conservative_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=10, Conservative")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_medium_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_var = data_var[:,0]*data_var[:,1]/365.
    # Precision in pct
    y_var = data_var[:,4]*100.
    plt.semilogx(x_var, y_var, linestyle='--', label="B=10, Medium")

    data_var = np.loadtxt(dir_prefix+"varying/ge_bkgd_optimistic_1.0e+01_thresh_1.0e-02/precision_ge.txt")
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

    plt.savefig('plots/lowenu_comm_precision_debug_5'+label+'.png')

    # DEBUG PLOT: Fixed low e envelope
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_conservative_1.0e+03_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='-', label="B=1000, Conservative")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_medium_1.0e+03_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='-', label="B=1000, Medium")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_optimistic_1.0e+03_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='-', label="B=1000, Optimistic")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_conservative_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='--', label="B=100, Conservative")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_medium_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='--', label="B=100, Medium")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_optimistic_1.0e+02_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle='--', label="B=100, Optimistic")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_conservative_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle=':', label="B=10, Conservative")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_medium_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle=':', label="B=10, Medium")

    data_fix = np.loadtxt(dir_prefix+"fixed/ge_bkgd_optimistic_1.0e+01_thresh_1.0e-02/precision_ge.txt")
    # Exposure in kg*years
    x_fix = data_fix[:,0]*data_fix[:,1]/365.
    # Precision in pct
    y_fix = data_fix[:,4]*100.
    plt.semilogx(x_fix, y_fix, linestyle=':', label="B=10, Optimistic")

    plt.axhline(y=1., color='k', linestyle=":")
    plt.axhline(y=5., color='k', linestyle=":")
    
    plt.legend(prop={'size':11})
    plt.gca().set_xlim(1.e-1, 100.)
    plt.gca().set_ylim(0., 20.)
    plt.xlabel("Exposure (kg*years)")
    plt.ylabel("Precision (%)")
    plt.title("Precision with No Uncertainty on Enu<1.8 MeV")

    plt.savefig('plots/lowenu_comm_precision_debug_0'+label+'.png')

    
if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    comparison_plots("optimistic", "1.0e+01", "optimistic")
    comparison_plots("medium", "1.0e+02", "medium")
    comparison_plots()
    debug_plots()
    debug_plots("lowe_nu_commercial_allLowE_precision_summed_shapes_",
                "summed_shapes")
