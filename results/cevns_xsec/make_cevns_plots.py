from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, total_XSec_cns, total_XSec_cns_compound, total_XSec_cns_compound_in_bin, get_atomic_arrs

import numpy as np
from scipy.optimize import curve_fit
import scipy.integrate as spint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os

#plt.rcParams.update({'font.size': 18})

def plot_cevns_rate_fixed_T():
    fig2 = plt.figure()
    Tmin = 0.001
    Z = 32
    N = 74.
    e_arr = np.linspace(0.,1e7, 10000)
    fig2.patch.set_facecolor('white')
    plt.ylim((1e-44, 1e-42))
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(10.,e_arr,Z,N),'k:',label='T=10 eV',linewidth=2)
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(50.,e_arr,Z,N),'b-',label='T=50 eV',linewidth=2)
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(100.,e_arr,Z,N),'r--',label='T=100 eV',linewidth=2)
    plt.semilogy(e_arr*1e-6,dsigmadT_cns(200.,e_arr,Z,N),'g-.',label='T=200 eV',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Differential XSec, cm^2/eV')
    plt.title('Ge-74 Differential CEvNS XSec, Fixed T')
    plt.savefig('plots/diff_xsec_fixed_T.png')
    fig2.clf()


def plot_cevns_rate_fixed_Enu():
    fig2 = plt.figure()
    Tmin = 0.001
    Z = 32
    N = 74
    t_arr = np.logspace(0, 4, 10000)
    fig2.patch.set_facecolor('white')
    plt.ylim((1e-44, 1e-42))
    plt.loglog(t_arr,dsigmadT_cns(t_arr,1e6,Z,N),'k:',label='Enu = 1 MeV',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns(t_arr,2e6,Z,N),'b-',label='Enu = 2 MeV',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns(t_arr,4e6,Z,N),'r--',label='Enu = 4 MeV',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns(t_arr,6e6,Z,N),'g-.',label='Enu = 6 MeV',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Energy (eV)')
    plt.ylabel('Differential XSec, cm^2/eV')
    plt.title('Ge-74 Differential CEvNS XSec, Fixed Enu')
    plt.savefig('plots/diff_xsec_fixed_Enu.png')
    fig2.clf()

def plot_cevns_rate_vs_T_Enu(nu_spec=None, nbins=1000):
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    e_arr = np.linspace(0.,1e7, nbins)
    Tmin = 0.001
    t_arr = np.logspace(0, 3, nbins)

    Z = 32
    N = 74

    T, E = np.meshgrid(t_arr,e_arr)
    spec = dsigmadT_cns(T,E,Z,N)
    smax = spec.max()
    smin = smax*1e-3
    spec[spec<smin] = smin

    im = plt.pcolor(T, E*1e-6, spec,
                    norm=LogNorm(vmin=smin, vmax=smax),
                    cmap='PuBu_r')
    fig.colorbar(im)
    plt.xlabel("Recoil Energy T (eV)")
    plt.ylabel("Neutrino Energy Enu (MeV)")
    plt.title("Ge-74 Differential XSec, cm^2/eV")
    plt.savefig('plots/diff_xsec_vs_E_T.png')

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # CEvNS Differential Cross Section
    plot_cevns_rate_fixed_T()
    plot_cevns_rate_fixed_Enu()
    plot_cevns_rate_vs_T_Enu(nbins=10)

    labels=["Ge", "Zn", "Si",
            "CaWO4",
            "Al2O3"]

    thresholds = [0., 10., 100.]
    emin = 0.
    emax = 12.e6
    bin_size = 1.e4 # 10 keV
    elbs = np.arange(emin, emax, bin_size)

    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        f = open("cevns_xsec_binned_%s.txt"%labels[i], 'w')
        f.write("# "+labels[i]+"\n")
        f.write("# Z=%s, N=%s, atoms=%s\n"%(Z_arr, N_arr, atom_arr))
        f.write("# Columns 2-4 contain CEvNS xsec averaged over a %.2e keV bin\n"%(bin_size/1.e3))
        f.write("# xsec units: cm^2\n")
        f.write("#\n")
        f.write("# Enu (keV) Thr=0 eV     Thr=10 eV    Thr=100 eV\n")
        for elb in elbs:
            line = "%.4e, "%(elb/1.e3)
            for Tmin in thresholds:
                line += "%.5e, "%\
                    (total_XSec_cns_compound_in_bin(Tmin,
                                                    elb, elb+bin_size,
                                                    Z_arr, N_arr,
                                                    atom_arr)/bin_size)
            line = line[:-2]+"\n"
            f.write(line)
        f.close()
        
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        f = open("cevns_xsec_%s.txt"%labels[i], 'w')
        f.write("# "+labels[i]+"\n")
        f.write("# Z=%s, N=%s, atoms=%s\n"%(Z_arr, N_arr, atom_arr))
        f.write("# Columns 2-4 contain CEvNS xsec in cm^2\n")
        f.write("#\n")
        f.write("# Enu (keV) Thr=0 eV     Thr=10 eV    Thr=100 eV\n")
        for elb in elbs:
            line = "%.4e, "%(elb/1.e3)
            for Tmin in thresholds:
                line += "%.5e, "%\
                    total_XSec_cns_compound(Tmin, elb,
                                            Z_arr, N_arr,
                                            atom_arr,
                                            form_factor=True, helm_ff=True)
            line = line[:-2]+"\n"
            f.write(line)
        f.close()
        
