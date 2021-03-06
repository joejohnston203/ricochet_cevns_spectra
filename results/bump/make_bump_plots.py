from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, cns_total_rate_integrated, cns_total_rate_integrated_compound, total_XSec_cns, total_XSec_ibd, total_XSec_ibd_0th, ibd_yield, cevns_yield_compound

import numpy as np
from scipy.optimize import curve_fit, fmin, fsolve
import scipy.integrate as spint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os

plt.rcParams.update({'font.size': 16})

Mn = cevns_spectra.Mn
Mn_eV = Mn*1e3

s_per_day = 60.0*60.0*24.0

def compare_ibd_cevns(nu_spec):
    figure = plt.figure()
    e_arr = np.linspace(0., 55e6, 1000)

    # CEvNS Cross Sections
    xsec_Cs = total_XSec_cns(0., e_arr, 55., 132.9-55.)
    plt.semilogy(e_arr*1e-6,xsec_Cs*1.e38, color='blue', linestyle="--", label=r'CEvNS Cs')

    xsec_I = total_XSec_cns(0., e_arr, 53., 126.9-53.)
    plt.semilogy(e_arr*1e-6,xsec_I*1.e38, color='blue', linestyle=":", label=r'CEvNS I')

    xsec_Ge = total_XSec_cns(0., e_arr, 32., 72.64-32.)
    plt.semilogy(e_arr*1e-6,xsec_Ge*1.e38, color='green', linestyle="-.", label=r'CEvNS Ge')

    # IBD Cross Section
    xsec_ibd = total_XSec_ibd(e_arr)
    plt.semilogy(e_arr*1e-6,xsec_ibd*1.e38, color='red', linestyle="-", label=r'IBD')

    plt.xlabel("Energy (MeV)")
    plt.ylabel(r'Cross Section ($10^{-38}$ cm$^2$)')
    plt.legend()
    plt.grid()

    plt.xlim(0., 10.)
    plt.ylim(1.e-6, 1.e0)
    plt.savefig("plots/ibd_cevns_xsec.png")
    plt.xlim(5., 55.)
    plt.ylim(4.e-4, 3.e1)
    plt.savefig("plots/ibd_coherent_plot.png")

def plot_ibd(nu_spec):
    figure = plt.figure()
    e_arr = np.linspace(0., 55e6, 1000)

    # IBD Cross Section
    xsec_ibd_0th = total_XSec_ibd_0th(e_arr)
    plt.plot(e_arr*1e-6,xsec_ibd_0th*1.e42, color='red', linestyle="--", label=r'0$^{th}$ Order')

    xsec_ibd = total_XSec_ibd(e_arr)
    plt.plot(e_arr*1e-6,xsec_ibd*1.e42, color='blue', linestyle=":", label=r'1$^{st}$ Order')

    plt.xlabel("Energy (MeV)")
    plt.ylabel(r'Cross Section ($10^{-42}$ cm$^2$)')
    plt.title("IBD Cross Section")
    plt.legend()
    plt.grid()

    plt.xlim(0., 10.)
    plt.ylim(0., 10.)
    plt.savefig("plots/ibd_xsec.png")

    def f(x, a, b):
        if x<b:
            return 0.
        else:
            return a*(x-b)**2
    f = np.vectorize(f)
    res = curve_fit(f, e_arr, xsec_ibd_0th, [1.e-42, 1.e6])
    print("a*(x-b)^2 Fit Parameters: %s"%res[0])
    #plt.plot(e_arr*1.e-6, f(e_arr, *res[0])*1.e42)
    #plt.show()

def fit_daya_bay_data():
    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    plt.subplots_adjust(bottom=0.075, top=0.95)
    #sharex=True)
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(8.5)

    # Plot data from Daya Bay paper arXiv:1607.05378v2
    db_data = np.loadtxt("data/daya_bay_unfolded_spectrum.csv", delimiter=",")

    bin_low = db_data[:,0]
    bin_high = db_data[:,1]
    xvals = (bin_low+bin_high)/2.
    bin_widths = bin_high-bin_low

    yvals = db_data[:,2]
    yerrs = np.sqrt(db_data[:,3])

    a0.errorbar(xvals, yvals, yerr=yerrs, xerr=bin_widths/2.,
                 color='k', linestyle='None',
                 label="Daya Bay Data")

    # Huber/Mueller Prediction for Daya Bay
        # Mueller spectra
    nu_spec_mueller = NeutrinoSpectrum(nu_spec.distance, nu_spec.power, True,
                                       0.561, 0.076, 0.307, 0.056) # Daya Bay Numbers (arXiv:1607.05378v2)
    nu_spec_mueller.initialize_d_r_d_enu("u235", "txt",
                                         "../../data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("u238", "mueller")
    nu_spec_mueller.initialize_d_r_d_enu("pu239", "txt",
                                         "../../data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("pu241", "txt",
                                         "../../data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("other", "mueller")

    pred_spectrum = []
    for i in range(len(bin_low)):
        res = spint.quad(lambda e_eV:
                         nu_spec_mueller.d_phi_d_enu_ev(e_eV)*\
                         total_XSec_ibd(e_eV),
                         bin_low[i]*1.e6, bin_high[i]*1.e6)[0]
        res /= (bin_high[i]-bin_low[i])
        res *= 1.e46/nu_spec.nuFlux() # Convert to cm^2/fission/MeV x 10^-46
        pred_spectrum.append(res)
    a0.errorbar(xvals, pred_spectrum, xerr=bin_widths/2.,
                 color='r', linestyle='None',
                 label="Huber/Mueller Prediction")

    # Plot Settings
    a0.legend()
    a0.set_xlim(1.8, 8.4)
    a0.set(ylabel=r'$\phi_{\bar{\nu}_e}*\sigma_{IBD}$ (cm^2/fission/MeV $\times$ 10^-46)')

    a1.errorbar(xvals, yvals/pred_spectrum,
                xerr=bin_widths/2., yerr=yerrs/yvals,
                color='k', linestyle='None')
    a1.plot(xvals, xvals*0.+1., color='k')
    a1.set_xlim(1.8, 8.4)
    a1.set_ylim(0.6, 1.25)
    a1.set(xlabel="Antineutrino Energy (MeV)",
           ylabel="Ratio")
    plt.savefig("plots/fit_daya_bay_plot_reproduction.png")

    # Fit the spectrum
    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    plt.subplots_adjust(bottom=0.075, top=0.95)
    #sharex=True)
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(8.5)

    def get_bin(x):
        if(x<bin_low[0]):
            return -1
        for i in range(0, len(bin_low)):
            if(bin_low[i]<=x and
               x<bin_high[i]):
                return i
        return len(bin_low)

    def f(x, k0, b0, mu, sig):
        i = get_bin(x)
        if(i<0 or i>=len(bin_low)):
            return 0.
        else:
            return k0*pred_spectrum[get_bin(x)]+b0*np.exp(-(x-mu)**2/ (2*sig**2))*total_XSec_ibd(x*1.e6)
    f = np.vectorize(f)
    x_fit = xvals[np.logical_and(3.<xvals, xvals<7.5)]
    y_fit = yvals[np.logical_and(3.<xvals, xvals<7.5)]
    res = curve_fit(f, x_fit, y_fit, [1., 1.e44, 5.7, 0.5])
    print("Unfolded Spectrum Fit Parameters: %s"%res[0])
    a0.errorbar(xvals, yvals, yerr=yerrs, xerr=bin_widths/2.,
                 color='k', linestyle='None',
                 label="Daya Bay Data")
    a0.errorbar(xvals, res[0][0]*np.array(pred_spectrum),
                xerr=bin_widths/2.,
                color='red', linestyle='None',
                label="Huber/Mueller Rescaled")
    a0.errorbar(xvals, f(xvals, *res[0]),
                xerr=bin_widths/2.,
                color='blue', linestyle='None',
                label=r'Huber/Mueller+Gaussian*$\sigma_{IBD}$ Fit')
    a0.legend()
    a0.set_xlim(3., 7.5)
    a0.set(ylabel=r'$\phi_{\bar{\nu}_e}*\sigma_{IBD}$ (cm^2/fission/MeV $\times$ 10^-46)',
           title=r'Reactor Spectrum Gaussian Fit: $\mu$=%.2f, $\sigma=$%.3f'%(res[0][2], res[0][3]))

    def g(x, a0, b0, mu, sig):
        i = get_bin(x)
        if(i<0 or i>=len(bin_low)):
           return 1.
        else:
           return a0 + b0*np.exp(-(x-mu)**2/ (2*sig**2))
    g = np.vectorize(g)
    x_fit_ratio = xvals[np.logical_and(3.<xvals, xvals<7.5)]
    y_fit_ratio = np.array(yvals)[np.logical_and(3.<xvals, xvals<7.5)]/np.array(pred_spectrum)[np.logical_and(3.<xvals, xvals<7.5)]
    res = curve_fit(g, x_fit_ratio, y_fit_ratio, [0.9, 0.1, 5.7, 0.5])
    print("Ratio Fit Parameters: %s"%res[0])
    a1.errorbar(xvals, yvals/pred_spectrum,
                xerr=bin_widths/2., yerr=yerrs/yvals,
                color='k', linestyle='None')
    a1.plot(xvals, xvals*0.+1., color='k')
    a1.plot(xvals, xvals*0.+res[0][0], 'r--')
    a1.errorbar(xvals, g(xvals, *res[0]),
                xerr=bin_widths/2.,
                color='blue', linestyle='None')
    a1.set_xlim(3., 7.5)
    a1.set_ylim(0.6, 1.25)
    a1.set(xlabel="Antineutrino Energy (MeV)",
           ylabel="Ratio",
           title=r'Ratio Gaussian Fit: $\mu$=%.2f, $\sigma=$%.3f'%(res[0][2], res[0][3]))
    plt.savefig("plots/fit_daya_bay_gaussian_fits.png")

def max_bump_ratio(nu_spec, bump_frac):
    print("Total flux: %.2e"%nu_spec.nuFlux())
    nu_spec.bump_frac = bump_frac
    print("Total yield: %.2e"%ibd_yield(nu_spec))
    def ratio(e):
        nu_spec.bump_frac = 0.0
        rate0 = nu_spec.d_phi_d_enu_ev(e)*total_XSec_ibd(e)
        nu_spec.bump_frac = bump_frac
        rate1 = nu_spec.d_phi_d_enu_ev(e)*total_XSec_ibd(e)
        return rate1/rate0
    res = fmin(lambda x: -ratio(x), 6.e6)
    print("bump_frac: %.5f"%bump_frac)
    print("Max ratio occurs at: %.6e"%res)
    print("Max ratio: %.4f"%ratio(res))
    print("")

def plot_nu_bump_ibd(nu_spec,
                     bump_fracs=[0.01, 0.05]):
    old_frac = nu_spec.bump_frac

    # Plot unzoomed neutrino flux and ratio
    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    plt.subplots_adjust(bottom=0.075, top=0.95)
    #sharex=True)
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(8.5)
    lines_arr = ['b--', 'r-.', 'y:',
                 'c-', 'g--', 'm-.']
    e_arr = np.linspace(0., 1e7, 100000)

    nu_spec.bump_frac = 0.
    spec_tot = nu_spec.d_phi_d_enu_ev(e_arr)*total_XSec_ibd(e_arr)/nu_spec.nuFlux()
    a0.plot(e_arr*1e-6,spec_tot*1e6,'k-',label="No Distortion",linewidth=2)
    ibd_yield_0 = ibd_yield(nu_spec)
    bump_specs = []
    ibd_yields = []
    for i in range(len(bump_fracs)):
        nu_spec.bump_frac = bump_fracs[i]
        spec = nu_spec.d_phi_d_enu_ev(e_arr)*total_XSec_ibd(e_arr)/nu_spec.nuFlux()
        bump_specs.append(spec)
        a0.plot(e_arr*1e-6,spec*1e6,lines_arr[i],
                 label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
        ibd_yields.append(ibd_yield(nu_spec))

    #a0.set_ylim(0., 2e17)
    a0.set_xlim(0., 10.)
    a0.set(ylabel='Reactor Spectrum * IBD xsec, cm^2/fission/MeV')
    a0.set_title('Reactor Spectrum * IBD xsec at Commercial Reactor')

    a1.plot(e_arr*1e-6,spec_tot/spec_tot,'k-',label="No Distortion",linewidth=2)
    for i in range(len(bump_fracs)):
        spec = bump_specs[i]
        plt.plot(e_arr*1e-6,spec/spec_tot,lines_arr[i],
                 label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
        print("Bump=%.2e, Ratio=%s"%(bump_fracs[i], spec/spec_tot))
        print("tot int: %s"%spint.simps(spec_tot*1.e6, e_arr*1e-6))
        print("bump int: %s"%spint.simps(spec*1.e6, e_arr*1e-6))
        print("ibd yield no bump: %.3e cm^2/fission"%ibd_yield_0)
        print("ibd yield bump: %.3e cm^2/fission"%ibd_yields[i])
    a1.legend(loc=2, prop={'size':11})
    a1.set_xlim(0., 10.)
    a1.set_ylim(0.75, 1.25)
    a1.set(xlabel='Neutrino Energy (MeV)', ylabel='Ratio')

    '''axins = inset_axes(a0, width=3.5, height=2.5, loc=3,
                       bbox_to_anchor=(0.24, 0.3),
                       bbox_transform=a0.transAxes)
    axins.plot(e_arr*1e-6,spec_tot*1e6,'k-',label="No Distortion",linewidth=2)
    for i in range(len(bump_fracs)):
        spec = bump_specs[i]
        axins.plot(e_arr*1e-6,spec*1e6,lines_arr[i],
                   label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    axins.set_xlim(4.5, 7.5)
    #axins.set_ylim(0., 4.e15)'''

    plt.savefig('plots/reactor_bump_ibd_product.png')
    nu_spec.bump_frac = old_frac

def plot_nu_bump(nu_spec,
                 bump_fracs=[0.01, 0.05]):
    old_frac = nu_spec.bump_frac

    # Plot unzoomed neutrino flux and ratio
    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    plt.subplots_adjust(bottom=0.075, top=0.95)
    #sharex=True)
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(8.5)
    lines_arr = ['b--', 'r-.', 'y:',
                 'c-', 'g--', 'm-.']
    e_arr = np.linspace(0., 1e7, 100000)

    nu_spec.bump_frac = 0.
    spec_tot = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    a0.plot(e_arr*1e-6,spec_tot*1e6,'k-',label="No Distortion",linewidth=2)
    bump_specs = []
    for i in range(len(bump_fracs)):
        nu_spec.bump_frac = bump_fracs[i]
        spec = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
        bump_specs.append(spec)
        a0.plot(e_arr*1e-6,spec*1e6,lines_arr[i],
                 label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2,
                 color='red')

    a0.set_ylim(0., 2e17)
    a0.set_xlim(0., 10.)
    a0.set(ylabel='Flux, nu/(MeV*day*cm^2)')
    #a0.set_title('Neutrino Flux at Commercial Reactor')

    a1.plot(e_arr*1e-6,spec_tot/spec_tot,'k-',label="No Distortion",linewidth=2)
    for i in range(len(bump_fracs)):
        spec = bump_specs[i]
        plt.plot(e_arr*1e-6,spec/spec_tot,lines_arr[i],
                 label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2,
                 color='red')
        print("Bump=%.2e, Ratio=%s"%(bump_fracs[i], spec/spec_tot))
        print("tot int: %s"%spint.simps(spec_tot*1.e6, e_arr*1e-6))
        print("bump int: %s"%spint.simps(spec*1.e6, e_arr*1e-6))
    #a1.legend(loc=2, prop={'size':11})
    a1.set_xlim(0., 10.)
    a1.set_ylim(0.75, 1.25)
    a1.set(xlabel='Neutrino Energy (MeV)', ylabel='Ratio')

    axins = inset_axes(a0, width=3.5, height=2.5, loc=3,
                       bbox_to_anchor=(0.24, 0.3),
                       bbox_transform=a0.transAxes)
    axins.plot(e_arr*1e-6,spec_tot*1e6,'k-',label="No Distortion",linewidth=2)
    for i in range(len(bump_fracs)):
        spec = bump_specs[i]
        axins.plot(e_arr*1e-6,spec*1e6,lines_arr[i],
                   label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2,
                 color='red')
    axins.set_xlim(4.5, 7.5)
    axins.set_ylim(0., 4.e15)
    axins.set(xlabel=r'E$_\nu$ (MeV)')

    plt.savefig('plots/reactor_bump_neutrino_spectrum.pdf', bbox_inches='tight')
    nu_spec.bump_frac = old_frac

def plot_cevns_bump(nu_spec, bump_fracs, cns_bounds):
    old_frac = nu_spec.bump_frac

    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    plt.subplots_adjust(bottom=0.075, top=0.95)
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(8.5)
    lines_arr = ['b--', 'r-.', 'y:',
                 'c-', 'g--', 'm-.']

    #t_arr = np.linspace(0, 10000, num=10000)
    #t_arr = np.linspace(0, 10000, num=100)
    t_arr = np.logspace(0, 4, num=100)

    nu_spec.bump_frac = 0.
    spec_0 = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
    a0.plot(t_arr*1.e-3,spec_0*1.e3,'k-',label='No Distortion',linewidth=2)
    bump_specs = []
    for i in range(len(bump_fracs)):
        nu_spec.bump_frac = bump_fracs[i]
        spec_bump = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
        bump_specs.append(spec_bump)
        a0.plot(t_arr*1.e-3, spec_bump*1.e3,
                lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    a0.set_xlim(1.e-3, 3.)
    a0.set_xscale("log")
    a0.set_ylim(cns_bounds)
    a0.set_yscale("log")
    a0.set(ylabel='Ge Differential Event Rate (dru)')
    a0.set_title("Ge Differential Rate at Commercial Reactor")
    a0.axvline(x=1.e-3, color='k', linestyle=":")
    a0.axvline(x=10.e-3, color='k', linestyle=":")
    a0.axvline(x=50.e-3, color='k', linestyle=":")

    a1.plot(t_arr*1.e-3, spec_0/spec_0, 'k-', label='No Distortion', linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        a1.plot(t_arr*1.e-3,spec_bump/spec_0,
                 lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    a1.set_xlim(1.e-3, 3.)
    a1.set_xscale("log")
    a1.set_ylim(0.75, 1.25)
    a1.legend(prop={'size':11})
    #plt.xlabel('Recoil Energy T (keV)')
    a1.set(xlabel='Recoil Energy T (keV)', ylabel='Ratio')
    a0.axvline(x=1.e-3, color='k', linestyle=":")
    a1.axvline(x=10.e-3, color='k', linestyle=":")
    a1.axvline(x=50.e-3, color='k', linestyle=":")

    axins = inset_axes(a0, width=3, height=2, loc=3,
                       bbox_to_anchor=(0.12, 0.075),
                       bbox_transform=a0.transAxes)
    axins.xaxis.set_major_locator(plt.MaxNLocator(1))
    axins.xaxis.set_minor_locator(plt.MaxNLocator(1))
    axins.plot(t_arr*1.e-3,spec_0*1.e3,'k-',label='No Distortion',linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        axins.plot(t_arr*1.e-3, spec_bump*1.e3,
                   lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    zoom_lb = 0.4
    zoom_ub = 1.2
    axins.set_xlim(zoom_lb, zoom_ub)
    axins.set_xscale("log")
    axins.set_ylim(2.e-1, 2.e1)
    axins.set_yscale("log")

    # On all plots, shade in the zoomed region
    x_fill = np.array([zoom_lb, zoom_ub])
    a0.fill_between(x_fill, -10, 1.e13,
                    color='lightgrey')
    a1.fill_between(x_fill, -10, 1.e13,
                    color='lightgrey')

    plt.savefig('plots/reactor_bump_dsigmadT.png')

    nu_spec.bump_frac = old_frac

def plot_cevns_bump_split(nu_spec, bump_fracs, cns_bounds):
    old_frac = nu_spec.bump_frac

    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    plt.subplots_adjust(bottom=0.075, top=0.95)
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(9)
    lines_arr = ['b--', 'r-.', 'y:',
                 'c-', 'g--', 'm-.']

    #t_arr = np.logspace(0, 5, num=10000)
    t_arr = np.logspace(0, np.log10(3000), num=50)

    nu_spec.bump_frac = 0.
    spec_0 = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
    a0.loglog(t_arr*1.e-3,spec_0*1.e3,'k-',linewidth=2)
    bump_specs = []
    for i in range(len(bump_fracs)):
        nu_spec.bump_frac = bump_fracs[i]
        spec_bump = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
        bump_specs.append(spec_bump)
        a0.loglog(t_arr*1.e-3, spec_0*(1.-bump_fracs[i])*1.e3,
                  lines_arr[i],label="Non-Bump Spectrum",linewidth=2)
        a0.loglog(t_arr*1.e-3, (spec_bump-spec_0*(1.-bump_fracs[i]))*1.e3,
                  lines_arr[i][0]+'-',label="Bump Spectrum",linewidth=2)
        
    a0.set_ylim(cns_bounds)
    a0.set_xlim(1.e-3, 3.)
    a0.set(ylabel='Ge Differential Event Rate (dru)')
    a0.set_title("Ge Differential Rate at Chooz Reactor")
    a0.legend(prop={'size':11})
    a0.axvline(x=1.e-3, color='k', linestyle=":")
    a0.axvline(x=10.e-3, color='k', linestyle=":")
    a0.axvline(x=50.e-3, color='k', linestyle=":")

    a1.semilogx(t_arr*1.e-3, spec_0/spec_0, 'k-', label='No Distortion', linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        a1.semilogx(t_arr*1.e-3,spec_bump/spec_0,
                 lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    a1.set_ylim(0.75, 1.25)
    a1.set_xlim(1.e-3, 3.)
    a1.legend(prop={'size':11})
    #plt.xlabel('Recoil Energy T (keV)')
    a1.set(xlabel='Recoil Energy T (keV)', ylabel='Ratio')
    a0.axvline(x=1.e-3, color='k', linestyle=":")
    a1.axvline(x=10.e-3, color='k', linestyle=":")
    a1.axvline(x=50.e-3, color='k', linestyle=":")

    '''axins = inset_axes(a0, width=3.5, height=2.5, loc=3,
                       bbox_to_anchor=(0.24, 0.3),
                       bbox_transform=a0.transAxes)
    axins.loglog(t_arr*1.e-3,spec_0*1.e3,'k-',label='No Distortion',linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        axins.loglog(t_arr*1.e-3, spec_bump*1.e3,
                lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    axins.set_xlim(400., 1000.)
    axins.set_ylim(0., 0.02)'''

    plt.savefig('plots/reactor_bump_dsigmadT_split.png')

    nu_spec.bump_frac = old_frac

def plot_cevns_bump_targets(nu_spec, bump_frac, cns_bounds,
                            xscale="log"):
    old_frac = nu_spec.bump_frac

    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(9)
    colors_arr = ["#e41a1c", "#377eb8", "#4daf4a",
                  "#984ea3", "#ff7f00"]
    lines_arr = ['--', '-.', ':', '--', ':']

    if(xscale=="log"):
        #t_arr = np.logspace(0, np.log10(10000), num=500)
        t_arr = np.logspace(-2, np.log10(10000), num=250)
    else:
        t_arr = np.linspace(0, 10000, num=500)

    targets = ["Ge", "Zn", "Si",
               "Al2O3",
               "CaWO4"]
    labels = ['Ge', 'Zn', 'Si',
              r'Al$_2$O$_3$', r'CaWO$_4$']
    Z_arrs = [[32], [30], [14],
              [13, 8],
              [20, 74, 8]]
    N_arrs = [[72.64-32.], [35.38], [28.08-14],
              [26.982-13., 16.0-8.],
              [40.078-20., 183.84-74., 16.0-8.]]
    atom_arrs = [[1], [1], [1],
                 [2, 3],
                 [1, 1, 4]]

    no_bump_specs = []
    bump_specs = []
    for i in range(len(targets)):
        nu_spec.bump_frac = 0.
        spec_0 = dsigmadT_cns_rate_compound(t_arr, Z_arrs[i],
                                            N_arrs[i], atom_arrs[i],
                                            nu_spec)
        a0.plot(t_arr*1.e-3,spec_0*1.e3,'-',color='k',linewidth=1)
        no_bump_specs.append(spec_0)
        nu_spec.bump_frac = bump_frac
        spec_bump = dsigmadT_cns_rate_compound(t_arr, Z_arrs[i],
                                               N_arrs[i], atom_arrs[i],
                                               nu_spec)
        bump_specs.append(spec_bump)
        a0.plot(t_arr*1.e-3, spec_bump*1.e3,
                lines_arr[i],color=colors_arr[i],label=labels[i],linewidth=2)
    a0.set_xlim(1.e-5, 10.)
    a0.set_xscale(xscale)
    a0.set_ylim(cns_bounds)
    a0.set_yscale("log")
    a0.set(ylabel='Differential Event Rate (dru)')
    #a0.set_title("Differential Rate at Chooz Reactor")
    a0.axvline(x=1.e-3, color='k', linestyle=":")
    a0.axvline(x=10.e-3, color='k', linestyle=":")
    a0.axvline(x=50.e-3, color='k', linestyle=":")

    a1.plot(t_arr*1.e-3, no_bump_specs[4]/no_bump_specs[4],
            '-', color='k', label='No Distortion', linewidth=2)
    for i in range(len(targets)):
        spec_bump = bump_specs[i]
        a1.plot(t_arr*1.e-3,spec_bump/no_bump_specs[i],
                 lines_arr[i],color=colors_arr[i],label=labels[i],linewidth=2)
    a1.set_xlim(1.e-5, 10.)
    a1.set_xscale(xscale)
    a1.set_ylim(0.975, 1.1)
    #a1.legend(prop={'size':11}, framealpha=0.9)
    #plt.xlabel('Recoil Energy T (keV)')
    a1.set(xlabel='Recoil Energy T (keV)', ylabel='Ratio')
    a0.axvline(x=1.e-3, color='k', linestyle=":")
    a1.axvline(x=10.e-3, color='k', linestyle=":")
    a1.axvline(x=50.e-3, color='k', linestyle=":")
    a1.legend(prop={'size':14}, framealpha=0.99)

    if(xscale=="log"):
        axins = inset_axes(a0, width=2.7, height=1.8, loc=3,
                           bbox_to_anchor=(0.1, 0.15),
                           bbox_transform=a0.transAxes)
    else:
        axins = inset_axes(a0, width=2.7, height=1.8, loc=3,
                           bbox_to_anchor=(0.15, 0.55),
                           bbox_transform=a0.transAxes)
    axins.xaxis.set_major_locator(plt.MaxNLocator(1))
    axins.xaxis.set_minor_locator(plt.MaxNLocator(1))
    for i in [4,3,2,1,0]:
        if(i>1):
            alpha=0.5
        else:
            alpha=1.
        axins.plot(t_arr*1.e-3, no_bump_specs[i]*1.e3,
                   '-',color='k',linewidth=2, alpha=alpha)
        axins.plot(t_arr*1.e-3, bump_specs[i]*1.e3,
                   lines_arr[i],color=colors_arr[i],label=labels[i],
                   linewidth=2, alpha=alpha)
    zoom_lb = 0.4
    zoom_ub = 1.2
    axins.set_xlim(zoom_lb, zoom_ub)
    axins.set_xscale(xscale)
    axins.set_ylim(2.e-1, 2.e1)
    axins.set_yscale("log")
    axins.set(xlabel='Recoil Energy T (keV)')

    # On all plots, shade in the zoomed region
    x_fill = np.array([zoom_lb, zoom_ub])
    a0.fill_between(x_fill, -10, 1.e13,
                    color='lightgrey')
    a1.fill_between(x_fill, -10, 1.e13,
                    color='lightgrey')

    a0.grid()
    a1.grid()
    plt.grid()

    plt.savefig('plots/reactor_bump_dsigmadT_targets_'+xscale+'.pdf',
                bbox_inches='tight')

    nu_spec.bump_frac = old_frac

def plot_total_rate_vs_bump(nu_spec, bump_frac):
    old_frac = nu_spec.bump_frac

    fig = plt.figure()
    fig.patch.set_facecolor('white')

    threshold = 10.
    bump_arr = np.linspace(0., 1., 20)

    targets = ["Ge", "Zn", "Si",
               "Al2O3",
               "CaWO4"]
    Z_arrs = [[32], [30], [14],
              [13, 8],
              [20, 74, 8]]
    N_arrs = [[72.64-32.], [35.38], [28.08-14],
              [26.982-13., 16.0-8.],
              [40.078-20., 183.84-74., 16.0-8.]]
    atom_arrs = [[1], [1], [1],
                 [2, 3],
                 [1, 1, 4]]

    outstr = ""
    for i in range(5):
        rates = []
        for b in bump_arr:
            nu_spec.bump_frac = b
            rates.append(total_cns_rate_an_compound([threshold], 1e7, Z_arrs[i], N_arrs[i], atom_arrs[i], nu_spec)[0])
        plt.plot(bump_arr, rates, label=targets[i])
        outstr += "%s Rates (evts/kg/day)\n"%targets[i]
        outstr += "\tb=0.   : %.3e\n"%rates[0]
        nu_spec.bump_frac = bump_frac
        tmp_rate = total_cns_rate_an_compound([threshold], 1e7, Z_arrs[i], N_arrs[i], atom_arrs[i], nu_spec)[0]
        outstr += "\tb=%.4f: %.3e\n"%(bump_frac, tmp_rate)
        outstr += "\t\tIncrease: %.3e\n"%(tmp_rate-rates[0])
        outstr += "\t\t%% Increase: %.3f\n"%((tmp_rate-rates[0])/rates[0]*100.)
        outstr += "\tb=1.   : %.3e\n"%rates[-1]
    print(outstr)
    f = open("plots/bump_rates.txt", 'w')
    f.write(outstr)
    f.close()

    plt.legend(prop={'size':11})
    plt.xlabel("Bump Fraction")
    plt.ylabel("Total CEvNS Rate (evts/kg/day)")
    plt.title("Total CEvNS Rate, Tthr=%.1f eV"%threshold)
    plt.grid()
    plt.axvline(x=bump_frac, color='k', linestyle=":")
    plt.ylim(0., 50.)
    plt.xlim(0., 1.)
    plt.savefig('plots/total_event_rate_vs_bump_unzoomed.png')
    plt.xlim(0., 0.1)
    plt.savefig('plots/total_event_rate_vs_bump.png')

    nu_spec.bump_frac = old_frac


def plot_cevns_yields_vs_bump(nu_spec, bump_frac):
    old_frac = nu_spec.bump_frac

    fig = plt.figure()
    fig.patch.set_facecolor('white')

    threshold = 10.
    bump_arr = np.linspace(0., 1., 20)

    targets = ["Ge", "Zn", "Si",
               "Al2O3",
               "CaWO4"]
    Z_arrs = [[32], [30], [14],
              [13, 8],
              [20, 74, 8]]
    N_arrs = [[72.64-32.], [35.38], [28.08-14],
              [26.982-13., 16.0-8.],
              [40.078-20., 183.84-74., 16.0-8.]]
    atom_arrs = [[1], [1], [1],
                 [2, 3],
                 [1, 1, 4]]

    outstr = ""
    for i in range(5):
        yields = []
        for b in bump_arr:
            nu_spec.bump_frac = b
            yields.append(cevns_yield_compound(threshold, 1e7, Z_arrs[i], N_arrs[i], atom_arrs[i], nu_spec))
        plt.plot(bump_arr, yields, label=targets[i])
        outstr += "%s Yields (evts/kg/day)\n"%targets[i]
        outstr += "\tb=0.   : %.3e\n"%yields[0]
        nu_spec.bump_frac = bump_frac
        tmp_yield = cevns_yield_compound(threshold, 1e7, Z_arrs[i], N_arrs[i], atom_arrs[i], nu_spec)
        outstr += "\tb=%.4f: %.3e\n"%(bump_frac, tmp_yield)
        outstr += "\t\tIncrease: %.3e\n"%(tmp_yield-yields[0])
        outstr += "\t\t%% Increase: %.3f\n"%((tmp_yield-yields[0])/yields[0]*100.)
        outstr += "\tb=1.   : %.3e\n"%yields[-1]
    print(outstr)
    f = open("plots/bump_yields.txt", 'w')
    f.write(outstr)
    f.close()

    plt.legend(prop={'size':11})
    plt.xlabel("Bump Fraction")
    plt.ylabel("Total CEvNS Yield (evts/kg/day)")
    plt.title("Total Yield, Tthr=%.1f eV"%threshold)
    plt.grid()
    plt.axvline(x=bump_frac, color='k', linestyle=":")
    plt.xlim(0., 1.)
    plt.savefig('plots/yield_cevns_vs_bump_unzoomed.png')
    plt.xlim(0., 0.1)
    #plt.ylim(0., 100.)
    plt.savefig('plots/yield_cevns_vs_bump.png')

    nu_spec.bump_frac = old_frac


def plot_ibd_yield_vs_bump(nu_spec, bump_frac):
    old_frac = nu_spec.bump_frac

    fig = plt.figure()
    fig.patch.set_facecolor('white')

    bump_arr = np.linspace(0., 1., 100)

    yields = []
    for b in bump_arr:
        nu_spec.bump_frac = b
        yields.append(ibd_yield(nu_spec))
    plt.plot(bump_arr, yields)

    #plt.legend(prop={'size':11})
    plt.xlabel("Bump Fraction")
    plt.ylabel("IBD Yield (cm^2/fission)")
    plt.title("IBD Yield vs Bump Fraction")
    plt.grid()
    plt.axvline(x=bump_frac, color='k', linestyle=":")
    plt.xlim(0., 1.)
    plt.ylim(0., 1.e-42)
    plt.savefig('plots/yield_ibd_vs_bump.png')
    nu_spec.bump_frac = old_frac

def integrate_bump_counts_plot(nu_spec, bump_frac):
    old_frac = nu_spec.bump_frac

    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(9)
    colors_arr = ["#e41a1c", "#377eb8", "#4daf4a",
                  "#984ea3", "#ff7f00"]
    lines_arr = ['--', '-.', ':', '--', ':']

    #t_arr = np.logspace(0, np.log10(10000), num=500)
    #t_arr = np.logspace(-2, np.log10(10000), num=50)
    t_arr = np.linspace(0., 2500., num=50)

    targets = ["Ge", "Zn", "Si",
               "Al2O3",
               "CaWO4"]
    labels = ['Ge', 'Zn', 'Si',
              r'Al$_2$O$_3$', r'CaWO$_4$']
    Z_arrs = [[32], [30], [14],
              [13, 8],
              [20, 74, 8]]
    N_arrs = [[72.64-32.], [35.38], [28.08-14],
              [26.982-13., 16.0-8.],
              [40.078-20., 183.84-74., 16.0-8.]]
    atom_arrs = [[1], [1], [1],
                 [2, 3],
                 [1, 1, 4]]

    no_bump_specs = []
    bump_specs = []
    #for i in range(len(targets)):
    for i in range(1):
        nu_spec.bump_frac = 0.
        spec_0 = dsigmadT_cns_rate_compound(t_arr, Z_arrs[i],
                                            N_arrs[i], atom_arrs[i],
                                            nu_spec)
        if(i==0):
            a0.plot(t_arr*1.e-3,spec_0*1.e3,'-',color='k',linewidth=1)
        no_bump_specs.append(spec_0)
        nu_spec.bump_frac = bump_frac
        spec_bump = dsigmadT_cns_rate_compound(t_arr, Z_arrs[i],
                                               N_arrs[i], atom_arrs[i],
                                               nu_spec)
        bump_specs.append(spec_bump)
        if(i==0):
            a0.plot(t_arr*1.e-3, spec_bump*1.e3,
                    lines_arr[i],color=colors_arr[i],label=labels[i],linewidth=1)
            a0.fill_between(t_arr*1.e-3, spec_0*1.e3, spec_bump*1.e3,
                            where= spec_bump>spec_0,
                            color='lightgrey')

    a0.set_xlim(0., 2.0)
    a0.set_xscale("linear")
    #a0.set_xscale("log")
    a0.set_ylim([2.e-1, 6.e0])
    a0.set_yscale("log")
    #a0.set_yscale("linear")
    a0.set(ylabel='Differential Event Rate (dru)')
    a0.set_title("Differential Rate at Chooz Reactor, Bump Rate Integral Highlighted")
    a0.legend(prop={'size':11}, framealpha=0.9)
    #a0.axvline(x=1.e-3, color='k', linestyle=":")
    #a0.axvline(x=10.e-3, color='k', linestyle=":")
    #a0.axvline(x=50.e-3, color='k', linestyle=":")

    a1.plot(t_arr*1.e-3, no_bump_specs[0]/no_bump_specs[0],
            '-', color='k', label='No Distortion', linewidth=2)
    
    #for i in range(len(targets)):
    for i in range(1):
        spec_bump = bump_specs[i]
        if(i==0):
            a1.plot(t_arr*1.e-3,spec_bump/no_bump_specs[i],
                    lines_arr[i],color=colors_arr[i],label=labels[i],linewidth=2)
            a1.fill_between(t_arr*1.e-3, no_bump_specs[0]/no_bump_specs[0], spec_bump/no_bump_specs[i],
                            where= spec_bump>no_bump_specs[0],
                            color='lightgrey')
    a1.set_xlim(0., 2.0)
    a1.set_xscale("linear")
    #a1.set_xscale("log")
    a1.set_ylim(0.975, 1.1)
    #a1.legend(prop={'size':11}, framealpha=0.9)
    #plt.xlabel('Recoil Energy T (keV)')
    a1.set(xlabel='Recoil Energy T (keV)', ylabel='Ratio')
    a0.axvline(x=1.e-3, color='k', linestyle=":")
    a1.axvline(x=10.e-3, color='k', linestyle=":")
    a1.axvline(x=50.e-3, color='k', linestyle=":")

    zoom_lb = 0.4
    zoom_ub = 1.2

    # On all plots, shade in the zoomed region
    x_fill = np.array([zoom_lb, zoom_ub])

    a0.grid()
    a1.grid()
    plt.grid()

    plt.savefig('plots/reactor_bump_integral.png')

    nu_spec.bump_frac = old_frac


def integrate_bump_counts(nu_spec, bump_frac):
    old_frac = nu_spec.bump_frac

    t_arr = np.linspace(0., 10000., num=50)

    targets = ["Ge", "Zn", "Si",
               "Al2O3",
               "CaWO4_b1", "CaWO4_b2"]
    Z_arrs = [[32.], [30.], [14.],
              [13., 8.],
              [20., 74., 8.],
              [20., 74., 8.]]
    N_arrs = [[72.64-32.], [35.38], [28.08-14],
              [26.982-13., 16.0-8.],
              [40.078-20., 183.84-74., 16.0-8.],
              [40.078-20., 183.84-74., 16.0-8.]]
    atom_arrs = [[1.], [1.], [1.],
                 [2., 3.],
                 [1., 1., 4.],
                 [1., 1., 4.]]
    low_guesses = [250., 300., 700., 800., 100., 700.]
    high_guesses = [1100., 1300., 3000., 5500., 400., 5500.]

    print("")
    for i in range(len(targets)):
        def spec_0(t):
            nu_spec.bump_frac = 0.
            res = dsigmadT_cns_rate_compound(t, Z_arrs[i],
                                             N_arrs[i], atom_arrs[i],
                                             nu_spec)
            return res
        def spec_bump(t):
            nu_spec.bump_frac = bump_frac
            res = dsigmadT_cns_rate_compound(t, Z_arrs[i],
                                             N_arrs[i], atom_arrs[i],
                                             nu_spec)
            return res
        def spec_diff(t):
            return spec_bump(t)-spec_0(t)

        print("Target: %s"%targets[i])
        intersection_low = fsolve(lambda x : spec_0(x)[0] - spec_bump(x)[0], low_guesses[i])[0]
        intersection_high = fsolve(lambda x : spec_0(x)[0] - spec_bump(x)[0], high_guesses[i])[0]
        print("Intersection Low: %.1f eV"%intersection_low)
        print("Intersection High: %.1f eV"%intersection_high)
        print("Bump Width: %.1f eV"%(intersection_high-intersection_low))

        bump_counts = spint.quad(lambda t: spec_diff(t), intersection_low, intersection_high)[0]
        print("Bump Rate: %.5f evts/kg/day"%bump_counts)
        print("Bump Rate per eV: %.3e"%(bump_counts/(intersection_high-intersection_low)))
        #nu_spec.bump_frac = 0.
        #tot_no_bump = spint.quad(lambda t: spec_0(t), 10., 10000.)[0]
        #print("Total Rate Check (No Distortion): %.2f evts/kg/day"%tot_no_bump)

        print("")

    nu_spec.bump_frac = old_frac


if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # The averaged spectrum is stored in U-235
    fractions = [1.0, 0.0, 0.0, 0.0]

    # We will assum 60 m from a 4.25 GW reactor for NuCLEUS
    power = 4250
    distance = 6000 # cm

    # The stored spectra are in neutrinos/MeV/s for a 4250 MW reactor
    # reactor_tools will multiply by: power*200./2.602176565e-19
    # We need to rescale to undo this
    scale = 1./(power/200.0/1.602176565e-19)

    nu_spec = NeutrinoSpectrum(distance, power, False, *fractions,
                               include_other=True)
    nu_spec.initialize_d_r_d_enu("u235", "root",
                                 "../../../final_spectra/sum_U_Pu_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW.root",
                                 "nsim_Fission_avg",
                                 scale=scale)
    nu_spec.initialize_d_r_d_enu("u238", "zero")
    nu_spec.initialize_d_r_d_enu("pu239", "zero")
    nu_spec.initialize_d_r_d_enu("pu241", "zero")
    nu_spec.initialize_d_r_d_enu("other", "root",
                                 "../../../final_spectra/sum_U_Pu_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW.root",
                                 "nsim_U239_Np239_Pu239_avg",
                                 scale=scale)
    '''
    # IBD vs CEvNS
    plot_ibd(nu_spec)
    compare_ibd_cevns(nu_spec)

    # Fit Daya Bay data to get bump parameters
    fit_daya_bay_data()
    '''
    # Plot bump
    bump_frac = 0.018
    '''
    max_bump_ratio(nu_spec, bump_frac*0.9)
    max_bump_ratio(nu_spec, bump_frac*0.95)
    max_bump_ratio(nu_spec, bump_frac)
    max_bump_ratio(nu_spec, bump_frac*1.05)
    max_bump_ratio(nu_spec, bump_frac*1.1)

    plot_nu_bump_ibd(nu_spec, bump_fracs=[bump_frac])
    '''
    plot_nu_bump(nu_spec, bump_fracs=[bump_frac])
    '''

    plot_cevns_bump(nu_spec, bump_fracs=[bump_frac],
                    cns_bounds=[1.e-2, 1.e3])

    plot_cevns_bump_split(nu_spec, bump_fracs=[bump_frac],
                          cns_bounds=[1.e-5, 1.e4])
    '''
    plot_cevns_bump_targets(nu_spec, bump_frac=bump_frac,
                            cns_bounds=[1.e-4, 1.e4])
    '''
    plot_cevns_bump_targets(nu_spec, bump_frac=bump_frac,
                            cns_bounds=[1.e-4, 1.e4],
                            xscale="linear")

    plot_total_rate_vs_bump(nu_spec, bump_frac)
    #plot_cevns_yields_vs_bump(nu_spec, bump_frac)
    plot_ibd_yield_vs_bump(nu_spec, bump_frac)

    print("IBD XSec @ 3 MeV: %.3e"%total_XSec_ibd(3.e6))
    print("IBD XSec @ 6 MeV: %.3e"%total_XSec_ibd(6.e6))
    print("Ratio of IBD @ 6 vs 3: %.1f"%(total_XSec_ibd(6.e6)/total_XSec_ibd(3.e6)))
    print("CEvNS XSec @ 3 MeV: %.3e"%total_XSec_cns(10., 3.e6, 32., 72.64-32.))
    print("CEvNS XSec @ 6 MeV: %.3e"%total_XSec_cns(10., 6.e6, 32., 72.64-32.))
    print("Ratio of CEvNS @ 6 vs 3: %.1f"%(total_XSec_cns(10., 6.e6, 32., 72.64-32.)/total_XSec_cns(10., 3.e6, 32., 72.64-32.)))

    integrate_bump_counts_plot(nu_spec, bump_frac)
    integrate_bump_counts(nu_spec, bump_frac)
    '''
