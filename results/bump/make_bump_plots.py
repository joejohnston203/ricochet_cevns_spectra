from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, cns_total_rate_integrated, cns_total_rate_integrated_compound, total_XSec_cns

import numpy as np
from scipy.optimize import curve_fit, fmin
import scipy.integrate as spint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os

Mn = cevns_spectra.Mn
Mn_eV = Mn*1e3

s_per_day = 60.0*60.0*24.0

def max_bump_ratio(nu_spec, bump_frac):
    print("Total flux: %.2e"%nu_spec.nuFlux())
    def ratio(e):
        nu_spec.bump_frac = 0.0
        rate0 = nu_spec.d_phi_d_enu_ev(e)
        nu_spec.bump_frac = bump_frac
        rate1 = nu_spec.d_phi_d_enu_ev(e)
        return rate1/rate0
    res = fmin(lambda x: -ratio(x), 6.e6)
    print("bump_frac: %.3f"%bump_frac)
    print("Max ratio occurs at: %s"%res)
    print("Max ratio: %.2f"%ratio(res))

def plot_nu_bump(nu_spec,
                 bump_fracs=[0.01, 0.05]):
    old_frac = nu_spec.bump_frac

    # Plot unzoomed neutrino flux and ratio
    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    #sharex=True)
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(9)
    lines_arr = ['b--', 'r-.', 'y:',
                 'c-', 'g--', 'm-.']
    e_arr = np.linspace(0., 1e7, 100000)

    nu_spec.bump_frac = 0.
    spec_tot = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    a0.plot(e_arr*1e-6,spec_tot*1e6,'k-',label="No Bump",linewidth=2)
    bump_specs = []
    for i in range(len(bump_fracs)):
        nu_spec.bump_frac = bump_fracs[i]
        spec = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
        bump_specs.append(spec)
        a0.plot(e_arr*1e-6,spec*1e6,lines_arr[i],
                 label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)

    a0.set_ylim(0., 0.4e18)
    a0.set_xlim(0., 10.)
    a0.set(ylabel='Flux, nu/(MeV*day*cm^2)')
    a0.set_title('Neutrino Flux at Chooz Reactor')

    a1.plot(e_arr*1e-6,spec_tot/spec_tot,'k-',label="No Bump",linewidth=2)
    for i in range(len(bump_fracs)):
        spec = bump_specs[i]
        plt.plot(e_arr*1e-6,spec/spec_tot,lines_arr[i],
                 label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    a1.legend(loc=2, prop={'size':11})
    a1.set_xlim(0., 10.)
    a1.set_ylim(0.75, 1.25)
    a1.set(xlabel='Neutrino Energy (MeV)', ylabel='Ratio')

    axins = inset_axes(a0, width=3.5, height=2.5, loc=3,
                       bbox_to_anchor=(0.24, 0.3),
                       bbox_transform=a0.transAxes)
    axins.plot(e_arr*1e-6,spec_tot*1e6,'k-',label="No Bump",linewidth=2)
    for i in range(len(bump_fracs)):
        spec = bump_specs[i]
        axins.plot(e_arr*1e-6,spec*1e6,lines_arr[i],
                   label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    axins.set_xlim(4.5, 7.5)
    axins.set_ylim(0., 6.e15)

    plt.savefig('plots/reactor_bump_neutrino_spectrum.')
    nu_spec.bump_frac = old_frac

def plot_cevns_bump(nu_spec, bump_fracs, cns_bounds):
    old_frac = nu_spec.bump_frac

    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(9)
    lines_arr = ['b--', 'r-.', 'y:',
                 'c-', 'g--', 'm-.']

    t_arr = np.linspace(0, 10000, num=10000)

    nu_spec.bump_frac = 0.
    spec_0 = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
    a0.plot(t_arr,spec_0,'k-',label='No Bump',linewidth=2)
    bump_specs = []
    for i in range(len(bump_fracs)):
        nu_spec.bump_frac = bump_fracs[i]
        spec_bump = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
        bump_specs.append(spec_bump)
        a0.plot(t_arr, spec_bump,
                lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    a0.set_xlim(0., 3000.)
    a0.set_ylim(cns_bounds)
    a0.set(ylabel='Ge Differential Event Rate (Events/kg/day/eV)')
    a0.set_title("Ge Differential Rate at Chooz Reactor")
    a0.axvline(x=1., color='k', linestyle=":")
    a0.axvline(x=10., color='k', linestyle=":")
    a0.axvline(x=50., color='k', linestyle=":")

    a1.plot(t_arr, spec_0/spec_0, 'k-', label='No Bump', linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        a1.plot(t_arr,spec_bump/spec_0,
                 lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    a1.set_xlim(0., 3000.)
    a1.set_ylim(0.75, 1.25)
    a1.legend(prop={'size':11})
    plt.xlabel('Recoil Energy T (eV)')
    a1.set(xlabel='Recoil Energy T (eV)', ylabel='Ratio')
    a0.axvline(x=1., color='k', linestyle=":")
    a1.axvline(x=10., color='k', linestyle=":")
    a1.axvline(x=50., color='k', linestyle=":")

    axins = inset_axes(a0, width=3.5, height=2.5, loc=3,
                       bbox_to_anchor=(0.24, 0.3),
                       bbox_transform=a0.transAxes)
    axins.plot(t_arr,spec_0,'k-',label='No Bump',linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        axins.plot(t_arr, spec_bump,
                lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    axins.set_xlim(400., 1000.)
    axins.set_ylim(0., 0.02)

    plt.savefig('plots/reactor_bump_dsigmadT.png')

    nu_spec.bump_frac = old_frac

def plot_cevns_bump_split(nu_spec, bump_fracs, cns_bounds):
    old_frac = nu_spec.bump_frac

    fig1, (a0, a1) = plt.subplots(2, 1,
                                  gridspec_kw={'height_ratios': [2, 1],})
    fig1.patch.set_facecolor('white')
    fig1.set_figheight(9)
    lines_arr = ['b--', 'r-.', 'y:',
                 'c-', 'g--', 'm-.']

    #t_arr = np.logspace(0, 5, num=10000)
    t_arr = np.logspace(0, np.log10(3000), num=50)

    nu_spec.bump_frac = 0.
    spec_0 = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
    a0.loglog(t_arr,spec_0,'k-',linewidth=2)
    bump_specs = []
    for i in range(len(bump_fracs)):
        nu_spec.bump_frac = bump_fracs[i]
        spec_bump = dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec)
        bump_specs.append(spec_bump)
        a0.loglog(t_arr, spec_0*(1.-bump_fracs[i]),
                  lines_arr[i],label="Non-Bump Spectrum",linewidth=2)
        a0.loglog(t_arr, spec_bump-spec_0*(1.-bump_fracs[i]),
                  lines_arr[i][0]+'-',label="Bump Spectrum",linewidth=2)
        print("bump_frac: %s"%bump_fracs[i])
        print("ratio: %s"%((spec_bump-spec_0*(1.-bump_fracs[i]))/
                           spec_0*(1.-bump_fracs[i])))
        print("bump: %s"%(spec_bump-spec_0*(1.-bump_fracs[i])))
        print("nonbump: %s"%(spec_0*(1.-bump_fracs[i])))
        
        
    a0.set_ylim(cns_bounds)
    a0.set_xlim(1., 3000.)
    a0.set(ylabel='Ge Differential Event Rate (Events/kg/day/eV)')
    a0.set_title("Ge Differential Rate at Chooz Reactor")
    a0.legend(prop={'size':11})
    a0.axvline(x=10., color='k', linestyle=":")
    a0.axvline(x=50., color='k', linestyle=":")

    a1.semilogx(t_arr, spec_0/spec_0, 'k-', label='No Bump', linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        a1.semilogx(t_arr,spec_bump/spec_0,
                 lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    a1.set_ylim(0.75, 1.25)
    a1.set_xlim(1., 3000.)
    a1.legend(prop={'size':11})
    plt.xlabel('Recoil Energy T (eV)')
    a1.set(xlabel='Recoil Energy T (eV)', ylabel='Ratio')
    a0.axvline(x=1., color='k', linestyle=":")
    a1.axvline(x=10., color='k', linestyle=":")
    a1.axvline(x=50., color='k', linestyle=":")

    '''axins = inset_axes(a0, width=3.5, height=2.5, loc=3,
                       bbox_to_anchor=(0.24, 0.3),
                       bbox_transform=a0.transAxes)
    axins.loglog(t_arr,spec_0,'k-',label='No Bump',linewidth=2)
    for i in range(len(bump_fracs)):
        spec_bump = bump_specs[i]
        axins.loglog(t_arr, spec_bump,
                lines_arr[i],label="Bump=%.2f%%"%(100.*bump_fracs[i]),linewidth=2)
    axins.set_xlim(400., 1000.)
    axins.set_ylim(0., 0.02)'''

    plt.savefig('plots/reactor_bump_dsigmadT_split.png')

    nu_spec.bump_frac = old_frac

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # The averaged spectrum is stored in U-235
    fractions = [1.0, 0.0, 0.0, 0.0]

    # 8.5 GW
    power = 8500

    # NuCLEUS: 60 m
    distance = 6000 # cm

    # The stored spectra are in neutrinos/MeV/s
    # We want to rescale to get neutrinos/MeV/fission
    # Divide by fissions/s: 7.2e20
    # TODO: CHECK THESE!!!
    scale = 1.0*5.7/(7.2e20)

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

    bump_frac = 0.0007

    # Plot bump
    max_bump_ratio(nu_spec, 0.0007)
    #plot_nu_bump(nu_spec, bump_fracs=[0.0007, 0.0015])
    #plot_cevns_bump(nu_spec, bump_fracs=[0.0007, 0.0015],
    #                cns_bounds=[0., 1.])
    plot_cevns_bump_split(nu_spec, bump_fracs=[0.0007, 0.1, 0.5],
                          cns_bounds=[1.e-12, 10.])
