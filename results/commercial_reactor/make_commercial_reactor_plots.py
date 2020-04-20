from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, cns_total_rate_integrated, cns_total_rate_integrated_compound

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

Mn = cevns_spectra.Mn
Mn_eV = Mn*1e3

s_per_day = 60.0*60.0*24.0

def plot_neutrino_spectrum_comparison(nu_spec1, nu_spec_kopeikin,
                                      num_points=1000):
    '''
    Make a plot of the comparison of two neutrino spectra

    Args:
        nu_spec1, nu_spec_kopeikin: Initialized NeutrinoSpectrum object
    '''
    # Plot neutrino spectrum + kopeikin spectrum
    e_arr = np.linspace(0., 1e7, num_points)
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    spec_tot1 = s_per_day*nu_spec1.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_tot1*1e6, "r-",linewidth=1, label="Bestiole")
    spec_tot_kopeikin = s_per_day*nu_spec_kopeikin.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_tot_kopeikin*1e6, "b-", linewidth=1, label="Kopeikin")
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Flux, nu/(MeV*day*cm^2)')
    plt.title('Commercial Reactor (%s) Neutrino Flux'%
              nu_spec1.get_settings_string())
    plt.savefig('plots/commercial_reactor_neutrino_spectrum.png')
    fig0.clf()

    # Difference
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    #diff = (spec_tot1-spec_tot_kopeikin)/spec_tot1
    diff = spec_tot1/spec_tot_kopeikin
    plt.plot(e_arr*1e-6,diff, "k-", linewidth=1)
    plt.plot(e_arr*1e-6,0*e_arr+1.0, "r--", linewidth=1)
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Bestiole/Kopeikin')
    plt.ylim(0., 2.)
    plt.xlim(0., 4.)
    plt.title('Commercial Reactor Spectrum Comparison With Kopeikin')
    plt.savefig('plots/commercial_reactor_kopeikin_comparison.png')
    fig0.clf()
    
def plot_dsigmadT_cns_rate(nu_spec,
                           bounds=[1e-4, 1e1],
                           num_points=100):
    t_arr = np.logspace(0, 3, num=num_points)
    
    fig3 = plt.figure()
    fig3.patch.set_facecolor('white')
    plt.ylim(bounds)
    plt.xlim(1e0, 1e3)
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, 14, 28.08-14, nu_spec),'g-',label='Si (A=28.1)',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, 30, 35.38, nu_spec),'b-',label='Zn (A=64.4)',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, 32, 72.64-32., nu_spec),'r-',label='Ge (A=72.6)',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns_rate_compound(t_arr, [13, 8], [26.982-13., 16.0-8.], [2, 3], nu_spec),'c-.',label='Al2O3 (A~20)',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns_rate_compound(t_arr, [20, 74, 8], [40.078-20., 183.84-74., 16.0-8.], [1, 1, 4], nu_spec),'m:',label='CaWO4 (A~48)',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Energy T (eV)')
    plt.ylabel('Differential Event Rate (Events/kg/day/eV)')
    plt.title("Commercial Reactor (%s) Differential Rate"%
              nu_spec.get_settings_string())
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig('plots/commercial_reactor_dsigmadT_event_rate.png')
    fig3.clf()

def plot_total_cns_rate(nu_spec, num_points=100):
    # Make a plot of integrated event rate per eV vs threshold energy
    t_arr = np.logspace(0, 3, num=num_points)
    
    fig4 = plt.figure()
    fig4.patch.set_facecolor('white')
    plt.loglog(t_arr,total_cns_rate_an(t_arr, 1e7, 14, 28.08-14, nu_spec),'g-',label='Si (A=28.1)',linewidth=2)
    plt.loglog(t_arr,total_cns_rate_an(t_arr, 1e7, 30, 35.38, nu_spec),'b-',label='Zn (A=64.4)',linewidth=2)
    plt.loglog(t_arr,total_cns_rate_an(t_arr, 1e7, 32, 72.64-32., nu_spec),'r-',label='Ge (A=72.6)',linewidth=2)
    plt.loglog(t_arr,total_cns_rate_an_compound(t_arr, 1e7, [13, 8], [26.982-13., 16.0-8.], [2, 3], nu_spec),'c-.',label='Al2O3 (A~20)',linewidth=2)
    plt.loglog(t_arr,total_cns_rate_an_compound(t_arr, 1e7, [20, 74, 8], [40.078-20., 183.84-74., 16.0-8.], [1, 1, 4], nu_spec),'m:',label='CaWO4 (A~48)',linewidth=2)
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Threshold (eV)')
    plt.ylabel('Event Rate (Events/kg/day)')
    plt.title("Commercial Reactor (%s) Total Rate"%
              nu_spec.get_settings_string())
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig('plots/commercial_reactor_event_rate_integrated.png')

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # The averaged spectrum is stored in U-235
    fractions = [1.0, 0.0, 0.0, 0.0]

    # 8.5 GW
    power = 8500

    # Very near site: 80 m
    distance = 8000 # cm

    # The stored spectra are in neutrinos/MeV/s
    # We want to rescale to get neutrinos/MeV/fission
    # Divide by fissions/s: 7.2e20
    scale = 1.0*5.7/(7.2e20)

    nu_spec = NeutrinoSpectrum(distance, power, False, *fractions,
                               include_other=True)
    nu_spec.initialize_d_r_d_enu("u235", "root",
                                 "../../data/sum_U_Pu_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW.root",
                                 "nsim_Fission_avg",
                                 scale=scale)
    nu_spec.initialize_d_r_d_enu("u238", "zero")
    nu_spec.initialize_d_r_d_enu("pu239", "zero")
    nu_spec.initialize_d_r_d_enu("pu241", "zero")
    nu_spec.initialize_d_r_d_enu("other", "root",
                                 "../../data/sum_U_Pu_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW.root",
                                 "nsim_U239_Np239_Pu239_avg",
                                 scale=scale)

    # Kopeikin spectra
    nu_spec_kopeikin = NeutrinoSpectrum(nu_spec.distance, nu_spec.power, False,
                                       *fractions)
    nu_spec_kopeikin.initialize_d_r_d_enu("u235", "txt",
                                         "../../data/kopeikin_spectrum.txt")
    nu_spec_kopeikin.initialize_d_r_d_enu("u238", "zero")
    nu_spec_kopeikin.initialize_d_r_d_enu("pu239", "zero")
    nu_spec_kopeikin.initialize_d_r_d_enu("pu241", "zero")
    nu_spec_kopeikin.initialize_d_r_d_enu("other", "zero")

    # Make Plots
    plot_neutrino_spectrum_comparison(nu_spec, nu_spec_kopeikin, num_points=1000)
    plot_dsigmadT_cns_rate(nu_spec, num_points=100)
    plot_total_cns_rate(nu_spec, num_points=100)
