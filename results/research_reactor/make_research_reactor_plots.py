from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, cns_total_rate_integrated, cns_total_rate_integrated_compound, get_atomic_arrs

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

def store_reactor_flux_kev(nu_spec1, outfile="flux.txt",
                           e_kev_lb=0.,
                           e_kev_ub=1.e4,
                           num_points=10000):
    '''
    Store the flux in a text file

    Output is in two columns:
    E_nu (keV), Flux (nu/(keV*day*cm^2))

    The stored flux is always from 1 eV to 10 MeV. e_kev_lb
    and e_kev_ub are used to set the stored flux to 0
    outside of that region
    '''
    e_arr = np.linspace(1., 1.e7, num_points)
    spec = s_per_day*nu_spec1.d_phi_d_enu_ev(e_arr)*1e3
    spec[np.logical_or(e_arr<e_kev_lb*1000.,
                       e_arr>e_kev_ub*1000.)] = 0.
    np.savetxt(outfile, np.stack((e_arr/1.e3, spec), 1),
               header="E_nu (keV), Flux (nu/(keV*day*cm^2))")

def plot_neutrino_spectrum_comparison(nu_spec1, nu_spec_mueller,
                                      num_points=1000):
    '''
    Make a plot of the comparison of two neutrino spectra

    Args:
        nu_spec1, nu_spec_mueller: Initialized NeutrinoSpectrum object
    '''
    # Plot neutrino spectrum + mueller spectrum
    e_arr = np.linspace(0., 1e7, num_points)
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    spec_tot1 = s_per_day*nu_spec1.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_tot1*1e6, "r-",linewidth=1, label="Bestiole")
    spec_tot_mueller = s_per_day*nu_spec_mueller.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_tot_mueller*1e6, "b-", linewidth=1, label="Mueller")
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Flux, nu/(MeV*day*cm^2)')
    plt.title('Research Reactor (%s) Neutrino Flux'%
              nu_spec1.get_settings_string())
    plt.savefig('plots/research_reactor_neutrino_spectrum.png')
    fig0.clf()

    # Difference
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    plt.fill_between([-1., 2.], [-1., -1.], [3., 3.],
                     alpha=0.5, color='lightgray')
    diff = spec_tot1/spec_tot_mueller
    plt.plot(e_arr*1e-6,diff, "k-", linewidth=1)
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Bestiole/Mueller')
    plt.ylim(0.75, 1.25)
    plt.xlim(0., 10.)
    plt.title('Research Reactor Spectrum Comparison with Mueller')
    plt.savefig('plots/research_reactor_mueller_comparison.png')
    fig0.clf()
    
def plot_dsigmadT_cns_rate(nu_spec,
                           bounds=[1e-4, 1e1],
                           num_points=100):
    t_arr = np.logspace(0, 3, num=num_points)
    
    fig3 = plt.figure()
    fig3.patch.set_facecolor('white')
    plt.ylim(bounds)
    plt.xlim(1e0, 1e3)
    labels = ["Si", "Zn", "Ge", "Al2O3", "CaWO4"]
    lines = ['g-', 'b-', 'r-', 'c-.', 'm:']
    widths = [1,1,1,2,2]
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        plt.loglog(t_arr,dsigmadT_cns_rate_compound(t_arr, Z_arr, N_arr, atom_arr, nu_spec),lines[i],label=labels[i],linewidth=widths[i])
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Energy T (eV)')
    plt.ylabel('Differential Event Rate (Events/kg/day/eV)')
    plt.title("Research Reactor (%s) Differential Rate"%
              nu_spec.get_settings_string())
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig('plots/research_reactor_dsigmadT_event_rate.png')
    fig3.clf()

def plot_total_cns_rate(nu_spec, num_points=100):
    # Make a plot of integrated event rate per eV vs threshold energy
    t_arr = np.logspace(0, 3, num=num_points)
    
    fig4 = plt.figure()
    fig4.patch.set_facecolor('white')
    labels = ["Si", "Zn", "Ge", "Al2O3", "CaWO4"]
    lines = ['g-', 'b-', 'r-', 'c-.', 'm:']
    widths = [1,1,1,2,2]
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        plt.loglog(t_arr,total_cns_rate_an_compound(t_arr, 1e7, Z_arr, N_arr, atom_arr, nu_spec),lines[i],label=labels[i],linewidth=widths[i])
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Threshold (eV)')
    plt.ylabel('Event Rate (Events/kg/day)')
    plt.title("Research Reactor (%s) Total Rate"%
              nu_spec.get_settings_string())
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig('plots/research_reactor_event_rate_integrated.png')

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # The averaged spectrum is stored in U-235
    fractions = [1.0, 0.0, 0.0, 0.0]

    # Reasonable research reactor assumptions
    power = 58.3 # 58.3 MW is stored in the file: see scale
    distance = 800 # cm
    # The stored spectra are in neutrinos/MeV/s for a 58.3 MW reactor
    # reactor_tools will multiply by: power*200./1.602176565e-19
    # We need to rescale to undo this
    scale = 1./(power/200.0/1.602176565e-19)

    nu_spec = NeutrinoSpectrum(distance, power, False, *fractions)
    nu_spec.initialize_d_r_d_enu("u235", "root",
                                 "../../../final_spectra/sum_U_Pu_research_reactor_20gspt_Tengblad-TAGSnew-ENSDF2020-Qbeta5br_FERMI.screen.QED.aW.root",
                                 "nsim_Fission_avg", scale=scale)
    nu_spec.initialize_d_r_d_enu("u238", "zero")
    nu_spec.initialize_d_r_d_enu("pu239", "zero")
    nu_spec.initialize_d_r_d_enu("pu241", "zero")
    nu_spec.initialize_d_r_d_enu("other", "root",
                                 "../../../final_spectra/sum_U_Pu_research_reactor_20gspt_Tengblad-TAGSnew-ENSDF2020-Qbeta5br_FERMI.screen.QED.aW.root",
                                 "nsim_Al28_Mn56_U239_Np239_avg",
                                 scale=scale)

    # Mueller spectra
    nu_spec_mueller = NeutrinoSpectrum(nu_spec.distance, power, True,
                                       0.993, 0.0, 0.007, 0.0)
    nu_spec_mueller.initialize_d_r_d_enu("u235", "txt",
                                         "../../data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("u238", "mueller")
    nu_spec_mueller.initialize_d_r_d_enu("pu239", "txt",
                                         "../../data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("pu241", "txt",
                                         "../../data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("other", "mueller")

    # Make Plots
    store_reactor_flux_kev(nu_spec, "flux_research_reactor_all.txt")
    store_reactor_flux_kev(nu_spec,
                           "flux_research_reactor_lt1800.txt",
                           0., 1800.)
    store_reactor_flux_kev(nu_spec,
                           "flux_research_reactor_gt1800.txt",
                           1800., 1.e4)

    nu_spec.include_other = False
    nu_spec.set_fractions(fractions)
    store_reactor_flux_kev(nu_spec, "flux_research_reactor_fission.txt")
    store_reactor_flux_kev(nu_spec, "flux_research_reactor_fission_lt1800.txt",
                           0., 1800.)
    store_reactor_flux_kev(nu_spec, "flux_research_reactor_fission_gt1800.txt",
                           1800., 1.e4)
    nu_spec.include_other = True
    nu_spec.set_fractions([0., 0., 0., 0.])
    store_reactor_flux_kev(nu_spec, "flux_research_reactor_nonfission.txt")
    store_reactor_flux_kev(nu_spec, "flux_research_reactor_nonfission_lt1800.txt",
                           0., 1800.)
    store_reactor_flux_kev(nu_spec, "flux_research_reactor_nonfission_gt1800.txt",
                           1800., 1.e4)
    nu_spec.set_fractions(fractions)

    plot_neutrino_spectrum_comparison(nu_spec, nu_spec_mueller, num_points=1000)
    plot_dsigmadT_cns_rate(nu_spec, num_points=100)
    plot_total_cns_rate(nu_spec, num_points=100)

