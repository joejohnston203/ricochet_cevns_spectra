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

def plot_neutrino_spectrum_comparison(nu_spec1, nu_spec_mueller):
    '''
    Make a plot of the comparison of two neutrino spectra

    Args:
        nu_spec1, nu_spec_mueller: Initialized NeutrinoSpectrum object
    '''
    initial_fracs1 = nu_spec1.get_fractions()

    # Plot neutrino spectrum
    e_arr = np.linspace(0., 1e7, 100000)

    fractions = [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
    labels = ["U_235", "U_238", "Pu_239", "Pu_241"]
    colors = ["b-", "g-", "r-", "c-"]

    for i, curr_fracs in enumerate(fractions):
        fig0 = plt.figure()
        fig0.patch.set_facecolor('white')
        nu_spec1.set_fractions(curr_fracs)
        spec_tot = nu_spec1.d_r_d_enu(e_arr*1.e-3)
        plt.plot(e_arr*1e-6,spec_tot*1e3, colors[i],linewidth=1)
        plt.legend(prop={'size':11})
        plt.xlabel('Neutrino Energy (MeV)')
        plt.ylabel('Neutrinos/ Fission/ keV')
        plt.title(labels[i]+' Neutrino Spectrum per Fission')
        plt.savefig('plots/neutrino_spectrum_'+labels[i]+'.png')
        fig0.clf()

    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    for i, curr_fracs in enumerate(fractions):
        nu_spec_mueller.set_fractions(curr_fracs)
        spec_tot = nu_spec_mueller.d_r_d_enu(e_arr*1.e-3)
        plt.plot(e_arr*1e-6,spec_tot*1e3, colors[i],linewidth=1)
        plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Neutrinos/ Fission/ keV')
    plt.title('Mueller Neutrino Spectrum per Fission')
    plt.savefig('plots/neutrino_spectrum_mueller.png')
    fig0.clf()

    # Compare to 2nd spectrum
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    plt.fill_between([-1., 2.], [-1., -1.], [1., 1.],
                     alpha=0.5, color='lightgray')
    for i, curr_fracs in enumerate(fractions):
        nu_spec1.set_fractions(curr_fracs)
        spec_tot1 = nu_spec1.d_r_d_enu(e_arr*1.e-3)
        nu_spec_mueller.set_fractions(curr_fracs)
        spec_tot2 = nu_spec_mueller.d_r_d_enu(e_arr*1.e-3)
        diff = (spec_tot1-spec_tot2)/spec_tot1
        plt.plot(e_arr*1e-6,diff, colors[i], label=labels[i], linewidth=1)
    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('(Spectrum - Mueller)/Spectrum')
    plt.ylim(-0.5, 0.5)
    plt.xlim(0., 10.)
    plt.title('Fractional Difference With Mueller')
    plt.savefig('plots/mueller_comparison.png')
    fig0.clf()
    

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # Separately turn on each spectrum in the comparison method
    fractions = [0.0, 0.0, 0.0, 0.0]

    # TO DO: 
    # Set so we just have neutrinos/fission
    power = 8500
    # Set so we just have neutrinos/fission
    distance = 8000 # cm
    # Set so we just have neutrinos/fission
    #scale = 1.0*5.7/(7.2e20)
    scale = 1.0

    nu_spec = NeutrinoSpectrum(distance, power, False, *fractions)
    nu_spec.initialize_d_r_d_enu("u235", "root",
                                 "../../data/TBS_235U_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_thermal_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("u238", "root",
                                 "../../data/TBS_238U_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_fast_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("pu239", "root",
                                 "../../data/TBS_239Pu_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_thermal_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("pu241", "root",
                                 "../../data/TBS_241Pu_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_thermal_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("other", "zero")

    # Mueller spectra
    nu_spec_mueller = NeutrinoSpectrum(nu_spec.distance, nu_spec.power, True,
                                       fractions)
    nu_spec_mueller.initialize_d_r_d_enu("u235", "txt",
                                         "../../data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("u238", "mueller")
    nu_spec_mueller.initialize_d_r_d_enu("pu239", "txt",
                                         "../../data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("pu241", "txt",
                                         "../../data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("other", "mueller")

    plot_neutrino_spectrum_comparison(nu_spec, nu_spec_mueller)
