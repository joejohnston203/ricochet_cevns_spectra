import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import ibd_rate_per_kg_per_year, total_cns_rate_an, total_cns_rate_an_compound

def calculate_ibd_rates(nu_spec, nu_spec_mueller=None):
    frac = nu_spec.get_fractions()
    frac_mueller = nu_spec_mueller.get_fractions()

    isotopes = ["U-235", "U-238", "Pu-239", "Pu-241"]
    print("")
    for i in range(len(isotopes)):
        temp_fracs = [0., 0., 0., 0.]
        temp_fracs[i] = 1.
        nu_spec.set_fractions(temp_fracs)
        print("%s rate/kg/yr: %.3e"%
              (isotopes[i], ibd_rate_per_kg_per_year(nu_spec)))
        nu_spec_mueller.set_fractions(temp_fracs)
        print("Mueller/Huber %s rate/kg/yr (will differ from summation): %.3e"%
              (isotopes[i], ibd_rate_per_kg_per_year(nu_spec_mueller)))
        print("")
    nu_spec.set_fractions(frac)
    nu_spec_mueller.set_fractions(frac_mueller)

def plot_cevns_rate_vs_threshold(nu_spec, num_points=100):
    # Make a plot of integrated event rate per eV vs threshold energy
    t_arr = np.logspace(1, 3, num=num_points)

    targets = ["C",  "Ne",   "Si",  "Ar",  "Ge",  "Xe",    "W"]
    Z       = [6.,   10.,    14.,   18.,   32.,   54.,     74.]
    A       = [12.,  20.180, 28.08, 39.95, 72.64, 131.293, 183.84]
    guess   = [791., 782.,   707.,  677.,  496.,  352.,    281.]
    colors  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]
    lines   = ['-', '-.', '--', ':', '-', '-.', '--', ':']
    ibd_rate = 429.

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i in range(len(targets)):
        plt.loglog(t_arr,
                   365.*total_cns_rate_an(t_arr, 1e7, Z[i], A[i]-Z[i], nu_spec),
                   label=targets[i],
                   color=colors[i], linestyle=lines[i], linewidth=2)
        intersection = fsolve(lambda x : 365.*total_cns_rate_an(x, 1e7, Z[i], A[i]-Z[i], nu_spec)[0] - ibd_rate, guess[i])[0]
        print("%s Intersection: %.1f eV, %% Agreement: %.2f"%
              (targets[i], intersection, 100.*(abs(guess[i]-intersection)/guess[i])))
        
    plt.axhline(y=ibd_rate, label=r"CH$_2$ IBD", color="#f781bf")
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Threshold (eV)')
    plt.xlim(1.e1, 1.e3)
    plt.ylabel('Event Rate (Events/kg/year)')
    plt.ylim(1.e2, 5.e4)
    plt.title("$^{235}$U Reactor (%s) Total CEvNS Rate"%
              nu_spec.get_settings_string())
    plt.grid(True)
    plt.savefig('plots/total_cevns_rate_vs_threshold.png')
    plt.xscale("linear")
    plt.yscale("linear")
    plt.ylim(0., 1.e4)
    plt.savefig('plots/total_cevns_rate_vs_threshold_linear.png')

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # Assume entire contributino is U-235
    fractions = [1.0, 0.0, 0.0, 0.0]

    # Reasonable research reactor assumptions
    power = 100 # MW
    distance = 1000 # cm
    # The spectra are already stored as neutrinos/fission
    scale = 1.0

    # Bestiole Results
    nu_spec = NeutrinoSpectrum(distance, power, False, *fractions)
    nu_spec.initialize_d_r_d_enu("u235", "root",
                                 "../../../final_spectra/TBS_235U_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_thermal_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("u238", "root",
                                 "../../../final_spectra/TBS_238U_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_fast_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("pu239", "root",
                                 "../../../final_spectra/TBS_239Pu_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_thermal_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("pu241", "root",
                                 "../../../final_spectra/TBS_241Pu_beta_10keV_10gspt_Paul_reprocess_2017TAGS_FERMI.screen.QED.aW_thermal_cumulative_JEFF3.1.1.root",
                                 "nsim100")
    nu_spec.initialize_d_r_d_enu("other", "zero")

    # Mueller spectra
    nu_spec_mueller = NeutrinoSpectrum(nu_spec.distance, nu_spec.power, True,
                                       *fractions)
    nu_spec_mueller.initialize_d_r_d_enu("u235", "txt",
                                         "../../data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("u238", "mueller")
    nu_spec_mueller.initialize_d_r_d_enu("pu239", "txt",
                                         "../../data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("pu241", "txt",
                                         "../../data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("other", "mueller")
    
    calculate_ibd_rates(nu_spec, nu_spec_mueller)
    plot_cevns_rate_vs_threshold(nu_spec, num_points=100)
