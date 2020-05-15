from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, cns_total_rate_integrated, cns_total_rate_integrated_compound, total_XSec_cns

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
    
def plot_neutrino_spectrum_other(nu_spec, num_points=1000):
    '''
    Make a plot of the fission and other spectra

    Args:
        nu_spec: Initialized NeutrinoSpectrum object
    '''
    # Plot neutrino spectrum + kopeikin spectrum
    e_arr = np.linspace(0., 8.e6, num_points)
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    spec_tot = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.semilogy(e_arr*1e-6,spec_tot*1e6, "k-",linewidth=1, label="Total")
    
    include_other = nu_spec.include_other
    nu_spec.include_other = False
    spec_fission = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.semilogy(e_arr*1e-6,spec_fission*1e6, 'r:', linewidth=1, label="Fission")
    nu_spec.include_other = include_other
    
    fractions = nu_spec.get_fractions()
    nu_spec.set_fractions([0., 0., 0., 0.])
    spec_other = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.semilogy(e_arr*1e-6,spec_other*1e6, 'b--', linewidth=1, label="U-238 n Capture")
    nu_spec.set_fractions(fractions)

    plt.legend(loc=8, prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Flux, nu/(MeV*day*cm^2)')
    plt.ylim(1.e15, 1.e18)
    #plt.title('Commercial Reactor Neutrino Flux')
    plt.savefig('plots/commercial_reactor_fission_vs_capture.png')
    fig0.clf()

    # Fractional Contribution
    fig0 = plt.figure(figsize=(4., 3.))
    fig0.patch.set_facecolor('white')
    plt.plot(e_arr*1e-6, spec_tot/spec_tot, "k-", linewidth=1)
    plt.plot(e_arr*1e-6, spec_fission/spec_tot, "r:", linewidth=1)
    plt.plot(e_arr*1e-6, spec_other/spec_tot, "b--", linewidth=1)
    #plt.xlabel('Neutrino Energy (MeV)')
    #plt.ylabel('Fractional Contribution')
    plt.xlim(0., 2.)
    plt.ylim(0., 1.1)
    plt.savefig('plots/commercial_reactor_fission_vs_capture_fraction.png')
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

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def plot_flux_xsec(nu_spec):
    # Ge
    Z = 32
    N = 72.64-32.

    e_arr = np.linspace(0., 1e7, 100000)

    # Plot neutrino flux
    fig, host = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(right=0.75)
    fig.patch.set_facecolor('white')

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.2))

    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    #make_patch_spines_invisible(par2)
    # Second, show the right spine.
    #par2.spines["right"].set_visible(True)

    lines = []

    # Spectrum in nu/(MeV*day*cm^2)
    spec_tot = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)*1e6
    p_spec, = host.plot(e_arr*1e-6,spec_tot, "k-", label="Neutrino Flux")
    lines.append(p_spec)

    xsec_1eV = total_XSec_cns(1., e_arr, Z, N)
    p_xsec_1, = par1.plot(e_arr*1e-6,xsec_1eV, "c-", label="Ethr=1eV")
    lines.append(p_xsec_1)
    prod_1eV = spec_tot*xsec_1eV
    p_prod_1, = par2.plot(e_arr*1e-6,spec_tot*xsec_1eV, ":", color=lighten_color('c', 0.7))

    xsec_10eV = total_XSec_cns(10., e_arr, Z, N)
    p_xsec_10, = par1.plot(e_arr*1e-6,xsec_10eV, "m-", label="Ethr=10eV")
    lines.append(p_xsec_10)
    prod_10eV = spec_tot*xsec_10eV
    p_prod_10, = par2.plot(e_arr*1e-6,spec_tot*xsec_10eV, ":", color=lighten_color('m', 0.7))

    xsec_50eV = total_XSec_cns(50., e_arr, Z, N)
    p_xsec_50, = par1.plot(e_arr*1e-6,xsec_50eV, "g-", label="Ethr=50eV")
    lines.append(p_xsec_50)
    prod_50eV = spec_tot*xsec_50eV
    p_prod_50, = par2.plot(e_arr*1e-6,spec_tot*xsec_50eV, ":", color=lighten_color('g', 0.7))

    xsec_100eV = total_XSec_cns(100., e_arr, Z, N)
    p_xsec_100, = par1.plot(e_arr*1e-6,xsec_100eV, "b-", label="Ethr=100eV")
    lines.append(p_xsec_100)
    prod_100eV = spec_tot*xsec_100eV
    p_prod_100, = par2.plot(e_arr*1e-6,spec_tot*xsec_100eV, ":", color=lighten_color('b', 0.7))

    xsec_200eV = total_XSec_cns(200., e_arr, Z, N)
    #p_xsec_200, = par1.plot(e_arr*1e-6,xsec_200eV, "r-", label="Ethr=200eV")
    #lines.append(p_xsec_200)
    prod_200eV = spec_tot*xsec_200eV
    p_prod_200, = par2.plot(e_arr*1e-6,spec_tot*xsec_200eV, ":", color=lighten_color('r', 0.7))

    '''host.set_xlim(0, 2)
    host.set_ylim(0, 2)
    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)'''

    host.set_xlabel("Neutrino Energy (MeV)")
    host.set_ylabel("Flux [nu/(MeV*day*cm^2)]")
    par1.set_ylabel("CEvNS XSec [cm^2]")
    par2.set_ylabel("Product [nu/(MeV*day)]")
    plt.text(9.8, 1.96*1.e-24, "1.e-40", bbox=dict(facecolor='white', alpha=1.0))
    plt.text(12., 1.96*1.e-24, "1.e-24", bbox=dict(facecolor='white', alpha=1.0))

    host.yaxis.label.set_color('k')
    par1.yaxis.label.set_color('k')
    par2.yaxis.label.set_color('k')

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors='k', **tkw)
    par1.tick_params(axis='y', colors='k', **tkw)
    par2.tick_params(axis='y', colors='k', **tkw)
    host.tick_params(axis='x', **tkw)

    host.legend(lines, [l.get_label() for l in lines], loc=(0.75, 0.1), prop={'size':9})
    #plt.legend(loc=4)

    plt.axvline(1.8, color='gray')

    plt.title('Neutrino Flux, CEvNS XSec, and product')
    plt.savefig('plots/flux_xsec_product.png')
    fig.clf()

    # Save results to file
    np.savetxt("plots/flux_xsec_product.txt",
               np.column_stack((1e-6*e_arr,spec_tot,
                                xsec_1eV, prod_1eV,
                                xsec_10eV, prod_10eV,
                                xsec_50eV, prod_50eV,
                                xsec_100eV, prod_100eV,
                                xsec_200eV, prod_200eV)),
               header="Neutrino Energies: MeV\n"+
               "Neutrino Flux: nu/(MeV*day*cm^2)\n"+
               "Cross Sections: cm^2\n"+
               "Product: nu/(MeV*day)\n"+
               "Neutrino Energy, Neutrino Flux, Ethr=1eV xsec, Ethr=1eV xsec,"+
               "Ethr=10eV xsec, Ethr=10eV xsec, Ethr=50eV xsec, Ethr=50eV xsec,"+
               "Ethr=100eV xsec, Ethr=100eV xsec, Ethr=200eV xsec, Ethr=200eV xsec")

def plot_lowe_spectra(nu_spec,
                      output_path_prefix="plots/",
                      Z=32, A=72.64, isotope_name='Ge',
                      site_title="Commerical Reactor",
                      enu_low=1.8e6):
    t_arr = np.logspace(0, 3, num=100)

    fig3 = plt.figure()
    fig3.patch.set_facecolor('white')
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, Z, A, nu_spec),'k-',label='Total',linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, Z, A, nu_spec, enu_min=enu_low),'m:',label='enu>%.1f MeV'%(enu_low/1.e6),linewidth=2)
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, Z, A, nu_spec, enu_max=enu_low),'c--',label='enu<%.1f MeV'%(enu_low/1.e6),linewidth=2)

    include_other = nu_spec.include_other
    nu_spec.include_other = False
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, Z, A, nu_spec),'r:',label='Fission',linewidth=2)
    nu_spec.include_other = include_other

    fractions = nu_spec.get_fractions()
    nu_spec.set_fractions([0., 0., 0., 0.])
    plt.loglog(t_arr,dsigmadT_cns_rate(t_arr, Z, A, nu_spec),'b--',label='U-238 n Capture',linewidth=2)
    nu_spec.set_fractions(fractions)
    
    plt.legend(prop={'size':11})
    plt.xlabel('Recoil Energy (eV)')
    plt.ylabel('Differential Event Rate (Events/kg/day/eV)')
    pre_label = "%s (A=%.1f)"%(isotope_name, A)
    plt.title(site_title+" "+pre_label+" Differential Rate")
    plt.ylim(1e-4, 1e1)
    plt.axvline(x=10.)
    plt.axvline(x=100.)
    plt.savefig(output_path_prefix+'low_e_spectra_'+isotope_name+'.png')
    fig3.clf()

def calc_lowe_fraction(nu_spec,
                       output_path_prefix="",
                       Z_arr=[32], A_arr=[72.64],
                       weights_arr=[1],
                       isotope_name='Ge',
                       site_title="Commerical Reactor"):
    enu_low = 1.8e6
    t_arr = np.logspace(-2, 4, num=200)

    N_arr = []
    for i in range(len(Z_arr)):
        N_arr.append(A_arr[i]-Z_arr[i])
    A_sum = 0
    weight_sum = 0
    for j in range(len(Z_arr)):
        A_sum += (Z_arr[j]+N_arr[j])*weights_arr[j]
        weight_sum += weights_arr[j]

    frac = total_cns_rate_an_compound(t_arr, enu_low, Z_arr, N_arr, weights_arr, nu_spec)/\
        total_cns_rate_an_compound(t_arr, 1e7, Z_arr, N_arr, weights_arr, nu_spec)
    frac[np.isnan(frac)] = 0
    frac[frac>1.] = 1.1 # It's a fraction, so it should be <1.0
    np.savetxt("%sthresh_vs_fraction_lt_1_8_%s.txt"%
               (output_path_prefix,isotope_name),
               np.column_stack((t_arr, frac)),
               header="# T (eV), Fraction")

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

    '''# Store flux to file, for use by statistical code
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_all.txt")
    store_reactor_flux_kev(nu_spec,
                           "flux_commercial_reactor_lt1800.txt",
                           0., 1800.)
    store_reactor_flux_kev(nu_spec,
                           "flux_commercial_reactor_gt1800.txt",
                           1800., 1.e4)

    nu_spec.include_other = False
    nu_spec.set_fractions(fractions)
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_fission.txt")
    nu_spec.include_other = True
    nu_spec.set_fractions([0., 0., 0., 0.])
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_u238n.txt")
    nu_spec.set_fractions(fractions)

    # Plot neutrino spectrum and CEvNS Rates
    plot_neutrino_spectrum_comparison(nu_spec, nu_spec_kopeikin, num_points=1000)
    plot_dsigmadT_cns_rate(nu_spec, num_points=100)
    plot_total_cns_rate(nu_spec, num_points=100)

    # Plot flux spectrum CEvNS xsec, and product
    plot_flux_xsec(nu_spec)

    # Compare fission and n capture neutrino spectra
    plot_neutrino_spectrum_other(nu_spec, num_points=1000)
    plot_lowe_spectra(nu_spec, "plots/",
                      Z=32, A=72.64, isotope_name='Ge')'''

    # Store fraction of neutrinos below 1.8 MeV for various threshold
    try:
        os.mkdir("fractions")
    except OSError:
        pass
    calc_lowe_fraction(nu_spec, "fractions/",
                       Z_arr=[14], A_arr=[28.08],
                       weights_arr=[1],
                       isotope_name='Si')
    calc_lowe_fraction(nu_spec, "fractions/",
                       Z_arr=[30], A_arr=[35.38+30.],
                       weights_arr=[1],
                       isotope_name='Zn')
    calc_lowe_fraction(nu_spec, "fractions/",
                       Z_arr=[32], A_arr=[72.64],
                       weights_arr=[1],
                       isotope_name='Ge')
    calc_lowe_fraction(nu_spec, "fractions/",
                       Z_arr=[13, 8], A_arr=[26.982, 16.0],
                       weights_arr=[2, 3],
                       isotope_name='Al2O3')
    calc_lowe_fraction(nu_spec, "fractions/",
                       Z_arr=[20, 74, 8], A_arr=[40.078, 183.84, 16.0],
                       weights_arr=[1, 1, 4],
                       isotope_name='CaWO4')

