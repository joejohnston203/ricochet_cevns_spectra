from reactor_tools import NeutrinoSpectrum

import cevns_spectra
from cevns_spectra import dsigmadT_cns, dsigmadT_cns_rate, dsigmadT_cns_rate_compound, total_cns_rate_an, total_cns_rate_an_compound, cns_total_rate_integrated, cns_total_rate_integrated_compound, total_XSec_cns, total_XSec_cns_compound, cevns_yield_compound, ibd_yield, get_atomic_arrs

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

plt.rcParams.update({'font.size': 18})

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
    e_arr = np.linspace(0., 10.e6, num_points)
    fig0 = plt.figure()
    fig0.patch.set_facecolor('white')
    spec_tot = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_tot*1e6, "k-",linewidth=2, label="Total")
    
    include_other = nu_spec.include_other
    nu_spec.include_other = False
    spec_fission = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_fission*1e6, 'r:', linewidth=2, label="Fission")
    nu_spec.include_other = include_other
    
    fractions = nu_spec.get_fractions()
    nu_spec.set_fractions([0., 0., 0., 0.])
    spec_other = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)
    plt.plot(e_arr*1e-6,spec_other*1e6, 'b--', linewidth=2, label="Capture")
    nu_spec.set_fractions(fractions)

    plt.legend(prop={'size':11})
    plt.xlabel('Neutrino Energy (MeV)')
    plt.ylabel('Flux, nu/(MeV*day*cm^2)')
    plt.xlim(0., 8.)
    plt.ylim(0., 1.7e17)
    plt.grid()
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
    labels = ["Si", "Zn", "Ge", "Al2O3", "CaWO4"]
    lines = ['g-', 'b-', 'r-', 'c-.', 'm:']
    widths = [1,1,1,2,2]
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        plt.loglog(t_arr,dsigmadT_cns_rate_compound(t_arr, Z_arr, N_arr, atom_arr, nu_spec),lines[i],label=labels[i],linewidth=widths[i])
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
    labels = ["Si", "Zn", "Ge", "Al2O3", "CaWO4"]
    lines = ['g-', 'b-', 'r-', 'c-.', 'm:']
    widths = [1,1,1,2,2]
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        plt.loglog(t_arr,total_cns_rate_an_compound(t_arr, 1e7, Z_arr, N_arr, atom_arr, nu_spec),lines[i],label=labels[i],linewidth=widths[i])
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
    (Z_arr, N_arr, atom_arr) = get_atomic_arrs("Ge")

    e_arr = np.linspace(0., 1e7, 100000)

    # Plot neutrino flux
    fig, host = plt.subplots(figsize=(7, 4))
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15, top=0.95)
    fig.patch.set_facecolor('white')

    par1 = host.twinx()
    par2 = host.twinx()

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    #par2.spines["right"].set_position(("axes", 1.2))

    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    #make_patch_spines_invisible(par2)
    # Second, show the right spine.
    #par2.spines["right"].set_visible(True)

    lines = []

    # Spectrum in nu/(MeV*day*cm^2)
    spec_tot = s_per_day*nu_spec.d_phi_d_enu_ev(e_arr)*1e6
    p_spec, = host.plot(e_arr*1e-6,spec_tot, "k-", label=r"$\nu$ Flux", linewidth=2.)
    lines.append(p_spec)

    xsec_0eV = total_XSec_cns_compound(0., e_arr, Z_arr, N_arr, atom_arr)
    p_xsec_0, = par1.plot(e_arr*1e-6,xsec_0eV, color="#e41a1c", linestyle="-", label=r'T$_{Thr}$=0 eV')
    lines.append(p_xsec_0)
    prod_0eV = spec_tot*xsec_0eV
    p_prod_0, = par2.plot(e_arr*1e-6,spec_tot*xsec_0eV, color=lighten_color("#e41a1c", 1.0), linestyle="-")

    xsec_10eV = total_XSec_cns_compound(10., e_arr, Z_arr, N_arr, atom_arr)
    p_xsec_10, = par1.plot(e_arr*1e-6,xsec_10eV, color="#377eb8", linestyle="--", label='T$_{Thr}$=10 eV')
    lines.append(p_xsec_10)
    prod_10eV = spec_tot*xsec_10eV
    p_prod_10, = par2.plot(e_arr*1e-6,spec_tot*xsec_10eV, color=lighten_color("#377eb8", 1.0), linestyle="--")

    xsec_50eV = total_XSec_cns_compound(50., e_arr, Z_arr, N_arr, atom_arr)
    p_xsec_50, = par1.plot(e_arr*1e-6,xsec_50eV, color="#4daf4a", linestyle=":", label='T$_{Thr}$=50 eV')
    lines.append(p_xsec_50)
    prod_50eV = spec_tot*xsec_50eV
    p_prod_50, = par2.plot(e_arr*1e-6,spec_tot*xsec_50eV, color=lighten_color('#4daf4a', 1.0), linestyle=":")

    xsec_100eV = total_XSec_cns_compound(100., e_arr, Z_arr, N_arr, atom_arr)
    p_xsec_100, = par1.plot(e_arr*1e-6,xsec_100eV, color="#984ea3", linestyle="-.", label='T$_{Thr}$=100 eV')
    lines.append(p_xsec_100)
    prod_100eV = spec_tot*xsec_100eV
    p_prod_100, = par2.plot(e_arr*1e-6,spec_tot*xsec_100eV, color=lighten_color("#984ea3", 1.0), linestyle="-.")

    host.set_xlim(0, 8)
    '''host.set_xlim(0, 2)
    host.set_ylim(0, 2)
    par1.set_ylim(0, 4)
    par2.set_ylim(1, 65)'''

    host.set_xlabel("Neutrino Energy (MeV)")
    host.set_ylabel("Arbitrary Units")
    #par1.set_ylabel("CEvNS XSec [cm^2]")
    #par2.set_ylabel("Product [nu/(MeV*day)]")
    #plt.text(9.8, 1.96*1.e-24, "1.e-40", bbox=dict(facecolor='white', alpha=1.0))
    #plt.text(12., 1.96*1.e-24, "1.e-24", bbox=dict(facecolor='white', alpha=1.0))

    #host.yaxis.label.set_color('k')
    #par1.yaxis.label.set_color('k')
    #par2.yaxis.label.set_color('k')

    tkw = dict(size=4, width=1.5)
    #host.tick_params(axis='y', colors='k', **tkw)
    #par1.tick_params(axis='y', colors='k', **tkw)
    #par2.tick_params(axis='y', colors='k', **tkw)
    host.tick_params(axis='x', **tkw)
    host.tick_params(axis='y',
                     which='both',      # both major and minor ticks are affected
                     left=True,      # ticks along the bottom edge are off
                     right=False,         # ticks along the top edge are off
                     labelleft=False, # labels along the bottom edge are off
                     labelcolor='white')
    #host.get_yaxis().set_visible(False)
    par1.get_yaxis().set_visible(False)
    par2.get_yaxis().set_visible(False)

    host.legend(lines, [l.get_label() for l in lines], loc=(0.585, 0.5), prop={'size':14}, framealpha=0.9)
    #plt.legend(loc=4)

    plt.axvline(1.8, color='k')
    host.set_ylim(bottom=0)
    par1.set_ylim(bottom=0)
    par2.set_ylim(bottom=0)

    plt.title('')
    host.grid()
    plt.savefig('plots/flux_xsec_product.pdf', bbox_inches='tight')
    fig.clf()

    # Save results to file
    np.savetxt("plots/flux_xsec_product.txt",
               np.column_stack((1e-6*e_arr,spec_tot,
                                xsec_0eV, prod_0eV,
                                xsec_10eV, prod_10eV,
                                xsec_50eV, prod_50eV,
                                xsec_100eV, prod_100eV)),
               header="Neutrino Energies: MeV\n"+
               "Neutrino Flux: nu/(MeV*day*cm^2)\n"+
               "Cross Sections: cm^2\n"+
               "Product: nu/(MeV*day)\n"+
               "Neutrino Energy, Neutrino Flux, Ethr=0eV xsec, Ethr=0eV xsec,"+
               "Ethr=10eV xsec, Ethr=10eV xsec, Ethr=50eV xsec, Ethr=50eV xsec,"+
               "Ethr=100eV xsec, Ethr=100eV xsec, Ethr=200eV xsec, Ethr=200eV xsec")

def print_cevns_xsec(nu_spec):
    print("CEvNS Yields per Average Atom (10^-43 cm^2/fission):")
    labels = ["Al2O3", "Si", "Zn", "Ge", "CaWO4"]
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        print("\t%s 10  eV: %.3e"%
              (labels[i],
               (cevns_yield_compound(10., 1.e8, Z_arr, N_arr, atom_arr, nu_spec)/1.e-43)))
        print("\t%s 100 eV: %.3e"%
              (labels[i],
               (cevns_yield_compound(100., 1.e8, Z_arr, N_arr, atom_arr, nu_spec)/1.e-43)))
    print("IBD Yield per Nucleon (10^-43 cm^2/fission): %.3e"%(ibd_yield(nu_spec)/1.e-43))

    print("")
    print("CEvNS Yields per Gram (10^-20 cm^2/fission/g):")
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        print("\t%s 10  eV: %.3e"%
              (labels[i],
               (cevns_yield_compound(10., 1.e8, Z_arr, N_arr, atom_arr, nu_spec, per_gram=True)/1.e-43)))
        print("\t%s 100 eV: %.3e"%
              (labels[i],
               (cevns_yield_compound(100., 1.e8, Z_arr, N_arr, atom_arr, nu_spec, per_gram=True)/1.e-43)))
    print("IBD Yield (10^-20 cm^2/fission/g): %.3e"%(ibd_yield(nu_spec, per_gram=True)/1.e-20))


def plot_lowe_spectra(nu_spec,
                      output_path_prefix="plots/",
                      Z=32, A=74, isotope_name='Ge-74',
                      site_title="Commerical Reactor",
                      enu_low=1.8e6,
                      lt18=False, u238n=False,
                      neutron_shapes=True,
                      neutron_levels=True):
    t_arr = np.logspace(0, 3, num=100)

    fig3 = plt.figure(figsize=[8., 4.8])
    fig3.patch.set_facecolor('white')
    plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate(t_arr, Z, A-Z, nu_spec)*1.e3,'k-',label='CEvNS Total',linewidth=1.)

    if(lt18):
        plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate(t_arr, Z, A-Z, nu_spec, enu_min=enu_low)*1.e3, color="#e41a1c", linestyle="--", label='CEvNS %s>%.1f MeV'%(r'E$_\nu$', enu_low/1.e6), linewidth=2.)
        plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate(t_arr, Z, A-Z, nu_spec, enu_max=enu_low)*1.e3, color="#377eb8", linestyle=":", label='CEvNS %s<%.1f MeV'%(r'E$_\nu$', enu_low/1.e6), linewidth=2.)

    if(u238n):
        include_other = nu_spec.include_other
        nu_spec.include_other = False
        plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate(t_arr, Z, A-Z, nu_spec)*1.e3, color="#e41a1c", linestyle="--", label='Fission', linewidth=2.)
        nu_spec.include_other = include_other

        fractions = nu_spec.get_fractions()
        nu_spec.set_fractions([0., 0., 0., 0.])
        plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate(t_arr, Z, A-Z, nu_spec)*1.e3, color="#377eb8", linestyle=":", label='U-238 n', linewidth=2.)
        nu_spec.set_fractions(fractions)

    def n_back(T_keV, tau_1, tau_2, fac_2,
               scale, norm=1., n_xsec=0.081):
        # Returns rate in evts/kg/day/keV
        rescale = n_xsec/0.081
        return 1.e-3*norm*rescale*scale*\
            (np.exp(-tau_1*T_keV)+fac_2*np.exp(-tau_2*T_keV))
    n_back = np.vectorize(n_back)

    n_cons_int = spint.quad(n_back, 0.01, 0.9,
                            args=(0.081*1.e3,
                                  0.0086*1.e3, 0.23/0.38,
                                  1.))[0]
    n_cons_scale = 1/n_cons_int
    if(neutron_levels):
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                 0.081*1.e3,
                                 0.0086*1.e3, 0.23/0.38,
                                 n_cons_scale,
                                 1000.*1.e-3)*1.e3,
                   ':', color='darkorange', label="B=100., Cons")
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                 0.081*1.e3,
                                 0.0086*1.e3, 0.23/0.38,
                                 n_cons_scale,
                                 100.*1.e-3)*1.e3,
                   ':', color='orange', label="B=10., Cons")

    if(neutron_shapes or neutron_levels):
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                       0.081*1.e3,
                                       0.0086*1.e3, 0.23/0.38,
                                       n_cons_scale,
                                       1000.*1.e-3)*1.e3,
                   color="#4daf4a", linestyle='-.',
                   linewidth=1.,
                   label="B (Conservative)")

    if(neutron_shapes):
        n_med_int = spint.quad(n_back, 0.01, 0.9,
                               args=(0.004*1.e3,
                                     0.0005*1.e3, 0.64,
                                     1.))[0]
        n_med_scale = 1./n_med_int
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                       0.004*1.e3,
                                       0.0005*1.e3, 0.64,
                                       n_med_scale,
                                       100.*1.e-3)*1.e3,
                   color="#984ea3", linestyle='-.',
                   linewidth=1.5,
                   label="B (Medium)")

        n_opt_int = spint.quad(n_back, 0.01, 0.9,
                               args=(0.0004*1.e3,
                                     0.00006*1.e3, 0.64,
                                     1.))[0]
        n_opt_scale = 1./n_opt_int
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                       0.0004*1.e3,
                                       0.00006*1.e3, 0.64,
                                       n_opt_scale,
                                       10.*1.e-3)*1.e3,
                   color="#ff7f00", linestyle='-.',
                   linewidth=2.,
                   label="B (Optimistic)")

    ax = plt.gca()
    plt.subplots_adjust(right=0.8)
    ax.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, prop={'size':14})
    plt.xlabel('Recoil Energy (keV)')
    plt.ylabel('Differential Event Rate (dru)')
    pre_label = "%s (A=%.1f)"%(isotope_name, A)
    #plt.title(site_title+" "+pre_label+" Differential Rates")
    plt.xlim(1.e-3, 1.0)
    plt.ylim(1e-1, 1.e5)
    plt.axvline(x=1.e-3, color="k")
    plt.axvline(x=10.e-3, color="k")
    plt.axvline(x=50.e-3, color="k")
    plt.grid()
    filename = output_path_prefix+'lowe_'+isotope_name
    if(lt18):
        filename += "_lt18"
    if(u238n):
        filename += "_u238n"
    if(neutron_shapes):
        filename += "_nShapes"
    if(neutron_levels):
        filename += "_nLevels"
    filename += '.pdf'
    plt.savefig(filename,  bbox_inches='tight')
    fig3.clf()

def plot_lowe_spectra_isotopes(nu_spec,
                               output_path_prefix="plots/",
                               Z_arrs=[[32]], A_arrs=[[72.64]], weights=[[1]],
                               isotope_names=['Ge'],
                               site_title="Commerical Reactor",
                               enu_low=1.8e6,
                               lt18=False, u238n=False,
                               plot_total=False,
                               plot_low=True,
                               plot_high=False,
                               plot_back=True):
    t_arr = np.logspace(0, 3, num=100)

    fig3 = plt.figure()
    fig3.patch.set_facecolor('white')

    high_colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"]
    low_colors = ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"]

    for i in range(len(Z_arrs)):
        Z_arr = np.array(Z_arrs[i])
        A_arr = np.array(A_arrs[i])
        weight_arr = weights[i]
        name = isotope_names[i]
        if(plot_total):
            plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate_compound(t_arr, Z_arr, A_arr-Z_arr, weight_arr, nu_spec)*1.e3,'-.', color=high_colors[i], label='%s Tot'%name,linewidth=float(i)/2.+0.5)

        if(lt18):
            if(plot_high):
                plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate_compound(t_arr, Z_arr, A_arr-Z_arr, weight_arr, nu_spec, enu_min=enu_low)*1.e3, color=high_colors[i], linestyle="--", label='%s enu>%.1f MeV'%(name, enu_low/1.e6), linewidth=float(i)/2.+0.5)
            if(plot_low):
                plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate_compound(t_arr, Z_arr, A_arr-Z_arr, weight_arr, nu_spec, enu_max=enu_low)*1.e3, color=low_colors[i], linestyle="-", label='%s enu<%.1f MeV'%(name, enu_low/1.e6), linewidth=float(i)/2.+0.5)

        if(u238n):
            if(plot_high):
                include_other = nu_spec.include_other
                nu_spec.include_other = False
                plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate_compound(t_arr, Z_arr, A_arr-Z_arr, weight_arr, nu_spec)*1.e3, color=high_colors[i], linestyle="--", label='%s Fission'%name, linewidth=float(i)/2.+0.5)
                nu_spec.include_other = include_other

            if(plot_low):
                fractions = nu_spec.get_fractions()
                nu_spec.set_fractions([0., 0., 0., 0.])
                plt.loglog(t_arr*1.e-3,dsigmadT_cns_rate_compound(t_arr, Z_arr, A_arr-Z_arr, weight_arr, nu_spec)*1.e3, color=low_colors[i], linestyle="-", label='%s U-238 n'%name, linewidth=float(i)/2.+0.5)
                nu_spec.set_fractions(fractions)

    if(plot_back):
        def n_back(T_keV, tau_1, tau_2, fac_2,
                   scale, norm=1., n_xsec=0.081):
            # Returns rate in evts/kg/day/keV
            rescale = n_xsec/0.081
            return 1.e-3*norm*rescale*scale*\
                (np.exp(-tau_1*T_keV)+fac_2*np.exp(-tau_2*T_keV))
        n_back = np.vectorize(n_back)

        n_cons_int = spint.quad(n_back, 0.01, 0.9,
                            args=(0.081*1.e3,
                                  0.0086*1.e3, 0.23/0.38,
                                  1.))[0]
        n_cons_scale = 1/n_cons_int
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                0.081*1.e3,
                                   0.0086*1.e3, 0.23/0.38,
                                   n_cons_scale,
                                   10.*1.e-3)*1.e3,
               color="lightgrey", linestyle=':',
               linewidth=2.,
               label="B=10, Conservative")
        n_med_int = spint.quad(n_back, 0.01, 0.9,
                               args=(0.004*1.e3,
                                     0.0005*1.e3, 0.64,
                                     1.))[0]
        n_med_scale = 1./n_med_int
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                       0.004*1.e3,
                                       0.0005*1.e3, 0.64,
                                       n_med_scale,
                                       10.*1.e-3)*1.e3,
                   color="grey", linestyle=':',
                   linewidth=1.5,
                   label="B=10, Medium")

        n_opt_int = spint.quad(n_back, 0.01, 0.9,
                               args=(0.0004*1.e3,
                                     0.00006*1.e3, 0.64,
                                    1.))[0]
        n_opt_scale = 1./n_opt_int
        plt.loglog(t_arr*1.e-3, n_back(t_arr*1.e-3,
                                       0.0004*1.e3,
                                       0.00006*1.e3, 0.64,
                                       n_opt_scale,
                                       10.*1.e-3)*1.e3,
                   color="k", linestyle=':',
                   linewidth=1.,
                   label="B=10, Optimistic")

    plt.legend(prop={'size':9})
    plt.xlabel('Recoil Energy (keV)')
    plt.ylabel('Differential Event Rate (dru)')
    plt.ylim(1e-1, 1.e4)
    plt.axvline(x=1.e-3, color="k")
    plt.axvline(x=10.e-3, color="k")
    plt.axvline(x=50.e-3, color="k")
    title="CEvNS Spectrum for "
    if(lt18):
        title += "Enu<1.8/Enu>1.8 MeV"
        if(u238n):
            title += " and "
    if(u238n):
        title += "U-238 n Capture/Fission"
    plt.title(title)
    filename = output_path_prefix+'lowe'
    for name in isotope_names:
        filename += '_'+name
    if(lt18):
        filename += "_lt18"
    if(u238n):
        filename += "_u238n"
    if(plot_total):
        filename += "_tot"
    if(plot_low):
        filename += "_low"
    if(plot_high):
        filename += "_high"
    if(plot_back):
        filename += "_back"
    filename += '.png'
    plt.savefig(filename)
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
    flux_lt_18 = spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu),
                            0., 1.8e6)[0]
    flux_tot = spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu),
                            0., 20.e6)[0]
    print("Isotope: %s"%isotope_name)
    print("\tFraction of flux <1.8 MeV: %.4f"%(flux_lt_18/flux_tot))
    for thresh in [0.0001, 1., 10., 50.]:
        frac = total_cns_rate_an_compound(thresh, enu_low, Z_arr, N_arr, weights_arr, nu_spec)/\
        total_cns_rate_an_compound(thresh, 1e7, Z_arr, N_arr, weights_arr, nu_spec)
        print("\tT=%.2e, Frac=%.5f"%(thresh,frac))

if __name__ == "__main__":
    try:
        os.mkdir('plots')
    except OSError as e:
        pass

    # The averaged spectrum is stored in U-235
    fractions = [1.0, 0.0, 0.0, 0.0]

    # Chooz reactors are at 102 m and 72, each 4.25 GW
    # With both on, this is equivalent to 58.82 m from one 4.25 GW reactor
    power = 4250
    distance = 5882 # cm

    # The stored spectra are in neutrinos/MeV/s for a 4250 MW reactor
    # reactor_tools will multiply by: power*200./2.602176565e-19
    # We need to rescale to undo this
    scale = 1./(power/200.0/1.602176565e-19)

    nu_spec = NeutrinoSpectrum(distance, power, False, *fractions,
                               include_other=True)
    nu_spec.initialize_d_r_d_enu("u235", "root",
                                 "../../../final_spectra/sum_U_Pu_20gspt_Tengblad-TAGSnew-ENSDF2020-Qbeta5br_FERMI.screen.QED.aW.root",
                                 "nsim_Fission_avg",
                                 scale=scale)
    nu_spec.initialize_d_r_d_enu("u238", "zero")
    nu_spec.initialize_d_r_d_enu("pu239", "zero")
    nu_spec.initialize_d_r_d_enu("pu241", "zero")
    nu_spec.initialize_d_r_d_enu("other", "root",
                                 "../../../final_spectra/sum_U_Pu_20gspt_Tengblad-TAGSnew-ENSDF2020-Qbeta5br_FERMI.screen.QED.aW.root",
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

    # Mueller spectra
    nu_spec_mueller = NeutrinoSpectrum(nu_spec.distance, nu_spec.power, True,
                                       0.564, 0.076, 0.304, 0.056) # Daya Bay Numbers (10.1103/PhysRevD.100.052004)
    nu_spec_mueller.initialize_d_r_d_enu("u235", "txt",
                                         "../../data/huber/U235-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("u238", "mueller")
    nu_spec_mueller.initialize_d_r_d_enu("pu239", "txt",
                                         "../../data/huber/Pu239-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("pu241", "txt",
                                         "../../data/huber/Pu241-anti-neutrino-flux-250keV.dat")
    nu_spec_mueller.initialize_d_r_d_enu("other", "mueller")

    # Store flux to file, for use by statistical code
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
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_fission_lt1800.txt",
                           0., 1800.)
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_fission_gt1800.txt",
                           1800., 1.e4)
    nu_spec.include_other = True
    nu_spec.set_fractions([0., 0., 0., 0.])
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_u238n.txt")
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_u238n_lt1800.txt",
                           0., 1800.)
    store_reactor_flux_kev(nu_spec, "flux_commercial_reactor_u238n_gt1800.txt",
                           1800., 1.e4)
    nu_spec.set_fractions(fractions)

    # Plot neutrino spectrum and CEvNS Rates
    plot_neutrino_spectrum_comparison(nu_spec, nu_spec_kopeikin, num_points=1000)
    plot_dsigmadT_cns_rate(nu_spec, num_points=100)
    plot_total_cns_rate(nu_spec, num_points=100)
    # Plot flux spectrum CEvNS xsec, and product
    plot_flux_xsec(nu_spec)
    print_cevns_xsec(nu_spec)
    print("IBD Yield Summation: %.3e [cm^2/fission]"%ibd_yield(nu_spec, per_gram=False))
    print("IBD Yield Mueller: %.3e [cm^2/fission]"%ibd_yield(nu_spec_mueller, per_gram=False))

    # Compare fission and n capture neutrino spectra
    plot_neutrino_spectrum_other(nu_spec, num_points=1000)

    plot_lowe_spectra(nu_spec, "plots/",
                      Z=32, A=74, isotope_name='Ge-74',
                      lt18=True, neutron_levels=False)
    plot_lowe_spectra(nu_spec, "plots/",
                      Z=32, A=74, isotope_name='Ge-74',
                      u238n=True, neutron_levels=False)

    labels = ["CaWO4",
             "Ge", "Zn", "Si",
             "Al2O3"]
    Z_arrs = list()
    A_arrs = list()
    weight_arrs = list()
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        Z_arrs.append(Z_arr)
        A_arr = np.array(Z_arr)+np.array(N_arr)
        A_arrs.append(A_arr)
        weight_arrs.append(atom_arr)
    plot_lowe_spectra_isotopes(nu_spec, "plots/",
                               Z_arrs=Z_arrs,
                               A_arrs=A_arrs,
                               weights=weight_arrs,
                               isotope_names=labels,
                               lt18=True)
    plot_lowe_spectra_isotopes(nu_spec, "plots/",
                               Z_arrs=Z_arrs,
                               A_arrs=A_arrs,
                               weights=weight_arrs,
                               isotope_names=labels,
                               lt18=True, plot_high=True,
                               plot_back=False)
    plot_lowe_spectra_isotopes(nu_spec, "plots/",
                               Z_arrs=Z_arrs,
                               A_arrs=A_arrs,
                               weights=weight_arrs,
                               isotope_names=labels,
                               u238n=True)
    plot_lowe_spectra_isotopes(nu_spec, "plots/",
                               Z_arrs=Z_arrs,
                               A_arrs=A_arrs,
                               weights=weight_arrs,
                               isotope_names=labels,
                               u238n=True, plot_high=True,
                               plot_back=False)

    # Store fraction of neutrinos below 1.8 MeV for various threshold
    try:
        os.mkdir("fractions")
    except OSError:
        pass
    labels = ["CaWO4",
             "Ge", "Zn", "Si",
             "Al2O3"]
    for i in range(len(labels)):
        (Z_arr, N_arr, atom_arr) = get_atomic_arrs(labels[i])
        A_arr = np.array(Z_arr)+np.array(N_arr)
        calc_lowe_fraction(nu_spec, "fractions/",
                           Z_arr=Z_arr, A_arr=A_arr,
                           weights_arr=atom_arr,
                           isotope_name=labels[i])
