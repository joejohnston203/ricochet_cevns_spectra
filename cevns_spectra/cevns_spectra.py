'''
Methods to calculate the CEvNS rate in various detectors

ALL ENERGIES IN eV
ALL DISTANCES IN cm
'''


import reactor_tools

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
from scipy.interpolate import UnivariateSpline


keVPerGeV       = 1e6             # [keV / GeV]
hbarc 	        = 0.19732705*keVPerGeV # [keV fm]
fmPercm		= 1.0e13	# [fm / cm]
sin2thetaW      = 0.2387        # Value at low energies
cLight	        = 3.0e8           # [m / s]
nAvogadro       = 6.022e23
Mn              = 0.93149410242 * keVPerGeV # keV
Mn_eV           = Mn*1e3          # eV
Gfermi	        = (1.1663787e-5 / (keVPerGeV**2.))*(hbarc/fmPercm) # [cm / keV]
Gfermi_cm_eV    = Gfermi*1e-3
Me              = 0.511*1e6       # eV
fineStruct      = 1.0/137.036
joulePereV      = 1.602e-19	  # [J / eV]
electronCharge  = joulePereV      # C
s_per_day       = 60.0*60.0*24.0

#roi_max = 1000 # Region of interest is detector threshold to 1000 eV
roi_max = 1000000 # Don't use a reasonable max- integrate all energies

def t_max(enu,M):
    if(enu>0):
        return min(enu/(1.0+M/(2*enu)),roi_max)
    #return enu/(1.0+M/(2*enu))

def e_min(T,M):
    return T/2.0 + 1/2.0 * np.sqrt(T**2+2.0*M*T)

#------------------------------------------------------------------------------
# Quantities to be used in the cross section
def weakFactors(Z, N, radiative_corrections=True):
    # Returns a 2-tuple of (GV, GA)
    if(radiative_corrections):
        sW = 0.23126
        rhoNC = 1.0086
        KvN = 0.9978
        lambdaUL = -0.0031
        lambdaDL = -0.0025
        lambdaDR = 7.5e-5
        lambdaUR = 0.5*lambdaDR
    else:
        sW = 0.2387
        rhoNC = 1.
        KvN = 1.
        lambdaUL = 0.
        lambdaDL = 0.
        lambdaDR = 0.
        lambdaUR = 0.5*lambdaDR
    gpV = rhoNC*(0.5-2.0*KvN*sW) + 2.0*lambdaUL + 2.0*lambdaUR + lambdaDL + lambdaDR
    gnV = -0.5*rhoNC + lambdaUL + lambdaUR + 2.0*lambdaDL + 2.0*lambdaDR
    gpA = 0.5*rhoNC + 2.0*lambdaUL + lambdaDL - 2.0*lambdaUR - lambdaDR
    gnA = - 0.5*rhoNC + 2.0*lambdaDL + lambdaUL - 2.0*lambdaDR - lambdaUR
    gV = (gpV*Z+gnV*N)
    gA = 0 # (spin up - spin down) ~ 0
    #print("gpV: %.2e, gnV: %.2e"%(gpV, gnV))
    return (gV, gA)

# T: Recoil energy in eV
# A: Z+N
# include_ff: Return 1. if this is False
# helm: Whether to use Helm (True) or Fermi (False) FF
def formFactor(T, A,
               include_ff=True,
               helm=True):
    if(not include_ff):
        return 1.

    xx = T/1.e6 # recoil energy in MeV
    s = 0.9 # Form factor parameterization together with s,a,c,r and q from Lewin/Smith 1996
    a = 0.52 # Form factor parameterization from Piekarewicz (2016)
    c = (1.23*A**(1.0/3.0) - 0.6)
    q = 6.92*1.e-3*np.sqrt(A*xx*1000.0)
    piqa = np.pi*q*a
    r = np.sqrt(c*c + (7.0/3.0)*np.pi*np.pi*a*a - 5.0*s*s)
    bessel = np.sin(q*r)/(q*r*q*r) - np.cos(q*r)/(q*r)

    if(helm):
        return (3.0*bessel/(q*r))*np.exp(q*q*s*s/2.0)
    else:
        return 3.0/(q*c*((q*c)**2 + piqa**2)) *\
            (piqa / np.sinh(piqa)) *\
            (piqa * np.sin(q*c) / np.tanh(piqa) - q*c*np.cos(q*c))

#------------------------------------------------------------------------------
# CEvNS Cross Section
# Units of cm^2/eV.
def dsigmadT_cns(T,enu,Z,N,
                 radiative_corrections=True,
                 form_factor=True,
                 helm_ff=True):
    M = Mn_eV*(N+Z) # eV
    (gV, gA) = weakFactors(Z, N, True)
    if(T>t_max(enu,M)):
        return 0.
    factor = Gfermi_cm_eV**2 * M / (2.*np.pi) # cm^2/eV
    shape = (gV+gA)**2 \
        + (gV-gA)**2*(1-T/enu)**2 \
        - (gV**2-gA**2)*M*T/enu**2
    ff = formFactor(T, Z+N,
                    form_factor, helm_ff)
    return shape*factor*ff**2
dsigmadT_cns = np.vectorize(dsigmadT_cns)

#------------------------------------------------------------------------------
# Rate in events per kg per day per keV
# Integral over reactor neutrino energies of
# reactor flux times the cross section
def dsigmadT_cns_rate(T, Z, N, nu_spec,
                      enu_min=None, enu_max=None,
                      form_factor=True, helm_ff=True):
    M = Mn_eV*(Z+N)
    targets_per_kg = nAvogadro/(Z+N)*1e3
    if enu_min is None:
        enu_min = e_min(T, M)
    else:
        enu_min = max(enu_min, e_min(T, M))
    if enu_max is None:
        enu_max = 1.e7
    res = spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*\
                     dsigmadT_cns(T, enu, Z, N,
                                  form_factor=form_factor, helm_ff=helm_ff),\
                     enu_min, enu_max)
    return res[0]*s_per_day*targets_per_kg
dsigmadT_cns_rate = np.vectorize(dsigmadT_cns_rate)

def dsigmadT_cns_rate_compound(T, Z_arr, N_arr, atom_arr, nu_spec,
                               enu_min=None, enu_max=None,
                               form_factor=True, helm_ff=True):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
    xsec_tot = 0*np.array(T).astype('float64')
    for i in range(len(Z_arr)):
        xsec_tot += atom_arr[i]*A_arr[i]/mass_tot*dsigmadT_cns_rate(T, Z_arr[i], N_arr[i], nu_spec, enu_min, enu_max, form_factor, helm_ff)
    return xsec_tot

#------------------------------------------------------------------------------
# Total CEvNS rate (integrated over recoil energies)

def total_XSec_cns(Tmin,enu,Z,N,
                   form_factor=True, helm_ff=True):
    M = Mn_eV*(N+Z) # eV
    Tmax = t_max(enu,M)
    if(Tmin>=Tmax):
        return 0.
    res = spint.quad(lambda T: dsigmadT_cns(T, enu, Z, N,
                                            form_factor=form_factor,
                                            helm_ff=helm_ff),
                     Tmin, Tmax)
    return res[0]
total_XSec_cns = np.vectorize(total_XSec_cns)

def total_XSec_cns_compound(Tmin, enu,
                            Z_arr, N_arr, atom_arr,
                            form_factor=True, helm_ff=True):
    atoms = 0
    for i in range(len(Z_arr)):
        atoms += atom_arr[i]
    xsec = 0
    for i in range(len(atom_arr)):
        xsec += total_XSec_cns(Tmin, enu, Z_arr[i], N_arr[i],
                               form_factor, helm_ff)*\
            atom_arr[i]/atoms
    return xsec

def total_XSec_cns_compound_in_bin(Tmin, enu_min, enu_max,
                                   Z_arr, N_arr, atom_arr,
                                   form_factor=True, helm_ff=True):
    return spint.quad(lambda enu:
                      total_XSec_cns_compound(Tmin, enu,
                                              Z_arr, N_arr, atom_arr,
                                              form_factor, helm_ff),
                      enu_min, enu_max)[0]


# CNS xsec analytically integrated over T
# Assumes FF = 1.
def total_XSec_cns_an(Tmin,enu,Z,N):
    M = Mn_eV*(Z+N)
    print("M: %.5e"%M)
    (gV, gA) = weakFactors(Z, N, True)

    Tmax = t_max(enu,M)
    if(Tmin>=Tmax):
        return 0.
    return Gfermi_cm_eV**2*M/2.0/np.pi*\
        (
            (gV+gA)**2*(Tmax-Tmin)
            -(gV-gA)**2*enu/3.*((1-Tmax/enu)**3-(1-Tmin/enu)**3)
            -(gV**2-gA**2)*M/enu**2*(Tmax**2/2.-Tmin**2/2.)
        )

#------------------------------------------------------------------------------
# CEvNS Yield

def cevns_yield(Tmin,enu_max,Z,N,nu_spec,enu_min=0.):
    ''' Return yield in cm^2 per fission'''
    M = Mn_eV*(Z+N)
    e_min_curr = max(enu_min, e_min(Tmin,M))
    return spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*\
                      total_XSec_cns(Tmin,enu,Z,N),\
                      e_min_curr,enu_max)[0]/\
                      nu_spec.nuFlux()
cevns_yield = np.vectorize(cevns_yield)

def cevns_yield_compound(Tmin, enu_max,
                         Z_arr, N_arr, atom_arr,
                         nu_spec, enu_min=0.,
                         per_gram=False):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    atoms = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
        atoms += atom_arr[i]
    yield_tot = 0*np.array(Tmin)

    if(per_gram):
        # Mass weighted average of cevns yield per gram
        for i in range(len(Z_arr)):
            yield_tot += atom_arr[i]*A_arr[i]/mass_tot*\
                (nAvogadro*cevns_yield(Tmin, enu_max, Z_arr[i], N_arr[i], nu_spec, enu_min)/A_arr[i])
        return yield_tot
    else:
        # Mass weighted average of cevns yield per atom
        for i in range(len(Z_arr)):
            yield_tot += atom_arr[i]*A_arr[i]/mass_tot*cevns_yield(Tmin, enu_max, Z_arr[i], N_arr[i], nu_spec, enu_min)
        return yield_tot

#------------------------------------------------------------------------------
# Total CEvNS xsec at a given neutrino energy

def total_cns_rate_an(Tmin,enu_max,Z,N,nu_spec,enu_min=0.):
    M = Mn_eV*(Z+N)
    e_min_curr = max(enu_min, e_min(Tmin,M))
    targets_per_kg = nAvogadro/(Z+N)*1e3
    return spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*\
                      total_XSec_cns(Tmin,enu,Z,N),\
                      e_min_curr,enu_max)[0]*\
                      s_per_day*targets_per_kg
total_cns_rate_an = np.vectorize(total_cns_rate_an)

def total_cns_rate_an_compound(Tmin, enu_max, Z_arr, N_arr, atom_arr, nu_spec, enu_min=0.):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
    rate_tot = 0*np.array(Tmin)
    for i in range(len(Z_arr)):
        rate_tot += atom_arr[i]*A_arr[i]/mass_tot*total_cns_rate_an(Tmin, enu_max, Z_arr[i], N_arr[i], nu_spec, enu_min)
    return rate_tot

# Slow: Only to be used to validate total_cns_rate_an or to
# calculate the rate up to some Tmax<inf
def cns_total_rate_integrated(Tmin, Z, N, nu_spec, Tmax=roi_max):
    if(Tmin>=Tmax):
        return 0.
    x = np.linspace(Tmin, Tmax, 1000.)
    y = dsigmadT_cns_rate(x, Z, N, nu_spec)
    spl = UnivariateSpline(x, y)
    res = spl.integral(Tmin, Tmax)
    return res
    '''res = spint.quad(lambda T_: dsigmadT_cns_rate(T_, Z, N, nu_spec),
                     Tmin, Tmax)
    return res[0]'''
cns_total_rate_integrated = np.vectorize(cns_total_rate_integrated)

def cns_total_rate_integrated_compound(Tmin, Z_arr, N_arr, atom_arr, nu_spec, Tmax=roi_max):
    A_arr = []
    for i in range(len(Z_arr)):
        A_arr.append(Z_arr[i]+N_arr[i])

    mass_tot = 0
    for i in range(len(Z_arr)):
        mass_tot += atom_arr[i]*A_arr[i]
    rate_tot = 0*np.array(Tmin)
    for i in range(len(Z_arr)):
        rate_tot += atom_arr[i]*A_arr[i]/mass_tot*cns_total_rate_integrated(Tmin, Z_arr[i], N_arr[i], nu_spec, Tmax)
    return rate_tot

#------------------------------------------------------------------------------
# Function to return 3-tuple containing the Z_arr, N_arr, and atom_arr
# Input: "Ge", "Zn", "Si", "CaWO4", or "Al2O3"
def get_atomic_arrs(target):
    if(target=="Ge"):
        Z_arr = [32]*5
        N_arr = [38, 40, 41, 42, 44]
        atom_arr = [0.2038, 0.2731, 0.0776, 0.3672, 0.0783]
    elif(target=="Zn"):
        Z_arr = [30]*5
        N_arr = [34, 36, 37, 38, 40]
        atom_arr = [48.268, 27.975, 4.102, 19.024, 0.631]
    elif(target=="Si"):
        Z_arr = [14]*3
        N_arr = [14, 15, 16]
        atom_arr = [0.92223, 0.04685, 0.03092]
    elif(target=="CaWO4"):
        Z_arr = [20, 20, 20, 20, 20, 20,
                 74, 74, 74, 74, 74,
                 8, 8, 8]
        N_arr = [20, 22, 23, 24, 26, 28,
                 106, 108, 109, 110, 112,
                 8, 9, 10]
        atom_arr = [0.96941, 0.00647, 0.00135, 0.02086, 0.00004, 0.00187,
                    0.0012, 0.265, 0.1431, 0.3064, 0.2843,
                    4.*0.9976, 4.*0.0004, 4.*0.0020]
    elif(target=="Al2O3"):
        Z_arr = [13, 8, 8, 8]
        N_arr = [14, 8, 9, 10]
        atom_arr = [2.*1.0, 3.*0.9976, 3.*0.0004, 3*0.0020]
    return (Z_arr, N_arr, atom_arr)

#------------------------------------------------------------------------------
# IBD XSec to 1st order in 1/M
# 10.1103/PhysRevD.60.053003
# Prefactor 0.0962(1) comes from neutron livetime
# from PDG-2018: tau_n=880.2(1.0) s
# enu in eV
def total_XSec_ibd(enu):
    Enu = enu*1.e-6
    Mn = 939.56542052  # Neutron mass in MeV
    Mp = 938.27208816  # Proton mass in MeV
    me = 0.51099895000 # Electron mass in MeV
    gA = 1.2723        # Axial coupling
    pref = 0.0962      # Prefactor
    
    DELTA = Mn-Mp
    M = 0.5*(Mn+Mp)
    a = 1./M
    b = 1.+DELTA/M
    c = DELTA+0.5*(DELTA*DELTA-me*me)/M-Enu
    Delta = b*b - 4.*a*c

    f = 1. # Vector coupling constant
    f2 = 3.706 # mu_p-mu_n, scaled to f value

    Ee1 = (-b + np.sqrt(Delta))/(2.*a) # Total positron energy at order 1
    if(Ee1*Ee1-me*me)<=0:
        return 0.
    Pe1 = np.sqrt(Ee1*Ee1-me*me) # Positron momentum at order 1

    cor_recoil = ( (gA*gA-f*f)*DELTA/M + (gA-f)*(gA-f)*(Ee1*(Ee1+DELTA)+Pe1*Pe1)/M/Ee1 ) / (f*f* + 3.*gA*gA)
    cor_weakmag = (-2.*f2*gA) * (Ee1 + DELTA + Pe1*Pe1/Ee1) / (f*f*+3.*gA*gA) / M

    total_cross_section = pref*Ee1*Pe1*(1.+cor_recoil+cor_weakmag)*1.e-42

    if(Enu<Ee1+DELTA or
       total_cross_section < 0.):
        return 0.
    else:
        return total_cross_section
    
total_XSec_ibd = np.vectorize(total_XSec_ibd)

def ibd_yield(nu_spec,enu_min=0.,enu_max=1.e7,
                    per_gram=False):
    ''' Return yield in cm^2 per fission'''
    ibd_yield = spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*
                      total_XSec_ibd(enu),
                      enu_min,enu_max)[0]/\
                      nu_spec.nuFlux()
    if(per_gram):
        h_frac_by_mass = 1./7.
        return ibd_yield*nAvogadro*h_frac_by_mass
    else:
        return ibd_yield
    return

def ibd_rate_per_kg_per_year(nu_spec, enu_min=0., enu_max=1.e7):
    # neutrinos/s/target nucleon
    rate_per_s_per_H = spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*
                                  total_XSec_ibd(enu),
                                  enu_min, enu_max)[0]
    h_frac_by_mass = 1./7.
    rate_per_s_per_g = rate_per_s_per_H*nAvogadro*h_frac_by_mass
    rate_per_year_per_kg = rate_per_s_per_g*1.e3*(365.*24.*60.*60.)
    return rate_per_year_per_kg

# IBD XSec to 0th order in 1/M
# 10.1103/PhysRevD.60.053003
# Prefactor 0.0962(1) comes from neutron livetime
# from PDG-2018: tau_n=880.2(1.0) s
def total_XSec_ibd_0th(enu):
    delta = 1.293332e6
    Ee0 = enu-delta
    me = 0.510999e6
    if(enu>delta and Ee0>me):
        pe0 = np.sqrt(Ee0**2-me**2)
        return 0.0962*(Ee0*pe0/(1.e6**2))*1.e-42
    else:
        return 0.
total_XSec_ibd_0th = np.vectorize(total_XSec_ibd_0th)

def ibd_yield_0th(nu_spec,enu_min=0.,enu_max=1.e7,
                    per_gram=False):
    ''' Return yield in cm^2 per fission'''
    ibd_yield = spint.quad(lambda enu: nu_spec.d_phi_d_enu_ev(enu)*
                      total_XSec_ibd_0th(enu),
                      enu_min,enu_max)[0]/\
                      nu_spec.nuFlux()
    if(per_gram):
        h_frac_by_mass = 1./7.
        return ibd_yield*nAvogadro*h_frac_by_mass
    else:
        return ibd_yield
    return 

#------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
