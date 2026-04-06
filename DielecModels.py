"""
# This module contains the function to calculate the effective complex dielectric model for multiphase dielectric mixture.
"""
import numpy as np
from scipy.optimize import root
from scipy.optimize import fsolve
from typing import Union


def EpsWater_Double(
        T: Union[float, np.ndarray], 
        f: Union[float, np.ndarray],
        S: Union[float, np.ndarray] = 0.0
        ) -> np.ndarray:
    """
    Computes the complex dielectric constant of water using the double-Debye model.

    Parameters:
        T (float): Temperature in degrees Celsius (valid range: 0 < T ≤ 30)
        f (float): Frequency in GHz (valid range: 0 < f ≤ 1000)
        S (float, optional): Salinity in PSU (valid range: 0 ≤ S ≤ 40), default is 0

    Returns:
        complex: Complex dielectric constant (real - j * imag)
    """
    if np.any((T <= 0) | (T > 30)):
        raise ValueError("Temperature must be in the range (0, 30] °C.")
    if np.any((f <= 0) | (f > 1000)):
        raise ValueError("Frequency must be in the range (0, 1000] GHz.")
    if np.any ((S < 0) | (S > 40)):
        raise ValueError("Salinity must be in the range [0, 40] PSU.")
    
    a = [
       0.46606917e-2, -0.26087876e-4, -0.63926782e-5, 
       0.63000075e1, 0.26242021e-2, -0.42984155e-2,
       0.34414691e-4, 0.17667420e-3, -0.20491560e-6,
       0.58366888e3, 0.12684992e3, 0.69227972e-4, 
       0.38957681e-6, 0.30742330e3, 0.12634992e3, 
       0.37245044e1, 0.92609781e-2, -0.26093754e-1
        ]
    
    sig_35 = 2.903602 + 8.607*10**-2*T + 4.738817e-4*T**2 + -2.991e-6*T**3 + 4.3041e-9*T**4
    P = S * ((37.5109 + 5.45216*S + 0.014409*S**2) / (1004.75 + 182.283*S + S**2))
    alpha0 = (6.9431 + 3.2841*S - 0.099486*S**2) / (84.85 + 69.024*S + S**2)
    alpha1 = 49.843  - 0.2276*S + 0.00198*S**2
    Q = 1 + ((alpha0 * (T - 15))/(T + alpha1))
    eps_0 = 8.854 * 10**-12
    eps_w0 = 87.85306 * np.exp(-0.00456992 * T - a[0] * S - a[1] * S**2 - a[2] * S * T)
    eps_w1 = a[3] * np.exp(-a[4] * T- a[5] * S - a[6] * S * T)
    tau1 = (a[7] + a[8] * S) * np.exp(a[9] / (T + a[10]))
    tau2 = (a[11] + a[12] * S) * np.exp(a[13] / (T + a[14]))
    epsInf = a[15] + a[16] * T + a[17] * S
    sigma = sig_35 * P * Q
 
    epsr = epsInf + (eps_w0 - eps_w1)/(1 + (2 * np.pi * f* tau1)**2) + (eps_w1 - epsInf)/(1 + (2 * np.pi * f * tau2)**2)
 
    epsi = 2 * np.pi * f * tau1 * (eps_w0 - eps_w1)/(1 + (2 * np.pi * f * tau1)**2) + 2 * np.pi * f * tau2 * (eps_w1 - epsInf) / (1 + (2 * np.pi * f * tau2)**2) + sigma/(2 * np.pi * f *1e9 *eps_0)
    
    return epsr - 1j*epsi



def EpsPureIce(T: Union[float, np.ndarray], fGHz: Union[float, np.ndarray]) -> np.ndarray:
    """
    Vectorized computation of the complex dielectric constant of pure ice 
    based on Mätzler and Wegmüller (1987).

    Parameters:
        T (float or np.ndarray): Temperature in Celsius (-40 < T < 0)
        fGHz (float or np.ndarray): Frequency in GHz (0.01 < f < 300)

    Returns:
        np.ndarray: Complex relative permittivity (eps = eps' - j * eps'')
    """
    T = np.asarray(T)
    fGHz = np.asarray(fGHz)

    if np.any((T < -40) | (T > 0)):
        raise ValueError("Temperature must be in the range (-40, 0) °C.")
    if np.any((fGHz < 0.01) | (fGHz > 300)):
        raise ValueError("Frequency must be in the range (0.01, 300) GHz.")

    T_k = T + 273.15
    theta = 300.0 / T_k - 1

    eps_r = 3.1884 + 9.1e-4 * T

    B1 = 0.0207
    B2 = 1.16e-11
    b = 335.0

    alpha_0 = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    exp_term = np.exp(b / T_k)
    beta_0 = (
        (B1 / T_k) * (exp_term / (exp_term - 1) ** 2)
        + B2 * fGHz**2
        + np.exp(-9.963 + 0.0372 * (T_k - 273.16))
    )

    eps_i = alpha_0 / fGHz + beta_0 * fGHz

    return eps_r - 1j * eps_i



def EpsDrySnow(
    Ros: Union[float, np.ndarray],
    eps_ice: Union[float, complex],
) -> np.ndarray:
    """
    Computes the complex dielectric constant of dry snow using Mätzler (2006).

    Parameters:
        Ros (float or np.ndarray): Snow density in kg/m³
        eps_ice (complex or np.ndarray): Dielectric constant of ice
        fGHz (float or np.ndarray): Frequency in GHz (not used but included for compatibility)

    Returns:
        np.ndarray: Complex dielectric constant of dry snow
    """
    roS = Ros / 1000       # convert to g/cm³
    vf = roS / 0.917       # volume fraction of ice

    # Relative permittivity (real part) depending on vf
    epsr_ds = np.where(
        vf <= 0.45,
        1 + 1.4667 * vf + 1.435 * vf**3,
        (1 + 0.4760 * vf)**3
    )

    # Imaginary part of dry snow permittivity
    denom = ((2 + vf) + np.real(eps_ice) * (1 - vf))**2
    epsi_ds = 9 * vf * np.abs(np.imag(eps_ice)) / denom

    return epsr_ds - 1j * epsi_ds


def brine_volume_fraction(S: float, T: float) -> float:
    """
    Calculate the brine volume fraction in sea ice using the Cox & Weeks (1983)
    linear approximation.

    Reference:
        Cox, G. F. N., & Weeks, W. F. (1983). Equations for determining the
        gas and brine volumes in sea ice samples. Journal of Glaciology,
        29(102), 306–316.

    Formula:
        v_b = 1e-3 * S_i * (-49.185 / T + 0.532)

    where S_i is ice salinity (PSU) and T is ice temperature (°C).

    Parameters
    ----------
    S  : float  Ice salinity (parts per thousand, PSU).
    T  : float  Ice temperature (°C).

    Returns
    -------
    float  Brine volume fraction (dimensionless, range ~ 0–1).
    """

    return 1e-3 * S * (-49.185 / T + 0.532)

def brine_permittivity(T: float, fGHz: float) -> complex:
    """
    Compute the complex dielectric permittivity of brine using the
    Stogryn & Desargant (1985) Debye relaxation model.

    Reference:
        Stogryn, A., & Desargant, G. J. (1985). The dielectric properties of
        brine in sea ice at microwave frequencies. IEEE Transactions on
        Antennas and Propagation, 33(5), 523–532.

    Sign convention:
        eps = eps_r - j * eps_i   (eps_i > 0 for a lossy medium)

        The Debye relaxation term and the ionic conductivity loss term both
        contribute negative imaginary parts, so eps.imag < 0 for all valid
        inputs, consistent with this convention.

    Model:
        eps = eps_inf
              + (eps_static - eps_inf) / (1 + j * 2*pi*f*tau)
              - j * sigma / (2*pi*f*1e9 * eps_0)

    The ionic conductivity uses two separate fits depending on temperature:
        T >= -22.9°C : sigma = -T * exp(0.5193 + 0.08755*T)
        T <  -22.9°C : sigma = -T * exp(1.0334 + 0.1100*T)

    Parameters
    ----------
    T : float  Brine temperature (°C); must be negative (below freezing).
    fGHz     : float  Electromagnetic frequency (GHz), e.g. 1.41 for L-band.

    Returns
    -------
    complex  Complex relative permittivity of brine in the form eps_r - j*eps_i.
    """
    eps_0 = 8.85419e-12   # Permittivity of free space (F/m)

    # Static dielectric constant (Stogryn & Desargant 1985, Eq. 4)
    eps_static = (939.66 - 19.068 * T) / (10.737 - T)

    # Debye relaxation time (ns) (Eq. 6)
    tau = (0.1099
           + 0.13603e-2 * T
           + 0.20894e-3 * T**2
           + 0.28167e-5 * T**3)

    # Ionic conductivity (S/m) — two-regime fit (Eq. 8)
    if T >= -22.9:
        sigma = -T * np.exp(0.5193 + 0.08755 * T)
    else:
        sigma = -T * np.exp(1.0334 + 0.1100 * T)

    # High-frequency dielectric constant (Eq. 5)
    eps_inf = (82.79 + 8.19 * T**2) / (15.68 + T**2)

    # Complex permittivity: Debye relaxation + ionic conductivity loss term.
    # tau (ns) * freq_ghz (GHz) = dimensionless — no extra scaling needed.
    # Both terms contribute negative imaginary parts → result is eps_r - j*eps_i.
    eps = (eps_inf
           + (eps_static - eps_inf) / (1 + fGHz * tau * 1j)
           - (sigma / (2 * np.pi * fGHz * 1e9 * eps_0)) * 1j)

    return eps



def depolarization_factors(shape: str = None, a: float = None, c: float = None) -> np.ndarray:
    """
    Compute the depolarization factors (Aa, Ab, Ac) based on the shape and eccentricity for Polder-van Santen/de Loor (1946) Model (PVSM)
    
    Input Parameters:
        shape (str): Shape of the particle ('sphere', 'disc', 'needle', 'prolate spheroid', 'oblate spheroid')
        a (float): Minor axis length (for spheroids)
        c (float): Major axis length (for spheroids)
        
    Returns:
        np.ndarray: Array of depolarization factors [Aa, Ab, Ac]
    """
    # Case: Known analytical shapes without eccentricity
    if a is None and c is None:
        if shape == 'sphere':
            return np.array([1/3, 1/3, 1/3])
        elif shape == 'circular disc':
            return np.array([0, 0, 1])
        elif shape == 'needle':
            return np.array([0.5, 0.5, 0])
        else: 
            raise ValueError("Unknown shape for predefined depolarization values.")
    
    # Missing one of the axis inputs
    if a is None or c is None:
        raise ValueError("Both 'a' and 'c' must be provided for spheroidal shapes.")
    # Find minor (a) and major (c) axes
    a_, c_ = min(a, c), max(a, c)
    e = np.sqrt(1- (a_ / c_)**2)   
    e = np.clip(e, 1e-10, 0.999999)
    if shape == 'prolate spheroid':  
            Ac = (1 - e**2) / (2*e**3) * (np.log((1 + e)/(1 - e)) - 2*e)
            Aa = Ab = (1 - Ac)/2
            return np.array([Aa, Ab, Ac])
    elif shape == 'oblate spheroid':
            Ac = (1/e**2) * (1 - (np.sqrt(1 - e**2)/e)*np.arcsin(e))
            Aa = Ab = (1 - Ac) / 2
            return np.array([Aa, Ab, Ac])
    else:
            raise ValueError("Unsupported shape for given eccentricity parameters.")



def polder_van_santen_loor(eps_h: Union[float, complex], eps_incl: Union[float, complex], vi: float, depol_factors: np.ndarray) -> complex:

    """
    Computes effective complex permittivity using the Polder-van Santen-Loor model.

    Parameters:
        eps_h: Host permittivity (real or complex)
        eps_incl: Inclusion permittivity (real or complex)
        vi: Volume fraction of inclusions (0-1)
        depol_factors: 1D array of depolarization factors [Aa, Ab, Ac]

    Returns:
    Effective complex permittivity as a complex number
    """
    if np.shape(depol_factors) != (3,):
        raise ValueError("depol_factors must be a 1D array of length 3.")
    if not (0<= vi <=1):
        raise ValueError("Volume fraction of inclusions needs to be between zero and one.")
    
    def pvsm(eps_m: Union[float, complex]) -> complex:
        term = 1 / (1 + depol_factors * (eps_incl / eps_m - 1))
        return eps_h +  vi / 3 * (eps_incl - eps_h) * np.sum(term) - eps_m

    def root_finder(vars):
        x, y = vars
        z = x + 1j * y
        fz = pvsm(z)
        return [fz.real, fz.imag] 
    i_guess = [(1 - vi) * eps_h.real + vi * eps_incl.real, (1 - vi) * eps_h.imag + vi * eps_incl.imag]
    eps_ms = fsolve(root_finder, i_guess)
    return eps_ms[0] + 1j * eps_ms[1]


def maxwell_garnett(eps_h: Union[float, complex], eps_incl: Union[float, complex], vi: float, depol_factors: np.ndarray) -> complex:
    """
    Computes effective complex permittivity using Maxwell-Garnett formula for random orientation of ellipsoidal inclusions.

    Parameters:
        eps_h: Host permittivity (real or complex)
        eps_incl: Inclusion permittivity (real or complex)
        vi: Volume fraction of inclusions (0-1)
        depol_factors: 1D array of depolarization factors [Aa, Ab, Ac]

    Returns:
        Effective complex permittivity as a complex number
    """
    if np.shape(depol_factors) != (3,):
        raise ValueError("depol_factors must be a 1D array of length 3.")
    if not (0<= vi <=1):
        raise ValueError("Volume fraction of inclusions needs to be between zero and one.")
    numer_term = (eps_incl - eps_h) / (eps_h + depol_factors * (eps_incl - eps_h))
    denom_term = (depol_factors * (eps_incl - eps_h)) / (eps_h + depol_factors * (eps_incl - eps_h))
    return eps_h + eps_h * (vi / 3 * np.sum(numer_term)) / (1 - vi / 3 * np.sum(denom_term))


def tinga_voss_blossey(eps_h: Union[float, complex], eps_i: Union[float, complex], vi: float, shape: str) -> complex:
    """
    Computes the effective complex permittivity using the Tinga-Voss-Blossey (TVB) formula
    for randomly dispersed confocal ellipsoidal inclusions of three types: circular disc,
    spherical, and needle-shaped inclusions.

    Parameters:
        eps_h (float or complex): Complex dielectric permittivity of the host material.
        eps_i (float or complex): Complex dielectric permittivity of the inclusion.
        vi (float): Volume fraction of the inclusion (0 <= vi <= 1).
        shape (str): Inclusion shape, one of {'circ_disc', 'sphere', 'needle'}.

    Returns:
        complex: Effective complex permittivity.
    """
    if not (0 <= vi <= 1):
        raise ValueError("Volume fraction 'vi' must be between 0 and 1.")

    if shape == 'circ_disc':
        return eps_h + vi / 3 * (eps_i - eps_h) * ((2 * eps_i * (1 - vi) + eps_h * (1 + 2*vi))) / (vi * eps_h + (1 - vi) * eps_i)

    elif shape == 'sphere':
        return eps_h + 3 * vi * eps_h * (eps_i - eps_h) / ((2 * eps_h + eps_i) - vi * (eps_i - eps_h))

    elif shape == 'needle':
        return eps_h + vi / 3 * (eps_i - eps_h) *((eps_h * (5 + vi) + (1 - vi) * eps_i) / (eps_h * (1 + vi) + eps_i * (1 - vi)))

    else:
        raise ValueError(f"Invalid shape: '{shape}'. Choose from 'circ_disc', 'sphere', or 'needle'.")
    