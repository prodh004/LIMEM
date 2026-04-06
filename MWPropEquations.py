import numpy as np
from typing import Union

def rt(epsUpper: Union[float, complex], epsLower: Union[float, complex], d: float, theta: float, fGHz: float) -> np.ndarray:
    """
    Return internal transmissivity and reflectivity of scatter free snow or lake ice
    
    Parameters:
        epsUpper: Complex dielectric constant of upper layer (air in case of snow, snow in case of ice)
        epsLower: Complex dielectric constant of the medium
        d: Thickness of snow/ice layer (meters)
        theta: Incident angle (degrees)
        fGHz: Frequency in GHz

    Returns:
        r: Internal reflectivity (currently placeholder, 0)
        t: Internal transmissivity
        theta_prop_deg: Propagation angle in snow/ice (degrees)
    """
    wavelength = 0.3 / fGHz  # in meters
    theta_rad = np.radians(theta)

    # Complex square root and arcsin for Snell's law
    theta_prop = np.arcsin(np.sqrt(epsUpper) * np.sin(theta_rad) / np.sqrt(epsLower))  # complex
    theta_prop_rad = theta_prop.real

    # Absorption coefficient
    alpha = (-4 * np.pi / wavelength) * np.sqrt(epsLower).imag

    # Compute transmissivity (allow for complex angle)
    t = np.exp(-d * alpha / np.cos(theta_prop_rad))

    # snow/ice reflectivity
    r = 0

    # Convert angle to degrees
    theta_prop_deg = theta_prop_rad * 180 / np.pi

    return r, t, theta_prop_deg

def InterfaceRefl(epsUpper: Union[float, complex], 
                epsLower: Union[float, complex],
                thetaUp: float,
                roughP: np.ndarray) -> float:
    """
    Compute the fresnel coefficients for reflectivity (H and V pol) at the interfaces of two media.
    
    Parameters:
        epsUpper: Complex dielectric constant of upper layer (medium 1)
        epsLower: Complex dielectric constant of lower layer (medium 2)
        thetaUp:  Refraction angle in upper layer (medium 1) in degrees
        roughP:   Roughness parameters at the interface (h, Q, nH, nV)
        
    Returns:
        rh: Fresnel reflectivity at H pol
        rv: Fresnel reflectivity at V pol
    
    """
    # Extract the roughness parameters
    h, Q, nH, nV = roughP
    
    # Convert angle to radians
    thetaUp_rad = np.radians(thetaUp)
    A = np.cos(thetaUp_rad)                               # cos(thetaUpper, 1)
    B = np.sqrt(1 - (epsUpper / epsLower) * (1 - A**2))   # cos(thetaLower, 2)
    
    # Fresnel specular reflectivities
    rFresh = abs((np.sqrt(epsUpper) * A - np.sqrt(epsLower) * B) / (np.sqrt(epsUpper) * A + np.sqrt(epsLower) * B))**2
    rFresv = abs((np.sqrt(epsUpper) * B - np.sqrt(epsLower) * A) / (np.sqrt(epsUpper) * B + np.sqrt(epsLower) * A))**2
    
    
    # Incorporate the impacts of roughness
    rh = (rFresh * (1 - Q) + rFresv * Q) * np.exp(-h * np.cos(thetaUp_rad)**nH)
    rv = (rFresv * (1 - Q) + rFresh * Q) * np.exp(-h * np.cos(thetaUp_rad)**nV)
    return rh, rv


def emitTb(nLayer: int, rj: np.ndarray, tj: np.ndarray, shj: np.ndarray, svj: np.ndarray, Tj: np.ndarray) -> np.ndarray:
    """
    Compute the matrix D, required to compute the TBs
    
    Parameters for n layers:
        nLayer: n number of snow and ice layers.
        rj: (n)-dim array of internal reflectivity of each layer (From bottom to top)  
        tj: (n)-dim array of internal transmissivity of each layer (From bottom to top)
        sj: (n+1)-dim array of Fresnel interface reflectivity (either H pol or V pol) (From bottom to top)
        Tj: (n+2)-dim Array of physical temperature of all media (in Kelvin) (Include underlying layer and also overlying sky radiation)
        
    Returns:
        D:  (n)-dim vector containing outgoing brightness temperature at the top of each layer
        Tb: brightness temperature at h- or v-pol 
    """  
    def TB_calculation(sj):
        # Check for any inconsistencies
        if ((len(rj) != len(tj)) or (len(sj) != len(rj) + 1) or (len(Tj) != len(sj) + 1) or (len(rj) != nLayer)):
            raise ValueError("Check the dimension of the input arrays.")

        ej = np.ones(nLayer) - rj - tj   # layers' emissivity values
        # Tj[0]:  Temperature of first bottom layer (i.e., lake water)
        # Tj[-1]: Downwelling sky radiation (i.e., T_sky)
        if nLayer == 1:
            neu1 = tj[0] * sj[0] * (rj[0] * (1 - sj[0]) * Tj[0] + tj[0] * (1 - sj[1]) * Tj[-1] + ej[0] * Tj[1])/(1 - rj[0] * sj[[0]])
            neu2 = tj[0] * (1 - sj[0]) * Tj[0] + rj[0] * (1 - sj[1]) * Tj[-1] + ej[0] * Tj[1]
            deno = 1 - rj[0] * sj[1] - tj[0]**2 * sj[0] * sj[1]/(1 - rj[0] * sj[0])
            D = np.array([(neu1 + neu2) / deno])
        else:
        # Calculation of matrix M1
            M1 = np.diag(np.multiply(rj, sj[:-1]))                  # Main diagonal
            sj_temp = np.ones(nLayer+1) - sj
            H = np.diag(np.multiply(tj[:-1], sj_temp[1:-1]), 1)     # Superdiagonal
            M1 += H

        # Calculation of matrix M2
            M2 = np.diag(np.multiply(tj, sj[1:]))                   # Main diagonal
            sj_temp = np.ones(nLayer+1) - sj
            H = np.diag(np.multiply(rj[1:], sj_temp[1:-1]), -1)     # Subdiagonal
            M2 += H

        # Calculation of matrix M3
            M3 = np.diag(np.multiply(tj, sj[:-1]))                  # Main diagonal
            sj_temp = np.ones(nLayer+1) - sj
            H = np.diag(np.multiply(rj[:-1], sj_temp[1:-1]), 1)     # Superdiagonal
            M3 += H

        # Calculation of matrix M4
            M4 = np.diag(np.multiply(rj, sj[1:]))                   # Main diagonal
            sj_temp = np.ones(nLayer+1) - sj
            H = np.diag(np.multiply(tj[1:], sj_temp[1:-1]), -1)     # Subdiagonal
            M4 += H

        # Calculation of matrix E and F
            E = np.multiply(ej, Tj[1:-1])
            F = E.copy()
            E[0]  += rj[0] * (1 - sj[0]) * Tj[0]
            E[-1] += tj[-1] * (1 - sj[-1]) * Tj[-1]
            F[0]  += tj[0] * (1 - sj[0]) * Tj[0]
            F[-1] += rj[-1] * (1 - sj[-1]) * Tj[-1]
            I = np.eye(nLayer)

            M5 = M3 @ (np.linalg.inv(I - M1) @ M2) + M4
            D = np.linalg.inv(I - M5) @ (M3 @ np.linalg.inv(I - M1) @ E + F)
        
        TBp = (1 - sj[-1]) * D[-1] + sj[-1] * Tj[-1]
        return TBp
    TBh = TB_calculation(shj)
    TBv = TB_calculation(svj)
    
    return TBh, TBv

def Ef_emissivity(nLayer: int, rj: np.ndarray, tj: np.ndarray, shj: np.ndarray, svj: np.ndarray, Tj: np.ndarray):
    
    """
    Calculate the effective emissivity
    
    Parameters:
        rj: Array of internal reflectivity of each layer (From bottom to top) 
        tj: Array of internal transmissivity of each layer (From bottom to top) 
        sj: Array of Fresnel interface reflectivity (H pol) (From bottom to top)
        Tj: Array of physical temperature of all media (in Kelvin) (Include underlying and overlying layer also)
        
    Returns:
        e: Effective emissivity for the composite layers at H or V pol 
    """
    Tj_100 = Tj.copy()
    Tj_100[-1] = 100
    Tbh100, Tbv100 = emitTb(nLayer, rj, tj, shj, svj, Tj_100)
    
    Tj_0 = Tj.copy()
    Tj_0[-1] = 0
    Tbh0, Tbv0 = emitTb(nLayer, rj, tj, shj, svj, Tj_0)
    
    eh_effective = 1 - (Tbh100 - Tbh0)/100
    ev_effective = 1 - (Tbv100 - Tbv0)/100
    
    return  eh_effective, ev_effective

