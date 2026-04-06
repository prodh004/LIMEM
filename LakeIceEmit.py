import numpy as np
from typing import Union
import DielecModels as dm
import MWPropEquations as two_S

def LakeIceEmit(WaterParams: dict, 
                IceParams: dict, 
                SnowParams: dict, 
                RoughnessParams: dict = {}, 
                RadioParams: dict = {}, 
                MixingParams: dict = {}, 
                TSky: float = 5 )-> np.ndarray:
    """
    Compute the TBs using two stream emission model for water-ice-snow continuum
    
    Parameters:
        
        WaterParams:
            TWater: Lake Water Temperature in celsius, 
            SWater: Water salinity in PSU
        
        IceParams: 
            TIce: Temperature in celsius, 
            dIce: Thickness in meter, 
            WIce: ice water fraction,
            SIce: ice density in PSU, 
            AIce: ice air fraction
        
        SnowParams: Parameters for snow as a dictionary 
            TSnow: Temperature in celsius
            RoS:   Density in kg/m3,
            dSnow: Depth in meter, 
            WSnow: Water fraction
        
        RoushnessParams: [Q, h, n_x, n_y] 
            rough_wi: water-ice interface, 
            rough_is: ice-snow interface, 
            rough_sa: snow-air interface
        
        RadioParams: Parameters for radiometer as a dictionary 
            f: frequency in GHz,
            angle: angle of incidence in degree

        MixParams: 
            Depolarization parameters:
                water: [Axw, Ayw, Azw]
                air:   [Axa, Aya, Aza]
            Method:  
                'pvsm':  Polder, D. and J. H. van Santen model (default)
                'mg'  :  Maxwell Garnett 

        Tsky: Downwelling sky radiation in celsius (default value is 5 kelvin or -268.15 celsius)
        
    Returns:
        Tbh: Brightness temperature in Kelvin at H pol 
        Tbv: Brightness temperature in Kelvin at V pol 
    """

    def DielcMixing(eps_host: Union[float, complex],
               eps_inc1: Union[float, complex],
               vWater: float = 0,   
               vAir: float = 0, 
               method: str = 'pvsm',
               depolParamWater = np.array([0.005, 0.4975, 0.4975]),
               depolParamAir = np.array([1/3, 1/3, 1/3])):
        """
        Computes the effective dielectric constant of a mixture using specified mixing model.

        Parameters:
            eps_host (float or complex): Dielectric constant of the host medium.
            eps_inc1 (float or complex): Dielectric constant of the first inclusion (e.g., water).
            vWater (float): Volume fraction of water inclusion.
            vAir (float, optional): Volume fraction of air inclusion. Defaults to 0.
            method (str, optional): Mixing model to use ('pvsm' for Polder-van Santen, 'mgm' for Maxwell Garnett, etc.).
            depolParamWater (np.ndarray): Depolarization factors for water inclusion (default is oblate spheroid).
            depolParamAir (np.ndarray): Depolarization factors for air inclusion (default is isotropic sphere).

        Returns:
            complex: Effective complex dielectric constant of the mixture.
        """
        if method == 'maxwell':
            eps_eff1 = dm.maxwell_garnett(eps_host, eps_inc1, vWater, depolParamWater)
            eps_eff2 = dm.maxwell_garnett(eps_eff1, 1, vAir, depolParamAir)
            return eps_eff2
        
        elif method == 'pvsm':
            eps_eff1 = dm.polder_van_santen_loor(eps_host, eps_inc1, vWater, depolParamWater)
            eps_eff2 = dm.polder_van_santen_loor(eps_eff1, 1, vAir, depolParamAir)
            return eps_eff2
        
        raise ValueError("Unidentified mixing method!")
    
    
    # Extracting the parameters
    TWater = WaterParams.get("TWater",0.01)
    SWater = WaterParams.get("SWater", 0)        
    TIce = IceParams["TIce"]
    dIce = IceParams["dIce"]
    WIce = IceParams.get("WIce", 0)
    SIce = IceParams.get("SIce", 0)              
    AIce = IceParams.get("AIce", 0)             
    TSnow = SnowParams["TSnow"]
    RoS = SnowParams.get("RoS", 300)          
    dSnow = SnowParams["dSnow"]
    WSnow = SnowParams.get("WSnow", 0) 

    if dIce==0:
        raise ValueError ("Thickness of ice can not be zero.")  
         
    if any(x is None for x in [TWater, TIce, dIce, TSnow, dSnow]):
        raise ValueError("Values for TWater, TIce, dIce, TSnow or dSnow is missing. Please check.")
    
    MixingMethod= MixingParams.get("Method", 'pvsm')                                      
    depolParamWater = MixingParams.get("depolParamWater", np.array([0.005, 0.4975, 0.4975]))     
    depolParamAir = MixingParams.get("depolParamAir", np.array([1/3, 1/3, 1/3]))   
    f = RadioParams.get("fGHz", 1.4)                                       # Assumes 1.4 GHz if no info is provided
    angle_inc = RadioParams.get("angle", 40)                               # Assumes angle of incidence as 40 degrees if no info is provided
    rough_wi = RoughnessParams.get("rough_wi", np.array([0,0,0,0]))        # Assumes specular surface if no info is provided
    rough_is = RoughnessParams.get("rough_is", np.array([0,0,0,0]))        
    rough_sa = RoughnessParams.get("rough_sa", np.array([0,0,0,0]))        
        
    # Calculation of complex permittivity values
    eps_water = dm.EpsWater_Double(TWater, f, SWater)                     # Complex permittivity of water
    eps_pure_ice = dm.EpsPureIce(TIce, f)                                  # Complex permittivity of dry ice
    if SIce > 0:
        brine_vol = dm.brine_volume_fraction(SIce, TIce)                                       # Brine volume fraction in ice (if the ice is not fresh)
        brine_eps = dm.brine_permittivity(TIce, f)
        eps_ice = DielcMixing(eps_pure_ice, brine_eps, brine_vol, vAir = 0, method=MixingMethod, depolParamWater=depolParamWater, depolParamAir=depolParamAir)
    else:
         eps_ice = eps_pure_ice
    eps_snow_ice_water = dm.EpsWater_Double(np.finfo(np.float64).eps, f)    # Complex permittivity of water in snow or ice (if the ice or snow is not dry)
    eps_ice = DielcMixing(eps_ice, eps_snow_ice_water, WIce, AIce, MixingMethod, depolParamWater, depolParamAir)       # Effective permittivity of ice with water and air voids (if exists)
    if dSnow != 0:
        eps_snow_ice = dm.EpsPureIce(TSnow, f)                                # Complex permittivity of ice in snow
        eps_dry_snow = dm.EpsDrySnow(RoS, eps_snow_ice)                       # Complex permittivity of dry snow
        eps_snow = DielcMixing(eps_dry_snow, eps_snow_ice_water, WSnow, 0, MixingMethod, depolParamWater, depolParamAir)       # Effective permittivity of snow with water (if exists)
    else:
        eps_snow = 1
    eps = np.array([eps_water, eps_ice, eps_snow, 1])                     # Array of all permittivity values (from bottom to top)
    
    
    # Calculation of internal reflectivities and transmissivities for all layers
    r_snow, t_snow, angle_snow = two_S.rt(1, eps_snow, dSnow, angle_inc, f)            # Internal reflectivity and transmissivity values for Snow layer
    r_ice, t_ice, angle_ice = two_S.rt(eps_snow, eps_ice, dIce, angle_snow, f)         # Internal reflectivity and transmissivity values for Ice layer
    
    rj = np.array([r_ice, r_snow])
    tj = np.array([t_ice, t_snow])
    angles = np.array([angle_ice, angle_snow, angle_inc])
    roughness = np.array([rough_wi, rough_is, rough_sa])
    Tj = np.array([TWater, TIce, TSnow]) + 273.15
    Tj = np.concatenate((Tj, np.array([TSky])))
    
    shj, svj = [], []                      # Lists of (Fresnel) interface reflectivities at H pol and V pol, respectively
    
    for i in range(3):
        rh, rv = two_S.InterfaceRefl(eps[i+1], eps[i], angles[i], roughness[i])
        shj.append(rh)
        svj.append(rv)
    shj, svj = np.array(shj), np.array(svj)
    
    Tbh, Tbv = two_S.emitTb(2, rj, tj, shj, svj, Tj)
    return np.array([Tbh, Tbv])
