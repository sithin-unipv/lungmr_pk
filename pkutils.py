import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math

import matplotlib.pyplot as plt

#T1w Singal to Concentration Function
def T1w_to_Con(T1w_t, T1_0, TR, FA_deg, r1, num_basepts=1):
    
    assert num_basepts>0, "need at least one time point for mean SI(t=0) computation"
    assert T1_0>0, "Assumed mean T1 value should be > 0"
    assert TR>0, "Repetition time should be > 0"
    assert r1>0, "relaxivity should be > 0"
    
    '''
    T1w_t - T x D x W x H
    T1_0 - Scalar
    TR - Scala; Repetition Time
    FA_deg - Flip Angle 
    
    
    '''
    
    #ref: https://qibawiki.rsna.org/images/1/1f/QIBA_DCE-MRI_Profile-Stage_1-Public_Comment.pdf
    
    e = 1e-12
    
    # t1w_t - array of motion corrected T1w combined in the order of acquisition time to form a single series as part of the dce acquisition
    # mask_t - is the ROI in 4D
    # (ms) T1_0 is the can be assumed/measured. In this case, we use the assumed mean T1 relaxation time for the tissue in consideration pre-contrast; for lungs T1 = 1200ms 
    # (ms) TR is the repitition time
    # FA_deg is the flip angle in degree
    # r1 is the relaxivity of the contrast agent
    # num_basepts - no of basepoints
    
    FA_rad = np.deg2rad(FA_deg) #because numpy requires radian as input
    
    E1_0 = np.exp(-TR/T1_0)
    B = (1-E1_0)/(1-np.cos(FA_rad)*E1_0)

    T1w_0 = T1w_t[:num_basepts].mean(axis=0) 
    
    A_t = B * (T1w_t)/(T1w_0 + e) # normalize by pre-contrast signal
    
    R1_0 = 1/T1_0 #baseline relation rate
    
    fraction = (1-A_t)/(1-np.cos(FA_rad) * A_t + e)
    
    R1_t = (-1/TR) * np.sign(fraction) * np.log(np.abs(fraction)) #to prevent negative values in the fraction when computing log
    
    delta_R1_t = R1_t - R1_0
    
    C_t = delta_R1_t/r1
    
    C_t[C_t<0] = 0
    
    return C_t
    
    
    
    

#BAT Estimation Methods
def search_interval(S_t, method="min-max-refined"):

    #based on https://github.com/mstorath/DCEBE/blob/master/auxiliary/DCEBE_searchIntervals.m
    #S_t corresponds to the singal over time - T1w intensity or Contrast agent concentration

    assert method in ["none", "min-max", "min-max-refined"], "method shoud either be none or min-max or min-max-refined"

    # nt = len(S_t)

    min_lb = 0
    max_ub = len(S_t)-1

    if method == "min-max":

        lb = np.argmin(S_t)
        ub = np.argmax(S_t)

    elif method == "min-max-refined":

        lb = np.argmin(S_t)
        ub = np.argmax(S_t)
        
        if lb>ub: #because of noises
            lb = min_lb

        tr = np.mean([S_t[lb], S_t[ub]])
        ub_aux = np.argmin(np.abs(S_t[lb:ub+1] - tr))

        ub = max(0, ub_aux + lb - 1) #ensure that ub is atleast 0; for example if ub_aux = 0, lb=0, then ub = -1
            
    else:
        lb = min_lb
        ub = max_ub
        
    
    if lb>ub: #because of noice
        lb = min_lb


    return max(min_lb, lb), min(max_ub, ub)

class LinearLinear_BATModel(object):
    
    # bolus arrival time estimation using Linear-Linear (L-L) Model 
    # Implementation of https://iopscience.iop.org/article/10.1088/0031-9155/48/5/403/pdf
    
    
    def __init__(self,  k=-1):
        
        self.k = k
        
    def __call__(self, t_arr, b0, b1):
        
        t_k = t_arr[self.k]
    
        return np.piecewise(t_arr, [t_arr<t_k], [b0, lambda t:b0+b1*(t-t_k)])
    
    def run(self, t_arr, S_t, search_interval, verbose=False):
        
        assert len(search_interval)==2, "expects a tuple (lowerbound, upperbound) as search_interval"
        
        lb, ub = search_interval
        
        self.bat_index = lb
        
        if ub-lb>2:
            
            if verbose:
                print(f"Search interval - {lb} to {ub}")

            p = ub-lb

            X = t_arr[lb:ub+1]
            y = S_t[lb:ub+1]
            
            if verbose:
                plt.plot(t_arr, S_t, marker=".")

            SSE = np.ones(p) * np.inf

            for k in range(p):

                p_opt, _ = curve_fit(LinearLinear_BATModel(k), X, y)

                SSE[k] = np.sum((y-LinearLinear_BATModel(k)(X, *p_opt))**2)

                if verbose:
                    plt.plot(X, LinearLinear_BATModel(k)(X, *p_opt), "--")
        
            self.bat_index = np.argmin(SSE) + lb

        
        
        if verbose:
            plt.plot(t_arr[self.bat_index], S_t[self.bat_index], marker="v", color='black')
            plt.show()
            print("bat_index instance variable generated")
        
        return self
    
class LinearQuadratic_BATModel(object):
    
    # bolus arrival time estimation using piecewise Linear-Qudratic(L-Q) Model 
    # Implementation of https://iopscience.iop.org/article/10.1088/0031-9155/48/5/403/pdf
    
    
    def __init__(self, k=-1):
        
        self.k = k
    
        
    def __call__(self, t_arr, b0, b1, b2):
        
        t_k = t_arr[self.k]
    
        return np.piecewise(t_arr, [t_arr<t_k], [b0, lambda t:b0+b1*(t-t_k)+b2*(t-t_k)**2])
    
    def run(self, t_arr, S_t, search_interval, verbose=False):

        assert len(search_interval)==2, "expects a tuple (lowerbound, upperbound) as search_interval"
        
        lb, ub = search_interval
        
        self.bat_index = lb
        
        if ub-lb>2:
            
            if verbose:
                print(f"Search interval - {lb} to {ub}")

            p = ub-lb

            X = t_arr[lb:ub+1]
            y = S_t[lb:ub+1]
            
            if verbose:
                plt.plot(t_arr, S_t, marker=".")

            SSE = np.ones(p) * np.inf

            for k in range(p):

                p_opt, _ = curve_fit(LinearQuadratic_BATModel(k), X, y)

                SSE[k] = np.sum((y-LinearQuadratic_BATModel(k)(X, *p_opt))**2)

                if verbose:
                    plt.plot(X, LinearQuadratic_BATModel(k)(X, *p_opt), ".--")

            self.bat_index = np.argmin(SSE) + lb
        
        if verbose:
            plt.plot(t_arr[self.bat_index], S_t[self.bat_index], marker="v", color='black')
            plt.show()
            print("bat_index instance variable generated")
            
        return self
    
class PeakGradient_BATModel(object):
    
    # based on https://github.com/millerjv/PkModeling/blob/master/PkSolver/PkSolver.cxx
    
    def __init__(self, window_factor = 10, polynomial_order=1, threshold_factor=10):
        
        self.smoothing_params = {"window_factor":window_factor, "polynomial_order":polynomial_order}
        self.threshold_factor = threshold_factor
    
    def run(self, t_arr, S_t, search_interval=None, verbose=False):
        
        #search interval argument is not necessary but was defined to keep the approach consistent with others
        
        window_size = len(t_arr)//self.smoothing_params["window_factor"]

        polynomial_order = self.smoothing_params["polynomial_order"]
        
        # Step 1: Smoothing done using Savizky-Golay before this call on Signal or Conc. data
        S_t_smooth = savgol_filter(S_t, window_size+1 if window_size%2==0 else window_size , polynomial_order)#expects odd number as window_size
        
        # Step 2: Spatial derivative of smoothed data independent of ascent/descent
        dS_t_smooth = np.sqrt(np.gradient(S_t_smooth)**2)
        
        # Step 3: Find point of steepest descent/ascent
        max_slope = np.max(dS_t_smooth)

        # Step 4: BAT index detection
        threshold = max_slope/self.threshold_factor
        self.bat_index = np.argmax(dS_t_smooth > threshold) + 1
        
        
        if verbose:
            plt.plot(t_arr, S_t, marker=".")
            plt.plot(t_arr, S_t_smooth, ".--")
            plt.plot(t_arr[self.bat_index], S_t[self.bat_index], marker="v", color='black')
            print("bat_index instance variable generated")
        
        return self
               

#Population AIF
#Population AIF

def parker_aif(t):
    
    '''
    t in seconds
    
    implements the Parker Arterial Input Function (2005)
    Parker et al, Experimentally derived functional form for a
    population-averaged high temporal resolution arterial input function 
    for dynamic contrast enhanced MRI
    
    ref: http://wpage.unina.it/msansone/compartmentalModelling/parker.m
    ref: https://github.com/mjt320/SEPAL/blob/master/src/aifs.py
    ref: https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/blob/develop/src/original/PvH_NKI_NL/AIF/PopulationAIF.py

    '''
    
    #constants
    A1 = 0.809 #mmol.min
    A2 = 0.330 #mmol.min
    T1 = 0.17046 #min
    T2 = 0.365 #min
    sigma1 = 0.0563 #min
    sigma2 = 0.132 #min
    alpha = 1.050 #mmol
    beta = 0.1685 #min-1
    s = 38.078 #min-1
    tau = 0.483 #min
    
    #input
    t = t/60 # converting seconds to minutes; this function need minutes input
    
    Cp =  A1 * (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(t-T1)**2/(2*sigma1**2)) + \
          A2 * (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(t-T2)**2/(2*sigma2**2)) + \
          alpha * (np.exp(-beta * t) / (1 + np.exp(-s*(t-tau))))
    
    # Cp[t<0] = 0 by default Cp will be 0 for t<0
    
    return Cp


    
def weinmann_aif(t):
    
    #t in seconds; 
    #ref: http://wpage.unina.it/msansone/compartmentalModelling/weinmann.m
    
    
    t = t/60 # (t-t_start)/60
    
    #constants
    A1 = 3.99 # kg/L
    A2 = 4.78 # kg/L
    M1 = 0.144 #min-1
    M2 = 0.0111 #min-1
    D = 0.2 #mmol/kg

    Cp = D * ((A1 * np.exp(-M1 * t) + A2 * np.exp(-M2 * t)))
    Cp[t<0] = 0 # is important to shift aif
    
    return Cp

def georgiou_aif(t):
    
    #ref https://github.com/OSIPI/DCE-DSC-MRI_CodeCollection/blob/develop/src/original/PvH_NKI_NL/AIF/PopulationAIF.py
    #adapatation of Geourgiou et al. MRM 2018, doi: 10.1002/mrm.27524
    
    #constants
    a1=0.37 #mM
    m1=0.11 #min-1
    a2=0.33 #mM
    m2=1.17 #min-1
    a3=10.06 #mM
    m3=16.02 #min-1
    alpha=5.26
    beta=0.032 #min
    tau=0.129 #min
    
    t = t/60 #to minutes
    nt = len(t)
    
    no_circ = round(t[nt-1] / tau)
    Cb = np.zeros(nt)
    
    for current_circ in range(0, no_circ+1):
        if current_circ < no_circ:
            timeindex = np.where((t >= current_circ*tau) & (t < (current_circ+1)*tau))
        else:
            timeindex = np.where(t >= current_circ*tau)

        current_time = t[timeindex]

        ftot = np.zeros(len(timeindex))
        for k in range(0, current_circ+1):
            alphamod = (k + 1) * alpha + k
            timemod = current_time - k * tau
            f1 = np.power(timemod, alphamod) * np.exp(-timemod / beta)
            try:
                ans = math.gamma(alphamod + 1)
                f2 = np.power(beta, alphamod + 1) * math.gamma(alphamod + 1)  
                f3 = f1 / f2
                ftot = ftot + f3
            except OverflowError:
                ans = float('inf')

        exp1 = a1 * np.exp(-m1 * current_time)
        exp2 = a2 * np.exp(-m2 * current_time)
        exp3 = a3 * np.exp(-m3 * current_time)
        sumexp = np.add(exp1, exp2)
        sumexp = np.add(sumexp, exp3)
        Cb[timeindex] = sumexp*ftot
        
    # Cb[t<0] = 0 by default Cb will be 0 for t<0
        
    return Cb


def delay_corrected_aif(aif_fn, t_arr, t0):
    
    return aif_fn(t_arr-t0)
    


#PK Models

class StandardTofts(object):
    
    def __init__(self, Cp_t):
        
        self.Cp_t = Cp_t
        
    def __call__(self, t_arr, ktrans, ve):
        
        Ct_t = np.zeros_like(t_arr, dtype=np.float32)
        
        Cp_t = self.Cp_t
        
        for i,t in enumerate(t_arr):
            
            ti = t_arr[:i+1]
            Cp_ti = Cp_t[:i+1]
            
            Ct_t[i] = ktrans * np.trapz(Cp_ti * np.exp((-ktrans/ve)*(t-ti)), ti)
            
        return Ct_t
    
class ExtendedTofts(object):
    
    def __init__(self, Cp_t):
        
        self.Cp_t = Cp_t
        
    def __call__(self, t_arr, vp, ktrans, ve):
        
        Ct_t = np.zeros_like(t_arr, dtype=np.float32)
        Cp_t = self.Cp_t
        
        for i,t in enumerate(t_arr):

            ti = t_arr[:i+1]
            Cp_ti = Cp_t[:i+1]

            Ct_t[i] = vp * Cp_t[i] + ktrans * np.trapz(Cp_ti * np.exp(-(ktrans/ve)*(t-ti)), ti)

        return Ct_t
        


class delayCorrectedStandardTofts(object):
    
    def __init__(self, aif_fn):
        
        self.aif_fn = aif_fn 
        
    def __call__(self, t_arr, t0, ktrans, ve):
        
        #t0 is the bolus arrival time to account for the delay
        #the estimated Ktrans will be in sec-1
        #Ve is in percent

        Ct_t = np.zeros_like(t_arr, dtype=np.float32)
        
        Cp_t = self.aif_fn(t_arr-t0)
        
        for i,t in enumerate(t_arr):

            ti = t_arr[:i+1]
            Cp_ti = Cp_t[:i+1]

            Ct_t[i] = ktrans * np.trapz(Cp_ti * np.exp((-ktrans/ve)*(t-ti)), ti)

        return Ct_t
    
# Kep = -Ktrans/Ve


class delayCorrectedExtendedTofts(object):
    
    def __init__(self, aif_fn):
        
        self.aif_fn = aif_fn
        
    def __call__(self, t_arr, t0, vp, ktrans, ve):
        
        #t0 is the bolus arrival time to account for the delay
        #the estimated Ktrans will be in sec-1
        # Vp, Ve are in percent
        
        Ct_t = np.zeros_like(t_arr, dtype=np.float32)
        Cp_t = self.aif_fn(t_arr - t0)
        
        for i,t in enumerate(t_arr):

            ti = t_arr[:i+1]
            Cp_ti = Cp_t[:i+1]

            Ct_t[i] = vp * Cp_t[i] + ktrans * np.trapz(Cp_ti * np.exp((-ktrans/ve)*(t-ti)), ti)

        return Ct_t
    
# Kep = Ktrans/Ve

#iAUCx and iAUCxbn
def get_iAUCpx_with_functionalAIF(aif_fn, t_arr, t0, x=90): #iAUCx of Cp_t

    #t0 is the bolus arrival time

    dt = np.gradient(t_arr).min() #minimum time interval
    t_arr = np.arange(t0, t0+x, dt)

    Cp_t = aif_fn(t_arr)
    
    # mask = (t_arr>=t0)&(t_arr<=(t0+x))

    # roi_t_arr = t_arr[mask]
    # roi_Cp_t = Cp_t[mask]

    return np.trapz(Cp_t, t_arr)

def get_iAUCpx_with_measuredAIF(Cp_t, t_arr, t0, x=90): #iAUCx of Cp_t
    
    #t0 is the bolus arrival time
    
    mask = (t_arr>=t0)&(t_arr<=(t0+x))
    
    roi_t_arr = t_arr[mask]
    roi_Cp_t = Cp_t[mask]
    
    return np.trapz(roi_Cp_t, roi_t_arr)


def get_iAUCx_with_functionalAIF(PKModel, aif_fn, p_opt, t_arr, t0, x=30): #iAUCx of fitted Ct_t
    
    #PKModel can either be standardTofts or extendedTofts, delayCorrection should be done explicity
    #t_arr, C_t are 1d arrays

    # interp_fn = interp1d(t_arr, C_t, kind=interp_kind, bounds_error=False, fill_value = "extrapolate") # this didn't work out
    
    
    dt = np.gradient(t_arr).min()
    
    t_arr = np.arange(t0, t0+x, dt)
    Cp_t = aif_fn(t_arr)
    
    
    C_t = PKModel(Cp_t)(t_arr, *p_opt)
    
    # mask = (t_arr>=t0)&(t_arr<=(t0+x))
    
    # roi_t_arr = t_arr[mask]
    # roi_C_t = C_t[mask]
    
    return np.trapz(C_t, t_arr)


def get_iAUCx_with_measuredAIF(C_t, Cp_t, t_arr, t0, x=30): #iAUCx of fitted Ct_t
    
    #t_arr, C_t are 1d arrays

    # interp_fn = interp1d(t_arr, C_t, kind=interp_kind, bounds_error=False, fill_value = "extrapolate") # this didn't work out

    mask = (t_arr>=t0)&(t_arr<=(t0+x))
    
    roi_t_arr = t_arr[mask]
    roi_C_t = C_t[mask]
    
    return np.trapz(roi_C_t, roi_t_arr)


def get_iAUCx_bn_with_functionalAIF(PKModel, aif_fn, p_opt, t_arr, t0, x=90):
    
    e = 1e-12
    
    # The blood normalized IAUGCBN is defined as the area under the concentration curve from the baseline timepoint up to 90 seconds post bolus arrival within the tumor
    # divided by the area under the vascular input function curve up to 90 seconds post the baseline timepoint within the vessel.
    # ref: https://qibawiki.rsna.org/images/7/7b/DCEMRIProfile_v1_6-20111213.pdf
    # Ct_t -> 1d array; tumor concentration
    # Cp_t -> 1d array; vascular input function
    
    return get_iAUCx_with_functionalAIF(PKModel, aif_fn, p_opt, t_arr, t0, x=x)/(get_iAUCpx_with_functionalAIF(aif_fn, t_arr, t0, x=x)+e)

def get_iAUCx_bn_with_measuredAIF(C_t, Cp_t, t_arr, t0, x=90):
    
    e = 1e-12
    
    # The blood normalized IAUGCBN is defined as the area under the concentration curve from the baseline timepoint up to 90 seconds post bolus arrival within the tumor
    # divided by the area under the vascular input function curve up to 90 seconds post the baseline timepoint within the vessel.
    # ref: https://qibawiki.rsna.org/images/7/7b/DCEMRIProfile_v1_6-20111213.pdf
    # Ct_t -> 1d array; tumor concentration
    # Cp_t -> 1d array; vascular input function
    
    return get_iAUCx_with_measuredAIF(C_t, Cp_t, t_arr, t0, x=x)/(get_iAUCpx_with_measuredAIF(Cp_t, t_arr, t0, x=x) + e)
    
    
    