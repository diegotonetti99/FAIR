from numpy import log, exp, sqrt, abs
import numpy as np
from scipy.optimize import root_scalar

Ma = 5.1352e18
omega_a = 28.96e-3 # air molecular mass

def landuse(F0, Eco2_land):
    """ returns the ERF for the landuse given the ERF F0 at time t-1 and the emissions Eco2_land at time t """
    a = -1.14e-3
    Ft = F0 + a*Eco2_land
    return Ft

def blac_carbon_snow(Ebc):
    return 0.00494*Ebc

def aerosol(Ebc, Eoc, Esox, Enox, Enh3, Enmovc):
    E = [Esox, Ebc, Eoc]
    E2011 = [0 , 0 , 0] # emission constants from 2011 -> to implement
    E1765 = [0 , 0 , 0] # emission constants from 1765 -> to implement
    Faci = -.045 * (G(E)-G(E1765))/(G(E2011)-G(E1765))
    gamma = [0, 0, 0, 0 ,0, 0] # gamma constants to implement
    Fari = gamma[0]*Ebc+gamma[1]*Eoc+gamma[2]*Esox+gamma[3]*Enox+gamma[4]*Enh3+gamma[5]*Enmovc
    return Fari + Faci


def G(Esox, Ebc, Eoc):
    Esox, Ebc, Eoc = E[0], E[1], E[2]
    return -1.95*log(1+0.0111*Esox+0.0139*(Ebc+Eoc))

def contrails(Enox):
    return Enox*0.0152

def f_o3st(C_ods):
    ri = np.array([0.47,0.23,0.29,0.12,0.04,.56,.67,.13,.34,.17,.62,.62,.28,.65,.60,.44])
    eta_cl = np.array([3,2,3,2,1,4,3,1,2,1,1,0,0,0,0,1])
    eta_br = np.array([0,0,0,0,0,0,0,0,0,0,1,2,1,2,1,0])
    s = 0
    for i in range(0,len(Ci)):
        s = s + (eta_cl[i]*Ci[i]+45*eta_br*Ci[i])*ri[i]/r[0]
    s = s * ri[0]
    a = -1.46e-5
    b = 2.05e-3
    c = 1.03
    return a*(b*s)**c

def f_o3tr(Cch4, Enox, Eco, Enmovc, T0):
    beta = np.array([.178, .076, .044, .125])
    v_pi = np.array([772, 170, 5, 2])
    v = np.array([Cch4, Enox, Eco, Enmovc])
    ft = min(0, 0.032*exp(T0)-0.032)
    return beta*(v-v_pi) + ft

def f_i(Ci, eta_i, Ci_pi):
    return eta_i*(Ci-Ci_pi)

def f_n2o(C, N, M):
    Cpi = 0 # constant
    Npi = 0 # constant
    Mpi = 0 # constant
    return (-4e-6*(C+Cpi)+2.1e-6*(N+Npi)-2.45e-6*(M+Mpi)+0.117)*(sqrt(N)-sqrt(Npi))

def f_ch4(N, M):
    Npi = 0 # constant
    Mpi = 0 # constant
    return (-6.5e-7*(M+Mpi)-4.1e-6*(N+Npi)+0.043)*(sqrt(M)-sqrt(Mpi))

def f_co2(C, N):
    Cpi = 0
    Npi = 0
    return (-2.4e-7*(C-Cpi)**2+7.2e-4*abs(C-Cpi)-1.05e-4*(N-Npi)+5.36)*log(C/Cpi)

def Ct(delta_Ct, delta_Ct1, Ct1, tau):
    return Ct1+0.5*(delta_Ct1+delta_Ct)-Ct1*(1-exp(-1/tau))

def delta_Ct(Et, omega_f):
    return Et*omega_a/(Ma*omega_f)

def C_co2(Eco2, R0):
    alpha = root_scalar(lambda alpha: get_alpha(alpha, T0, Cacc, tau, a))
    R = get_R(R0, Eco2, tau, a, T0, alpha)
    Cco2_pi = 0
    omega_co2 = 0
    Cco2 = Cco2_pi
    for i in range(0,4):
        Cco2=Cco2+R[i]*omega_co2/(Ma*omega_a)
    return Cco2, R
    
def get_R(R0, Eco2, tau, a, T0, alpha):
    return (R0+a*Eco2)/(1+1/(alpha*tau))

def get_alpha(alpha, T0, Cacc, tau, a):
    r0 = 35
    rt = 4.165
    rc = 0.019
    Tpi = 0
    c = r0+rc*Cacc+rt*(T0-Tpi)
    s = -c
    for i in range(0,4):
        s = s + alpha * tau[i] * a[i] * (1-exp(-100/(alpha*tau[i])))

def get_Cacc(Et, Cacc0, Ct, Ct0):
    return Et + Cacc0-(Ct-Ct0)