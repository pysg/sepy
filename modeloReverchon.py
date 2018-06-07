import numpy as np
import pandas as pd
import math
import cmath
from scipy.optimize import root
from scipy.integrate import odeint
from __future__ import division
from scipy import *
from pylab import *
import matplotlib.pyplot as plt
%matplotlib inline

import pylab as pp
from scipy import integrate, interpolate
from scipy import optimize

#----------------------------------------------------------------------------------------

P = 9 #MPa
T = 323 # K
Q = 8.83 #g/min
e = 0.4
rho = 285 #kg/m3
miu = 2.31e-5 # Pa*s
dp = 0.75e-3 # m
Dl = 0.24e-5 #m2/s
De = 8.48e-12 # m2/s
Di = 6e-13
u = 0.455e-3 #m/s
kf = 1.91e-5 #m/s
de = 0.06 # m
W = 0.160 # kg
kp = 0.2

r = 0.31 #m

n = 10
V = 12

#C = kp * qE
C = 0.1
qE = C / kp

Cn = 0.05
Cm = 0.02


t = np.linspace(0,10, 1)


def reverchon(x, t, Di, kp):
    
    #Ecuaciones diferenciales del modelo Reverchon    
    #dCdt = - (n/(e * V)) * (W * (Cn - Cm) / rho + (1 - e) * V * dqdt)
    #dqdt = - (1 / ti) * (q - qE)
    
    q = x[0]
    C = x[1]
    ti = (r ** 2) / (15 * Di)
    qE = C / kp
    dqdt = - (1 / ti) * (q - qE)
    dCdt = - (n/(e * V)) * (W * (C - Cm) / rho + (1 - e) * V * dqdt)
    
    return [dqdt, dCdt]  


reverchon([1, 2], 0, Di, kp)

x0 = [0, 0]
#t = np.linspace(0, 3000, 500)
t = np.linspace(0, 3000, 30)

resultado = odeint(reverchon, x0, t, args=(Di, kp))

qR = resultado[:, 0]
CR = resultado[:, 1]
plt.plot(t, CR)
plt.title("Modelo Reverchon")
plt.xlabel("t [=] min")
plt.ylabel("C [=] $kg/m^3$")

#----------------------------------------------------------------------------------------



P = 9 #MPa
T = 323 # K
Q = 8.83 #g/min
e = 0.4
rho = 285 #kg/m3
miu = 2.31e-5 # Pa*s
dp = 0.75e-3 # m
Dl = 0.24e-5 #m2/s
De = 8.48e-12 # m2/s
Di = 6e-13
u = 0.455e-3 #m/s
kf = 1.91e-5 #m/s
de = 0.06 # m
W = 0.160 # kg
kp = 0.2

r = 0.31 #m

n = 10
V = 12

#C = kp * qE
C = 0.1
qE = C / kp

Cn = 0.05
Cm = 0.02


#Datos experimentales
x_data = np.linspace(0,9,10)
y_data = array([ 0.00429861,  0.00907806,  0.01142553,  0.01471523,  0.01585107,
        0.01674278,  0.01744284,  0.01799243,  0.01860349,  0.01902855])


def reverchon(x,t, parametros):
    
    #Ecuaciones diferenciales del modelo Reverchon    
    #dCdt = - (n/(e * V)) * (W * (Cn - Cm) / rho + (1 - e) * V * dqdt)
    #dqdt = - (1 / ti) * (q - qE)
    
    q = x[0]
    C = x[1]
    #ti = (r ** 2) / (15 * Di)
    ti = (r ** 2) / (15 * parametros[0])
    #qE = C / kp
    qE = C / parametros[1]
    dqdt = - (1 / ti) * (q - qE)
    dCdt = - (n/(e * V)) * (W * (C - Cm) / rho + (1 - e) * V * dqdt)
    
    return [dqdt, dCdt]  

def my_ls_func(x, parametros):
    f2 = lambda y, t: reverchon(y, t, parametros)
    # calcular el valor de la ecuación diferencial en cada punto
    r = integrate.odeint(f2, y0, x)
    return r[:,1]

def f_resid(p):
    # definir la función de minimos cuadrados para cada valor de y"""
    
    return y_data - my_ls_func(x_data,p)

#resolver el problema de optimización
# guess = [0.2, 0.3] #valores inicales para los parámetros # funcionan bien
guess = [0.2, 0.3] #valores inicales para los parámetros # funcionan bien
#y0 = [1,0,0] #valores inciales para el sistema de ODEs
y0 = [1,0] #valores inciales para el sistema de ODEs
(c, kvg) = optimize.leastsq(f_resid, guess) #get params

print("parameter values are ",c)

# interpolar los valores de las ODEs usando splines
xeval = np.linspace(min(x_data), max(x_data),30) 
gls = interpolate.UnivariateSpline(xeval, my_ls_func(xeval,c), k=3, s=0)


xeval = np.linspace(min(x_data), max(x_data), 200)
#Gráficar los resultados
pp.plot(x_data, y_data,'.r',xeval,gls(xeval),'-b')
pp.xlabel('t [=] min',{"fontsize":16})
pp.ylabel("C",{"fontsize":16})
pp.legend(('Datos','Modelo'),loc=0)
pp.show()


#----------------------------------------------------------------------------------------

















