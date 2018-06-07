import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.integrate import odeint
from __future__ import division
from scipy import *
from pylab import *
import matplotlib.pyplot as plt
#%matplotlib inline


xo = 0.5
gamma = 0.8
yr = 0.1 #kg kgCO2 −1
TAO = 15


# en la matriz parametros se define cada conbinación (xk, A) como una fila de la matriz
parametros = np.array([[0.1, 1],[0.3, 2],[0.4, 4]])
parametros


def intervalosExtraccion(tao, xk, A):
    tao1 = (xo - xk) / (gamma * A * yr)
    tao2 = tao1 + xk / (gamma * A * yr) * np.log(xk / xo + (1 - xk / xo) * np.exp(xo / xk * A))
    zk = xk / (A * xo) * np.log((xo * np.exp(gamma * A * yr / xk * (tao - tao1)) - xk) / (xo - xk))

    return tao1, tao2, zk



def modeloLack(tao, xk, A):
    
    tao1, tao2, zk = intervalosExtraccion(tao, xk, A)
    
    if tao <= tao1 and tao < tao2:
        e = gamma * yr* tao * (1- np.exp(- A))
        print("tao < tao1 and tao < tao2")
        return e
    if tao > tao1 and tao <= tao2:
        zk = xk / (A * xo) * np.log((xo * np.exp(gamma * A * yr / xk * (tao - tao1)) - xk) / (xo - xk))
        e = gamma * yr * (tao - tao1 * np.exp(- A * (1 - zk)))
        print("tao >= tao1 and tao < tao2")
        return e
    if tao > tao2:
        e = xo - xk / A * np.log(1 + xk / xo * (np.exp(xo / xk * A) - 1) * np.exp(gamma * A * yr / xk * (tao1 - tao)))
        print("tao >= tao2")
        return e


rendimiento = [[modeloLack(tao, xk, A) for tao in np.linspace(0,TAO)] for xk, A in parametros]



def graficarLack():

	Tao = np.linspace(0,TAO)
	# plt.plot(Tao, caso1[2],label="xk=0.3")
	# plt.plot(Tao,caso2[2],label="xk=0.1")
	# plt.plot(Tao,caso3[2],label="xk=0.4")
	
	plt.plot(Tao, rendimiento[0],label="xk=0.3")
	plt.plot(Tao, rendimiento[1],label="xk=0.1")
	plt.plot(Tao, rendimiento[2],label="xk=0.4")
	plt.title("Modelo Lack")
	plt.xlabel(" $tao $ ")
	plt.ylabel("Rendimiento e")
	plt.legend()

	return 0




















