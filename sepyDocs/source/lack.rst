
.. code:: ipython3

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

Evaluation des modèles pour l'extraction supercritique
======================================================

L'extraction supercritique est de plus en plus utilisée afin de retirer
des matières organiques de différents liquides ou matrices solides. Cela
est dû au fait que les fluides supercritiques ont des avantages non
négligeables par rapport aux autres solvants, ils ont des
caractèreistiques comprises entre celles des gaz et celles des solides.
En changeant la température et la pression ils peuvent capter des
composés différents, ils sont donc très efficaces. Le méchanisme de
l'extraction supercritique est le suivant : - Transport du fluide vers
la particule, en premier lieu sur sa surface et en deuxième lieu a
l'intérieur de la particule par diffusion - Dissolution du soluté avec
le fluide supercritique - Transport du solvant de l'intérieur vers la
surface de la particule - Transport du solvant et des solutés de la
surface de la particule vers la masse du solvant

Importance du modèle
====================

Il est important d'avoir outil permettant de modéliser l'extraction
supercritique car cela permet de déterminer les paramètres optimaux sans
avoir à passer par l'expérience. Cela permet également d'exporter ce
modèle sur des échelles plus grandes et donc sortir du laboratoire. De
plus l'extraction supercritique est difficile à mettre en place car les
pressions et les températures sont très élevées (Exemple pour le CO2 :
31°C et 74 bars).

A - Le modèle de Reverchon :

Afin d'utiliser ce modèle, définissons les variables qui vont y être
admises, ci-dessous la nomenclature du modèle : - Ap : Total surface of
particles - c : extract concentration in the fluid phase - cn : fluid
phase concentration in the nth stage - kp : volumetric partition
coefficient - K : internal mass-transfer coefficient - n : number of
stages deriving fromf the bed subdivision - q : extract concentration in
the solid phase - qn : solid phase concentration in the nth stage - q\*
: concentration at the solid-fluid interface - t : extraction time - ti
: internal diffusion time - u : superficial velocity - V : extractor
volume - W : CO2 mass flow rate - E : bed porosity - p : solvent density

Le modèle : Il est basé sur l'intégration des bilans de masses
différentielles tout le long de l'extraction, avec les hypothèses
suivants : - L'écoulement piston existe à l'intérieur du lit, comme le
montre le schéma ci-contre : |activit%C3%A910.PNG| - La dispersion
axiale du lit est négligeable - Le débit, la température et la pression
sont constants

Cela nous permet d'obtenir les équations suivantes : -
:math:`uV.\frac{\partial c_{c}}{\partial t}+eV.\frac{\partial c_{c}}{\partial t}+ AK(q-q*) = 0`
- :math:`(1-e).V.uV*\frac{\partial c_{q}}{\partial t}= -AK(q-q*)`

-  Les conditions initiales sont les suivantes : C = 0, q=q0 à t = 0 et
   c(0,t) à h=0

La phase d'équilibre est : :math:`c = k.q*`

.. |activit%C3%A910.PNG| image:: attachment:activit%C3%A910.PNG

Sachant que le fluide et la phase sont uniformes à chaque stage, nous
pouvons définir le modèle en utilisant les équations différentielles
ordinaires (2n). Les équations sont les suivantes : -
:math:`(\frac{W}{p}).(Cn- Cn-1) + e (\frac{v}{n}).(\frac{dcn}{dt})+(1-e).(\frac{v}{n}).(\frac{dcn}{dt}) = 0`
- :math:`(\frac{dqn}{dt} = - (\frac{1}{ti})(qn-qn*)` - Les conditions
initiales sont : cn = 0, qn = q0 à t = 0

Ejemplo ODE
===========

.. code:: ipython3

    import numpy as np
    from scipy import integrate
    from matplotlib.pylab import *

Ejemplo 2 funciona
==================

.. code:: ipython3

    import numpy as np
    from scipy import integrate
    import matplotlib.pyplot as plt
    
    def vdp1(t, y):
        return np.array([y[1], (1 - y[0]**2)*y[1] - y[0]])
    
    t0, t1 = 0, 20                # start and end
    t = np.linspace(t0, t1, 100)  # the points of evaluation of solution
    y0 = [2, 0]                   # initial value
    y = np.zeros((len(t), len(y0)))   # array for solution
    y[0, :] = y0
    
    r = integrate.ode(vdp1).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t0)   # initial values
    
    for i in range(1, t.size):
       y[i, :] = r.integrate(t[i]) # get one more value, add it to the array
       if not r.successful():
           raise RuntimeError("Could not integrate")
    
    plt.plot(t, y)
    plt.show()



.. image:: output_10_0.png


Modelo Reverchon
================

Mathematical Modeling of Supercritical Extraction of Sage Oil

.. code:: ipython3

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
    
    


.. code:: ipython3

    N = 10
    
    Q = np.ones(N+1)
    Q[0] = 0
    
    CC = np.ones(N+1)
    CC[0] = 0
    CC
    
    





.. parsed-literal::

    array([ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])



.. code:: ipython3

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


.. code:: ipython3

    reverchon([1, 2], 0, Di, kp)




.. parsed-literal::

    [8.428720083246617e-10, -0.0023158021167643352]



.. code:: ipython3

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




.. parsed-literal::

    <matplotlib.text.Text at 0x7fc5c0d120f0>




.. image:: output_16_1.png


.. code:: ipython3

    CR




.. parsed-literal::

    array([ 0.        ,  0.00227917,  0.00429861,  0.00608793,  0.00767333,
            0.00907806,  0.01032272,  0.01142553,  0.01240263,  0.01326841,
            0.01403553,  0.01471523,  0.01531746,  0.01585107,  0.01632387,
            0.01674278,  0.01711396,  0.01744284,  0.01773424,  0.01799243,
            0.01822119,  0.01842389,  0.01860349,  0.01876262,  0.01890362,
            0.01902855,  0.01913924,  0.01923732,  0.01932422,  0.01940122])



.. code:: ipython3

    #y_data = np.array([0.000,0.416,0.489,0.595,0.506,0.493,0.458,0.394,0.335,0.309])
    datos = np.array([CR[2], CR[5], CR[7], CR[11], CR[13], CR[15], CR[17], CR[19], CR[22], CR[25]])
    datos




.. parsed-literal::

    array([ 0.00429861,  0.00907806,  0.01142553,  0.01471523,  0.01585107,
            0.01674278,  0.01744284,  0.01799243,  0.01860349,  0.01902855])



.. code:: ipython3

    x0 = [0, 0]
    t = np.linspace(0, 3000, 500)
    
    resultado = odeint(reverchon, x0, t)
    
    qR = resultado[:, 0]
    CR = resultado[:, 1]
    plt.plot(t, qR)
    plt.title("Modelo Reverchon")
    plt.xlabel("t [=] min")
    plt.ylabel("C solid–fluid interface [=] $kg/m^3$")




.. parsed-literal::

    <matplotlib.text.Text at 0x7f1d7976e518>




.. image:: output_19_1.png


.. code:: ipython3

    print(CR)


.. parsed-literal::

    [ 0.2         0.19995782  0.19991564  0.19987347  0.19983132  0.19978917
      0.19974704  0.19970491  0.1996628   0.19962069  0.1995786   0.19953651
      0.19949443  0.19945237  0.19941031  0.19936826  0.19932623  0.1992842
      0.19924218  0.19920017  0.19915818  0.19911619  0.19907421  0.19903224
      0.19899029  0.19894834  0.1989064   0.19886447  0.19882255  0.19878064
      0.19873874  0.19869685  0.19865497  0.1986131   0.19857124  0.19852939
      0.19848755  0.19844572  0.1984039   0.19836209  0.19832029  0.1982785
      0.19823672  0.19819495  0.19815318  0.19811143  0.19806969  0.19802796
      0.19798623  0.19794452  0.19790282  0.19786112  0.19781944  0.19777777
      0.1977361   0.19769445  0.1976528   0.19761117  0.19756954  0.19752793
      0.19748632  0.19744473  0.19740314  0.19736156  0.19732     0.19727844
      0.19723689  0.19719536  0.19715383  0.19711231  0.1970708   0.1970293
      0.19698782  0.19694634  0.19690487  0.19686341  0.19682196  0.19678052
      0.19673909  0.19669767  0.19665626  0.19661485  0.19657346  0.19653208
      0.19649071  0.19644935  0.19640799  0.19636665  0.19632532  0.19628399
      0.19624268  0.19620137  0.19616008  0.19611879  0.19607752  0.19603625
      0.195995    0.19595375  0.19591251  0.19587129  0.19583007  0.19578886
      0.19574766  0.19570648  0.1956653   0.19562413  0.19558297  0.19554182
      0.19550068  0.19545955  0.19541843  0.19537732  0.19533622  0.19529512
      0.19525404  0.19521297  0.19517191  0.19513085  0.19508981  0.19504877
      0.19500775  0.19496673  0.19492573  0.19488473  0.19484375  0.19480277
      0.1947618   0.19472085  0.1946799   0.19463896  0.19459803  0.19455711
      0.1945162   0.1944753   0.19443441  0.19439353  0.19435266  0.1943118
      0.19427095  0.19423011  0.19418927  0.19414845  0.19410764  0.19406683
      0.19402604  0.19398525  0.19394448  0.19390371  0.19386296  0.19382221
      0.19378147  0.19374075  0.19370003  0.19365932  0.19361862  0.19357793
      0.19353725  0.19349658  0.19345592  0.19341527  0.19337463  0.19333399
      0.19329337  0.19325276  0.19321215  0.19317156  0.19313098  0.1930904
      0.19304983  0.19300928  0.19296873  0.1929282   0.19288767  0.19284715
      0.19280664  0.19276614  0.19272565  0.19268517  0.1926447   0.19260424
      0.19256379  0.19252335  0.19248291  0.19244249  0.19240208  0.19236167
      0.19232128  0.19228089  0.19224052  0.19220015  0.19215979  0.19211945
      0.19207911  0.19203878  0.19199846  0.19195815  0.19191785  0.19187756
      0.19183728  0.19179701  0.19175674  0.19171649  0.19167625  0.19163601
      0.19159579  0.19155557  0.19151537  0.19147517  0.19143498  0.1913948
      0.19135464  0.19131448  0.19127433  0.19123419  0.19119406  0.19115394
      0.19111382  0.19107372  0.19103363  0.19099355  0.19095347  0.19091341
      0.19087335  0.19083331  0.19079327  0.19075324  0.19071322  0.19067322
      0.19063322  0.19059323  0.19055325  0.19051327  0.19047331  0.19043336
      0.19039342  0.19035348  0.19031356  0.19027365  0.19023374  0.19019384
      0.19015396  0.19011408  0.19007421  0.19003435  0.1899945   0.18995466
      0.18991483  0.18987501  0.1898352   0.1897954   0.1897556   0.18971582
      0.18967604  0.18963628  0.18959652  0.18955678  0.18951704  0.18947731
      0.18943759  0.18939788  0.18935818  0.18931849  0.18927881  0.18923914
      0.18919947  0.18915982  0.18912018  0.18908054  0.18904091  0.1890013
      0.18896169  0.18892209  0.1888825   0.18884292  0.18880335  0.18876379
      0.18872424  0.1886847   0.18864517  0.18860564  0.18856613  0.18852662
      0.18848713  0.18844764  0.18840816  0.18836869  0.18832923  0.18828979
      0.18825034  0.18821091  0.18817149  0.18813208  0.18809267  0.18805328
      0.1880139   0.18797452  0.18793515  0.1878958   0.18785645  0.18781711
      0.18777778  0.18773846  0.18769915  0.18765984  0.18762055  0.18758127
      0.18754199  0.18750273  0.18746347  0.18742423  0.18738499  0.18734576
      0.18730654  0.18726733  0.18722813  0.18718894  0.18714975  0.18711058
      0.18707142  0.18703226  0.18699312  0.18695398  0.18691485  0.18687573
      0.18683662  0.18679752  0.18675843  0.18671935  0.18668028  0.18664122
      0.18660216  0.18656312  0.18652408  0.18648505  0.18644604  0.18640703
      0.18636803  0.18632904  0.18629006  0.18625109  0.18621212  0.18617317
      0.18613423  0.18609529  0.18605636  0.18601745  0.18597854  0.18593964
      0.18590075  0.18586187  0.185823    0.18578414  0.18574528  0.18570644
      0.1856676   0.18562878  0.18558996  0.18555115  0.18551235  0.18547356
      0.18543478  0.18539601  0.18535725  0.1853185   0.18527975  0.18524102
      0.18520229  0.18516357  0.18512487  0.18508617  0.18504748  0.1850088
      0.18497013  0.18493146  0.18489281  0.18485416  0.18481553  0.1847769
      0.18473829  0.18469968  0.18466108  0.18462249  0.18458391  0.18454534
      0.18450677  0.18446822  0.18442967  0.18439114  0.18435261  0.18431409
      0.18427558  0.18423708  0.18419859  0.18416011  0.18412164  0.18408318
      0.18404472  0.18400628  0.18396784  0.18392941  0.18389099  0.18385258
      0.18381418  0.18377579  0.18373741  0.18369903  0.18366067  0.18362231
      0.18358397  0.18354563  0.1835073   0.18346898  0.18343067  0.18339237
      0.18335408  0.18331579  0.18327752  0.18323925  0.18320099  0.18316275
      0.18312451  0.18308628  0.18304806  0.18300984  0.18297164  0.18293345
      0.18289526  0.18285709  0.18281892  0.18278076  0.18274261  0.18270447
      0.18266634  0.18262822  0.1825901   0.182552    0.1825139   0.18247581
      0.18243774  0.18239967  0.18236161  0.18232356  0.18228551  0.18224748
      0.18220946  0.18217144  0.18213343  0.18209544  0.18205745  0.18201947
      0.1819815   0.18194353  0.18190558  0.18186764  0.1818297   0.18179177
      0.18175386  0.18171595  0.18167805  0.18164016  0.18160228  0.1815644
      0.18152654  0.18148868  0.18145084  0.181413    0.18137517  0.18133735
      0.18129954  0.18126174  0.18122394  0.18118616  0.18114838  0.18111061
      0.18107286  0.18103511  0.18099737  0.18095964  0.18092191  0.1808842
      0.18084649  0.1808088   0.18077111  0.18073343  0.18069576  0.1806581
      0.18062045  0.18058281  0.18054517  0.18050755  0.18046993  0.18043232
      0.18039472  0.18035713  0.18031955  0.18028198  0.18024442  0.18020686
      0.18016931  0.18013178]


.. code:: ipython3

    r = 0.31 #m
    x0 = [0, 0]
    t = np.linspace(0, 3000, 500)
    
    resultado = odeint(reverchon, x0, t)
    
    qR = resultado[:, 0]
    CR = resultado[:, 1]
    plt.plot(t, CR)
    plt.title("Modelo Reverchon")
    plt.xlabel("t [=] min")
    plt.ylabel("C [=] $kg/m^3$")




.. parsed-literal::

    <matplotlib.text.Text at 0x7f1d79cbc908>




.. image:: output_21_1.png


.. code:: ipython3

    r = 0.231 #m
    x0 = [0, 0]
    t = np.linspace(0, 3000, 500)
    
    resultado = odeint(reverchon, x0, t)
    
    qR = resultado[:, 0]
    CR = resultado[:, 1]
    plt.plot(t, CR)
    plt.title("Modelo Reverchon")
    plt.xlabel("t [=] min")
    plt.ylabel("C [=] $kg/m^3$")




.. parsed-literal::

    <matplotlib.text.Text at 0x7f1d79bfc1d0>




.. image:: output_22_1.png


.. code:: ipython3

    fig,axes=plt.subplots(2,2)
    axes[0,0].plot(t,CR)
    axes[1,0].plot(t,qR)





.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x7f1d7937dd30>]




.. image:: output_23_1.png


Ajuste de parámetros con ODEs: modelo Reverchon
===============================================



.. code:: ipython3

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


.. parsed-literal::

    /home/andres-python/anaconda3/lib/python3.5/site-packages/scipy/integrate/odepack.py:218: ODEintWarning: Excess work done on this call (perhaps wrong Dfun type). Run with full_output = 1 to get quantitative information.
      warnings.warn(warning_msg, ODEintWarning)


.. parsed-literal::

    parameter values are  [  4.18843852e-06   1.85225631e-02]



.. image:: output_27_2.png


.. code:: ipython3

    #Di = 6e-13
    #kf = 1.91e-5 #m/s

Conclusion
==========

Pour conclure, le but de ce notebook est de modéliser l'extraction
supercritique en suivant le modèle de Reverchon. Pour cela il fallait
résoudre des équations différentielles ordinaires. Nous avons ensuite
mis les résultats obtenus sous forme de graphique où C = f(t). Nous
pouvons changer certaines valeurs afin de voir si cela influe ou non sur
l'efficacité de l'extraction. Nous pouvons également ajuster le
paramètres aux valeurs expérimentales afin d'avoir une courbe se
rapprochant de la pratique. Cependant, il est important de modéliser
l'extraction afin de voir l'efficacité des extractions sans faire des
expériences.

Referencias
===========

-  Evaluation of models for supercritical fluid extraction, Amit Rai,
   Kumargaurao D. Punase, Bikash Mohanty, Ravindra Bhargava.
-  [1] E. Reverchon, Mathematical modelling of supercritical extraction
   of sage oil, AIChE J. 42 (1996) 1765–1771.
   https://onlinelibrary.wiley.com/doi/pdf/10.1002/aic.690420627

-  [2] Amit Rai, Kumargaurao D.Punase, Bikash Mohanty, Ravindra
   Bhargava, Evaluation of models for supercritical fluid extraction,
   International Journal of Heat and Mass Transfer Volume 72, May 2014,
   Pages 274-287.
   https://www.sciencedirect.com/science/article/pii/S0017931014000398
