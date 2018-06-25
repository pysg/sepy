
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

MODELE DE LACK
==============

INTRODUCTION
============

Le modèle de Lack va être utilisé afin de modéliser l'extraction
supercritique sans utiliser des équations différentielles ordinaires, en
effet il existe une solution analytique. Il est basé sur l'écoulement
piston, et sur les équations de transfert de matière. Le but est donc de
montrer les équations analytiques permettant de résoudre le modèle puis
d'ajuster les différentes expériences à ce modèle afin de voir les
valeurs qui vont changer.

EQUATIONS UTILISEES
===================

-  Premièrement l'équation de transfert de matière :
   :math:`p_{f}.E.\frac{\partial y}{\partial t}+ p_{f}.U.\frac{\partial y}{\partial h} = J(x,y)`
-  Regardons les équations utilisées par le modèle de Lack avec la
   solution analytique :

   -  :math:`t_{1} = \frac{xo-xk}{yAyr}`
   -  :math:`t_{2} = \frac{t1 +xk}{yAyr}`
   -  :math:`z_{k} = \frac{xk}{A.xo} . ln \frac{(xo.e^\frac{y A yr}{xk} (t-t1) - xk)}{xo-xk}`
   -  si t<t1 et t<t2 : :math:`e = gamma . yr. tao . (1- exp(- A)`
   -  si t>t1 et t<t2 :
      :math:`z_{k} = \frac{xk}{A.xo} . ln \frac{(xo.e^\frac{y A yr}{xk} (t-t1) - xk)}{xo-xk}`
      et :math:`e = y . yr . (t - t_{1}. exp(- A . (1 - zk)`
   -  si t> t2 :
      :math:`e = xo - \frac{xk}{A}.log(1 + xk / xo * (exp(\frac{xo}{xk} * A) - 1).exp(\frac{y A  yr}{xk}.(tao1 - tao)`

EXPLICATIONS
============

-  Tout d'abord, il faut entrer les paramètres principaux tels que la
   valeur de gamma (voir nomenclature), du yr et du contenu de l'extrait
   à l'état initial dans la phase solide.
-  Ensuite rentrer les équations du modèle (tao1, tao2, et zk)
-  Il faut ensuite définir une fonction rendement prenant comme valeur
   xk et A (cf nomenclature), afin de calculer le rendement, cependant,
   le temps est défini et les équations changent lorsque le temps
   change, les équations vont donc changer en fonction de l'intervalle
   de temps.
-  On obtient donc les résultats du rendement
-  On peut enfin réaliser les graphiques afin de voir l'efficacité de
   l'extraction.

.. code:: ipython3

    xo = 0.5
    gamma = 0.8
    yr = 0.1 #kg kgCO2 −1
    TAO = 15

.. code:: ipython3

    # en la matriz parametros se define cada conbinación (xk, A) como una fila de la matriz
    parametros = np.array([[0.1, 1],[0.3, 2],[0.4, 4]])
    parametros




.. parsed-literal::

    array([[0.1, 1. ],
           [0.3, 2. ],
           [0.4, 4. ]])



.. code:: ipython3

    def intervalosExtraccion(tao, xk, A):
        tao1 = (xo - xk) / (gamma * A * yr)
        tao2 = tao1 + xk / (gamma * A * yr) * np.log(xk / xo + (1 - xk / xo) * np.exp(xo / xk * A))
        zk = xk / (A * xo) * np.log((xo * np.exp(gamma * A * yr / xk * (tao - tao1)) - xk) / (xo - xk))
    
        return tao1, tao2, zk


.. code:: ipython3

    def rendimiento(tao, xk, A):
        
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
        

.. code:: ipython3

    caso1 = [[rendimiento(tao, xk, A) for tao in np.linspace(0,TAO)] for xk, A in parametros]
    caso1


.. parsed-literal::

    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2


.. parsed-literal::

    C:\Users\Agnès\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in log
      after removing the cwd from sys.path.




.. parsed-literal::

    [[0.0,
      0.015480503481515702,
      0.030961006963031404,
      0.046441510444547106,
      0.06192201392606281,
      0.07740251740757852,
      0.09288302088909421,
      0.1083635243706099,
      0.12384402785212562,
      0.1393245313336413,
      0.15480503481515703,
      0.17028553829667273,
      0.18576604177818842,
      0.20124654525970415,
      0.2167270487412198,
      0.23220755222273556,
      0.24768805570425123,
      0.26316226702728307,
      0.27855816883989115,
      0.29378225368823213,
      0.30876157355231604,
      0.32343654083397494,
      0.33775619815162916,
      0.35167527000149623,
      0.3651522290408746,
      0.3781479701886501,
      0.39062486466268487,
      0.402546060258831,
      0.4138749462390364,
      0.42457473119368283,
      0.43460810015827545,
      0.44393692830690873,
      0.45252203554707215,
      0.460322970897461,
      0.46729781856882635,
      0.4734030197400882,
      0.4785933211346348,
      0.4828693172545476,
      0.4863489367200255,
      0.489159809316668,
      0.49141660746817667,
      0.49321943250109174,
      0.49465369004190646,
      0.4957909366284409,
      0.49669026865739097,
      0.4973999390469195,
      0.49795899361401663,
      0.4983988041113998,
      0.49874443572454247,
      0.499015826796625],
     [0.0,
      0.021175462451348267,
      0.042350924902696534,
      0.0635263873540448,
      0.08470184980539307,
      0.10586240724665469,
      0.12693119197306726,
      0.14783738881474864,
      0.1685261464121838,
      0.18894922443715043,
      0.20906060982884964,
      0.22881410899010135,
      0.24816182896433522,
      0.26705309024678237,
      0.28543355559493905,
      0.30324446322126325,
      0.32042190154796807,
      0.33689608714873753,
      0.35259062034949884,
      0.36742169991480716,
      0.38130372981839566,
      0.3941960908220593,
      0.406090123152443,
      0.41699047089218877,
      0.4269147321107626,
      0.43589255261409354,
      0.44396415365496233,
      0.45117844370314303,
      0.457590892189559,
      0.46326134405019936,
      0.46825193197286874,
      0.4726252062684879,
      0.4764425591202635,
      0.479762978345391,
      0.48264213103884324,
      0.48513175212947535,
      0.48727929723524216,
      0.48912781199630445,
      0.4907159692896542,
      0.4920782292572582,
      0.49324508303711023,
      0.4942433480260757,
      0.49509648943078877,
      0.4958249491782092,
      0.49644646866322384,
      0.49697639621796236,
      0.4974279736363982,
      0.4978125986875685,
      0.49814006243866604,
      0.49841876152514436],
     [0.0,
      0.024041249659867737,
      0.048071045167731294,
      0.07204855156465495,
      0.09594482230629861,
      0.11973478671205953,
      0.14339127944692867,
      0.16688275925732202,
      0.19017165999482702,
      0.21321275986469856,
      0.23595135235832557,
      0.2583210946009588,
      0.28024142838261135,
      0.301614464669515,
      0.3223212067750781,
      0.3422169643639321,
      0.36113676813322215,
      0.378934489364718,
      0.39547915297515,
      0.41065899741032463,
      0.4243921053446273,
      0.4366350926454827,
      0.4473880158319006,
      0.4566943655439216,
      0.4646361475721564,
      0.4713251864255304,
      0.4768924922334224,
      0.48147761541705325,
      0.4852194768336529,
      0.48824948300646315,
      0.4906870935567196,
      0.49263756209496956,
      0.49419135465912917,
      0.49542471233898844,
      0.49640089297803586,
      0.4971717372028536,
      0.49777931571338624,
      0.49825750797968005,
      0.49863343155527706,
      0.49892868792607403,
      0.4991604198386353,
      0.4993421914103874,
      0.4994847102707726,
      0.4995964137138029,
      0.49968394057592386,
      0.49975250873476684,
      0.49980621564599303,
      0.49984827671470905,
      0.49988121381246126,
      0.4999070040309815]]



.. code:: ipython3

    caso1[2]




.. parsed-literal::

    [0.0,
     0.024041249659867737,
     0.048071045167731294,
     0.07204855156465495,
     0.09594482230629861,
     0.11973478671205953,
     0.14339127944692867,
     0.16688275925732202,
     0.19017165999482702,
     0.21321275986469856,
     0.23595135235832557,
     0.2583210946009588,
     0.28024142838261135,
     0.301614464669515,
     0.3223212067750781,
     0.3422169643639321,
     0.36113676813322215,
     0.378934489364718,
     0.39547915297515,
     0.41065899741032463,
     0.4243921053446273,
     0.4366350926454827,
     0.4473880158319006,
     0.4566943655439216,
     0.4646361475721564,
     0.4713251864255304,
     0.4768924922334224,
     0.48147761541705325,
     0.4852194768336529,
     0.48824948300646315,
     0.4906870935567196,
     0.49263756209496956,
     0.49419135465912917,
     0.49542471233898844,
     0.49640089297803586,
     0.4971717372028536,
     0.49777931571338624,
     0.49825750797968005,
     0.49863343155527706,
     0.49892868792607403,
     0.4991604198386353,
     0.4993421914103874,
     0.4994847102707726,
     0.4995964137138029,
     0.49968394057592386,
     0.49975250873476684,
     0.49980621564599303,
     0.49984827671470905,
     0.49988121381246126,
     0.4999070040309815]



.. code:: ipython3

    caso2 = [[rendimiento(tao, 0.1, 2) for tao in np.linspace(0,TAO)] for xk, A in parametros]



.. parsed-literal::

    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2


.. parsed-literal::

    C:\Users\Agnès\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in log
      after removing the cwd from sys.path.


.. code:: ipython3

    caso2[2]




.. parsed-literal::

    [0.0,
     0.021175462451348267,
     0.042350924902696534,
     0.0635263873540448,
     0.08470184980539307,
     0.10587731225674134,
     0.1270527747080896,
     0.14822823715943784,
     0.16940369961078613,
     0.19056244307692105,
     0.2115985523819776,
     0.23241231901077886,
     0.25293204075922016,
     0.2730979737195935,
     0.29285509070985855,
     0.3121492447301399,
     0.33092483248166144,
     0.34912318936363385,
     0.3666813631548401,
     0.38353108568240424,
     0.3995978406227672,
     0.4147999649617682,
     0.4290477427597768,
     0.4422424617966035,
     0.4542754106011764,
     0.4650267973875253,
     0.47436457463264287,
     0.4821431540722443,
     0.48820199713032947,
     0.49244677140858195,
     0.49523801600034545,
     0.49702868120368104,
     0.49815846686029064,
     0.4988635736049403,
     0.49930059390946535,
     0.4995702784096971,
     0.4997362495881598,
     0.4998382215558004,
     0.4999008079382453,
     0.4999391965786946,
     0.4999627338446445,
     0.49997716181473445,
     0.4999860046316513,
     0.49999142385322737,
     0.4999947447792546,
     0.499996779790427,
     0.4999980267873645,
     0.4999987909018916,
     0.4999992591199328,
     0.4999995460233749]



.. code:: ipython3

    caso3 = [[rendimiento(tao, 0.4, 2) for tao in np.linspace(0,TAO)] for xk, A in parametros]



.. parsed-literal::

    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2


.. parsed-literal::

    C:\Users\Agnès\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in log
      after removing the cwd from sys.path.


.. code:: ipython3

    caso3[2]




.. parsed-literal::

    [0.0,
     0.021175462451348267,
     0.042350924902696534,
     0.06350297259732109,
     0.08453257906194382,
     0.10536364658406067,
     0.12594149887299486,
     0.14621874129133108,
     0.16614990905585156,
     0.18568887260168956,
     0.2047873122796894,
     0.22339366018347454,
     0.2414524876088145,
     0.25891535727689663,
     0.2757492150470181,
     0.2919245718380339,
     0.3074150753126732,
     0.3221981730973905,
     0.3362557129570245,
     0.3495744476870033,
     0.3621464158492363,
     0.37396917557071563,
     0.38504587700153353,
     0.3953851689058501,
     0.40500094521559876,
     0.4139119471162316,
     0.42214124435860356,
     0.4297156252542512,
     0.43666492781561683,
     0.4430213447281909,
     0.44881873260340144,
     0.45409195181668727,
     0.4588762578659975,
     0.4632067592823114,
     0.4671179512930065,
     0.47064332914770646,
     0.47381508056817023,
     0.4766638533260055,
     0.47921859150137464,
     0.4815064324543937,
     0.4835526558121042,
     0.4853806756745014,
     0.4870120676075547,
     0.4884666226679049,
     0.48976242156453903,
     0.49091592300550474,
     0.4919420612271014,
     0.49285434860689475,
     0.4936649800876988,
     0.49438493686990626]



.. code:: ipython3

    caso11 = [[rendimiento(tao, 0.3, 2) for tao in np.linspace(0,TAO)] for xk, A in parametros]



.. parsed-literal::

    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2


.. parsed-literal::

    C:\Users\Agnès\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in log
      after removing the cwd from sys.path.


.. code:: ipython3

    caso11[2]




.. parsed-literal::

    [0.0,
     0.021175462451348267,
     0.042350924902696534,
     0.0635263873540448,
     0.08470184980539307,
     0.10586240724665469,
     0.12693119197306726,
     0.14783738881474864,
     0.1685261464121838,
     0.18894922443715043,
     0.20906060982884964,
     0.22881410899010135,
     0.24816182896433522,
     0.26705309024678237,
     0.28543355559493905,
     0.30324446322126325,
     0.32042190154796807,
     0.33689608714873753,
     0.35259062034949884,
     0.36742169991480716,
     0.38130372981839566,
     0.3941960908220593,
     0.406090123152443,
     0.41699047089218877,
     0.4269147321107626,
     0.43589255261409354,
     0.44396415365496233,
     0.45117844370314303,
     0.457590892189559,
     0.46326134405019936,
     0.46825193197286874,
     0.4726252062684879,
     0.4764425591202635,
     0.479762978345391,
     0.48264213103884324,
     0.48513175212947535,
     0.48727929723524216,
     0.48912781199630445,
     0.4907159692896542,
     0.4920782292572582,
     0.49324508303711023,
     0.4942433480260757,
     0.49509648943078877,
     0.4958249491782092,
     0.49644646866322384,
     0.49697639621796236,
     0.4974279736363982,
     0.4978125986875685,
     0.49814006243866604,
     0.49841876152514436]



.. code:: ipython3

    caso22 = [[rendimiento(tao, 0.3, 1) for tao in np.linspace(0,TAO)] for xk, A in parametros]



.. parsed-literal::

    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2


.. parsed-literal::

    C:\Users\Agnès\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in log
      after removing the cwd from sys.path.


.. code:: ipython3

    caso22[2]




.. parsed-literal::

    [0.0,
     0.015480503481515702,
     0.030961006963031404,
     0.046441510444547106,
     0.06192201392606281,
     0.07740251740757852,
     0.09288302088909421,
     0.1083635243706099,
     0.12384402785212562,
     0.1393163850771649,
     0.15472400277901396,
     0.17000390606445162,
     0.1851050494996573,
     0.19998398178759502,
     0.21460217683511587,
     0.22892432427524176,
     0.2429171818777176,
     0.256548772838341,
     0.2697878030122515,
     0.2826032228902517,
     0.29496388730463446,
     0.3068394018692976,
     0.3182169273086793,
     0.3290989447206328,
     0.33948993217002776,
     0.3493959162718351,
     0.3588243949425004,
     0.367784243305447,
     0.37628560519873167,
     0.3843397730049986,
     0.39195905868361053,
     0.3991566589328712,
     0.4059465173534512,
     0.4123431863344058,
     0.4183616911562489,
     0.42401739851937914,
     0.4293258913795972,
     0.4343028516238834,
     0.4389639517659152,
     0.4433247564965974,
     0.4474006346020073,
     0.45120668146852694,
     0.45475765213851016,
     0.4580679046629286,
     0.4611513533209832,
     0.4640214311397041,
     0.4666910610466474,
     0.4691726349224559,
     0.47147799978317007,
     0.47361845031033867]



.. code:: ipython3

    caso33 = [[rendimiento(tao, 0.3, 4) for tao in np.linspace(0,TAO)] for xk, A in parametros]



.. parsed-literal::

    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao < tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao1 and tao < tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2
    tao >= tao2


.. parsed-literal::

    C:\Users\Agnès\Anaconda3\lib\site-packages\ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in log
      after removing the cwd from sys.path.


.. code:: ipython3

    caso33




.. parsed-literal::

    [[0.0,
      0.024041249659867737,
      0.048082499319735475,
      0.07211552176266275,
      0.09610561668412693,
      0.1200239506791445,
      0.14384530041379165,
      0.1675428527052751,
      0.19108589401139028,
      0.2144381377979206,
      0.23755611234399351,
      0.2603873848125538,
      0.2828684919363397,
      0.30492247141412315,
      0.32645588666728914,
      0.34735522422443205,
      0.36748252175818197,
      0.38667005664842596,
      0.40471388955358684,
      0.4213660138137079,
      0.43632658333647556,
      0.44934844592443113,
      0.4603805933904678,
      0.46948553729951736,
      0.4768162592708929,
      0.4825884242637978,
      0.48704696195626984,
      0.4904364279978098,
      0.49298042215044546,
      0.4948708228856948,
      0.4962648073984426,
      0.49728679434964007,
      0.4980328228279217,
      0.4985756698783728,
      0.49896974603445643,
      0.49925533260257154,
      0.499462038464087,
      0.4996115152217228,
      0.499719536370824,
      0.4997975619548788,
      0.49985390179656397,
      0.49989457290976796,
      0.49992392765832455,
      0.49994511196468905,
      0.49996039851183477,
      0.4999714285026688,
      0.4999793867912866,
      0.4999851286024885,
      0.49998927114604386,
      0.49999225981138556],
     [0.0,
      0.024041249659867737,
      0.048082499319735475,
      0.07211552176266275,
      0.09610561668412693,
      0.1200239506791445,
      0.14384530041379165,
      0.1675428527052751,
      0.19108589401139028,
      0.2144381377979206,
      0.23755611234399351,
      0.2603873848125538,
      0.2828684919363397,
      0.30492247141412315,
      0.32645588666728914,
      0.34735522422443205,
      0.36748252175818197,
      0.38667005664842596,
      0.40471388955358684,
      0.4213660138137079,
      0.43632658333647556,
      0.44934844592443113,
      0.4603805933904678,
      0.46948553729951736,
      0.4768162592708929,
      0.4825884242637978,
      0.48704696195626984,
      0.4904364279978098,
      0.49298042215044546,
      0.4948708228856948,
      0.4962648073984426,
      0.49728679434964007,
      0.4980328228279217,
      0.4985756698783728,
      0.49896974603445643,
      0.49925533260257154,
      0.499462038464087,
      0.4996115152217228,
      0.499719536370824,
      0.4997975619548788,
      0.49985390179656397,
      0.49989457290976796,
      0.49992392765832455,
      0.49994511196468905,
      0.49996039851183477,
      0.4999714285026688,
      0.4999793867912866,
      0.4999851286024885,
      0.49998927114604386,
      0.49999225981138556],
     [0.0,
      0.024041249659867737,
      0.048082499319735475,
      0.07211552176266275,
      0.09610561668412693,
      0.1200239506791445,
      0.14384530041379165,
      0.1675428527052751,
      0.19108589401139028,
      0.2144381377979206,
      0.23755611234399351,
      0.2603873848125538,
      0.2828684919363397,
      0.30492247141412315,
      0.32645588666728914,
      0.34735522422443205,
      0.36748252175818197,
      0.38667005664842596,
      0.40471388955358684,
      0.4213660138137079,
      0.43632658333647556,
      0.44934844592443113,
      0.4603805933904678,
      0.46948553729951736,
      0.4768162592708929,
      0.4825884242637978,
      0.48704696195626984,
      0.4904364279978098,
      0.49298042215044546,
      0.4948708228856948,
      0.4962648073984426,
      0.49728679434964007,
      0.4980328228279217,
      0.4985756698783728,
      0.49896974603445643,
      0.49925533260257154,
      0.499462038464087,
      0.4996115152217228,
      0.499719536370824,
      0.4997975619548788,
      0.49985390179656397,
      0.49989457290976796,
      0.49992392765832455,
      0.49994511196468905,
      0.49996039851183477,
      0.4999714285026688,
      0.4999793867912866,
      0.4999851286024885,
      0.49998927114604386,
      0.49999225981138556]]



GRAPHIQUE
=========

Ici, nous avons les graphiques réalisés pour xk = 0.3 et A variant
(premier graphique). Tandis que le deuxième graphique c'est xk qui va
varier et A va être égale à 2. - Nous remarquons dans le premier
graphique que le rendement de l'extraction est supérieur à temps égale
pour A = 4, donc nous pouvons dire que l'effet de transfert de masse
influence le rendement de l'extraction, plus cette valeur est faible
moins le rendement sera bon à temps identique, en effet on remarque le
maximum est déjà atteint pour A = 4 à t = 10 (rendement = 0.5) tandis
que pour A = 2 et A = 1, nous sommes respectivement à 0.46 et 0.4. - Si
nous regardons le deuxième graphique, nous pouvons remarquer que plus xk
(contenu minimum de l'extrait dans la phase solide) est petit plus le
rendement de l'extraction sera élevée. Cependant les valeurs de xk sont
assez proches : 0.1, 0.3, 0.4 donc les courbes ne sont pas réelllement
distinctes à part pour xk = 0.4 où il n'y a aucun croisement
contrairement à xk = 0.1 et xk = 0.3

.. code:: ipython3

    Tao = np.linspace(0,TAO)
    
    plt.plot(Tao, caso11[2],label="A=2")
    plt.plot(Tao,caso22[2],label="A=1")
    plt.plot(Tao,caso33[2],label="A=4")
    plt.title("Modelo Lack")
    plt.xlabel(" $tao $ ")
    plt.ylabel("Rendimiento e")
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x277de07ae80>




.. image:: output_22_1.png


.. code:: ipython3

    Tao = np.linspace(0,TAO)
    
    plt.plot(Tao, caso1[2],label="xk=0.3")
    plt.plot(Tao,caso2[2],label="xk=0.1")
    plt.plot(Tao,caso3[2],label="xk=0.4")
    plt.title("Modelo Lack")
    plt.xlabel(" $tao $ ")
    plt.ylabel("Rendimiento e")
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x277dddac710>




.. image:: output_23_1.png


.. code:: ipython3

    def graphique(tao,xk,A):
        caso = [[rendimiento(tao, xk, A) for tao in np.linspace(0,TAO)] for xk, A in parametros]
        caso
        plt.plot(Tao, caso[2],label="A")
        plt.title("Modelo Lack")
        plt.xlabel(" $tao $ ")
        plt.ylabel("Rendimiento e")
        plt.legend()
        return 

AJUSTEMENT
==========

.. code:: ipython3

    xk = 0.1
    A = 2 
    
    #Datos experimentales
    x_data = np.linspace(0,9,10)#temps
    y_data = array([ 0.00429861,  0.00907806,  0.01142553,  0.01471523,  0.01585107,0.01674278,  0.01744284,  0.01799243,  0.01860349,  0.01902855])#concentration
    
    
    def lack(tao,xk,A):
        
        tao1 = (xo - xk) / (gamma * A * yr)
        tao2 = tao1 + xk / (gamma * A * yr) * np.log(xk / xo + (1 - xk / xo) * np.exp(xo / xk * A))
        zk = xk / (A * xo) * np.log((xo * np.exp(gamma * A * yr / xk * (tao - tao1)) - xk) / (xo - xk))
        
        return [tao1,tao2,zk]  
    
    def my_ls_func(tao,xk,A):
        f2 = lambda tao, xk: lack(tao,xk,A)
        # calcular el valor de la ecuación diferencial en cada punto
        r = integrate.odeint(f2, y0, xk)
        return r[:,1]
    
    # def rendement
    
    def f_resid(p):
        # definir la función de minimos cuadrados para cada valor de y"""
        
        return y_data - my_ls_func(x_data,A,xk)
    
    #resolver el problema de optimización
    # guess = [0.2, 0.3] #valores inicales para los parámetros # funcionan bien
    guess = [0.2, 0.3] #valores inicales para los parámetros # funcionan bien
    #y0 = [1,0,0] #valores inciales para el sistema de ODEs
    y0 = [1,0] #valores inciales para el sistema de ODEs
    (e,kvg) = optimize.leastsq(f_resid, guess) #get params
    
    print("parameter values are ",e)
    
    # interpolar los valores de las ODEs usando splines
    xeval = np.linspace(min(x_data), max(x_data),30) 
    gls = interpolate.UnivariateSpline(xeval, my_ls_func(xeval,xk,A), k=3, s=0)
    
    
    xeval = np.linspace(min(x_data), max(x_data), 200)
    #Gráficar los resultados
    pp.plot(x_data, y_data,'.r',xeval,gls(xeval),'-b')
    pp.xlabel('t [=] min',{"fontsize":16})
    pp.ylabel("C",{"fontsize":16})
    pp.legend(('Datos','Modelo'),loc=0)
    pp.show()


.. parsed-literal::

    parameter values are  [0.2]


::


    ---------------------------------------------------------------------------

    error                                     Traceback (most recent call last)

    <ipython-input-127-dbc8a8fee794> in <module>()
         39 # interpolar los valores de las ODEs usando splines
         40 xeval = np.linspace(min(x_data), max(x_data),30)
    ---> 41 gls = interpolate.UnivariateSpline(xeval, my_ls_func(xeval,xk,A), k=3, s=0)
         42 
         43 


    ~\Anaconda3\lib\site-packages\scipy\interpolate\fitpack2.py in __init__(self, x, y, w, bbox, k, s, ext, check_finite)
        183 
        184         data = dfitpack.fpcurf0(x,y,k,w=w,
    --> 185                                 xb=bbox[0],xe=bbox[1],s=s)
        186         if data[-1] == 1:
        187             # nest too small, setting to maximum bound


    error: failed in converting 2nd argument `y' of dfitpack.fpcurf0 to C/Fortran array



CONCLUSION
==========

Pour conclure, nous avons pu voir grâce à ce modèle les effets de deux
paramètres qui sont l'effet de transfert de masse (A) et le contenu
minimum de l'extrait dans la phase solide (xk). Cela nous a permis de
voir que plus A était élevée plus l'extraction était efficace tandis que
plus xk était faible plus le rendement était élevée.

NOMENCLATURE
============

-  :math:`a_0` : Surface spécifique , m2 m−3
-  A : (=tr /tf ), inverse de l'effet de transfert de masse de la phase
   fluide sur le taux d'extraction
-  e : rendement de l'extraction, kg kgfeed−1
-  h : coordonnée axiale le long du lit d'extraction, m
-  H : hauteur du lit d'extraction, m
-  J : taux du transfère de matière, kg m−3 s−1
-  kf: coefficient de transfère de matière phase liquide , m s−1
-  ks: coefficient de transfère de matière phase solide, m s−1
-  K : coefficient de partition, kgfeed kgCO2−1
-  n : nombre de mélangeur
-  q : (=q’t), rapport solvant-alimentation, kgCO2 kgfeed−1
-  q’: débit spécifique, kgCO2 s−1 kgfeed−1
-  t : durée de l'extraction, s
-  tf: (=/kf a0), temps caractéristique de transfert de matière dans le
   fluide, s
-  tr: temps de résidence, s
-  ts: (= (1-)/kf a0), temps caractéristique de diffusion interne,s
-  U : vélocité superficielle, m s−1
-  v : facteur de taux de transfert de matière dans le modèle de Lack
-  x : contenu de l'extrait dans la phase solide, kg kgsolid−1
-  xd: teneur critique de l'extrait dans la phase solide
-  xB: contenu de l'extrait au sein des cellules brisées,
-  kg: kgsolid−1
-  xI: contenu de l'extrait au sein des cellules intactes,
-  kg: kgsolid−1
-  xk: contenu minimal de l'extrait dans la phase solide pour lequel y+
   = yr , kg kgsolid−1
-  xu: contenu de l'extrait dans les particules placées dans le réacteur
   , kg kgsolid−1
-  xuI: contenu de l'extrait dans les cellules intactes,kg kgsolid−1
-  y :contenu de l'extrait dans la phase liquid, kg kgCO2−1
-  yeq: concentration de la phase liquide à l'équilibre, kg kgCO2−1
-  yr: solubilité de l'extrait dans le CO2, kg kgCO2−1
-  z :(h/H), coordonnée axiale sans dimension
-  zk:coordonnée axiale sans dimension où x = xk
-  ε : porosité du lit (degré de vide)
-  gamma : Rapport massique du CO2 dans le volume de vide du lit
   d'extraction, kgCO2 kgfeed−1
-  pf: densité du CO2, kgCO2 m−3
-  ps: densité des particules, kg m−3
-  tao: (=t/tr ),temps (sans unité)

BIBLIOGRAPHIE
=============

-  Broken-and-intact cell model for supercritical fluid extraction: Its
   origin and limits / Helena Sovová / Institute of Chemical Process
   Fundamentals of the Czech Academy of Sciences, v. v. i., Rozvojová
   135, 16502 Prague, Czechia


.. code:: ipython3

    # CER
    m = Q*Y*(1 - np.exp(-Z))*t
    
    #FER
    m = Q*Y*(t - tcer*np.exp(Zw-Z))
    
    #DC
    m = msi*(Xo - Y/W * np.log(1 + (np.exp(W*Xo/Y) - 1) * np.exp(W*Q*(tcer-t)/msi) * Xk/Xo))
    
    
    Z = (msi*kya*rho) / (Q*)







