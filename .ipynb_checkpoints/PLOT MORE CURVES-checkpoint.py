# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:22:23 2019

@author: James Joseph
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model(y,t):
    dy_dt=-k*y
    return dy_dt

y0=5
t=np.linspace(0,20)

for k in np.linspace(0.1,0.9,10):
    y=odeint(model,y0,t)
    plt.plot(t,y,'g',linewidth=1,label=k)
    
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
