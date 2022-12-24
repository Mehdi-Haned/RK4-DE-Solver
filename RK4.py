import numpy as np
import matplotlib.pyplot as plt

# Note: In this example I showcase a seconds order system of ODEs using RK4 but in principle you can do this
# for an nth order system. I only chose 2nd order because it is more convinent to graph, analyze, and debug with

# Note: See overview.ipynb for error comparison with scipy.integrate.odeint and slightly different use of parameters

tmin, tmax, dt = 0, 10, 0.01

#Consider the driven damped pendulum for this example

def f(L, t):
    """
    System of differential equations you want to solve.
    t is your time series array you want to solve on
    L is the list containing all of your variables

    returns a list of functions with the i'th function corresponding to 
    the derivative of the ith variable
    """
    #parametreters of the system:
    om = 2*np.pi
    om_0 = 1.5*om
    b = om_0 * 0.25
    gamma = 1.073

    x1, x2 = L
    dxdt = [x2, -2*b*x2 - (om_0**2)*np.sin(x1) + gamma*(om_0**2)*np.cos(om*t)]
    return dxdt

ics = [0, 1] #initial conditions for x1 and x2 respectively

t = np.arange(tmin, tmax, dt) #time range we are solving on

def RK4(f, ic, time):
    """
    takes a list of functions f, a list of initial conditions ic, and
    a numpy array for all the times you want solutions on

    if n is how many variables you have, and there are t* time points
    then the output will be a (t*, n) sized numpy array where every row
    is the solution at the corresponding time
    """
    n = len(ic)
    dt = (tmax - tmin)/len(time) #timestep; here so that you only need to input the time array
    ans = []
    L = ic
    for t in time:
        ans.append(L)
        k1 = [dt * xi for xi in f(L, t)]
        k2 = list(map(lambda x: dt*x, f([L[i] + 0.5*k1[i] for i in range(n)], t+0.5*dt)))
        k3 = list(map(lambda x: dt*x, f([L[i] + 0.5*k2[i] for i in range(n)], t+0.5*dt)))
        k4 = list(map(lambda x: dt*x, f([L[i] + k3[i] for i in range(n)], t+dt)))
        L = [L[i] + 1/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(n)]
    
    return np.array(ans, dtype="float") #Turning everything into a numpy array

fig, axs = plt.subplots(2)

data = RK4(f, ics, t)

axs[0].plot(t, data[:,0], c="b") 
axs[0].set_xlabel("t")
axs[0].set_ylabel("x1")
axs[0].grid()

axs[1].plot(t, data[:,1], c="r")
axs[1].set_xlabel("t")
axs[1].set_ylabel("x2")
axs[1].grid()

plt.show()