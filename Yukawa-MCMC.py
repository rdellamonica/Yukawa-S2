import pygro
import numpy as np
import emcee
import pickle
import time
import datetime
from multiprocessing import Pool
from scipy.interpolate import interp1d

# Data from Gillessen et al. (2017) for the astrometric positions of the S2 star
keckvlt_data = np.genfromtxt("tab_gillessen.csv", delimiter = ";")
t_dat = keckvlt_data[:,0]
x_dat = keckvlt_data[:,1]/1000
errx = keckvlt_data[:,2]/1000
y_dat = keckvlt_data[:,3]/1000
erry = keckvlt_data[:,4]/1000

# Data from Gillessen et al. (2017) for the radial velocities of the S2 star
v_data = np.genfromtxt("radial_v.csv", delimiter = ";")
t_dat_v = v_data[:,0]
v_dat = v_data[:,1]
errv = v_data[:,2]

# Constants
G_N = 6.67430e-11 #N*m^2/kg^2
c = 2.99792458e+8 #m/s
yr_s = 3600*24*365.25 # Seconds in a Julian Year
Msun = 1.98847e+30 #Kg
kpc = 3.086e+19 #m
AU = 149597870691 #m  

# Fit Parameters from GRAVITY as priors
M_bh_s = 4.261e+6*Msun 
D_s = 8.2467*kpc
r_g = G_N*M_bh_s/(c**2)
T_s = 16.0455
Tp_s =  2018.379
a_s = 0.125058
e_s = 0.884649
inc_s = np.deg2rad(134.567)
Omega_s = np.deg2rad(228.171)
omega_s = np.deg2rad(66.263)
vLSR_s = -1.6

# Reference frame priors from Plewa et al. (2015)
x0_s = -0.0002
y0_s = -0.0001
vx0_s = 0.00002
vy0_s = 0.00006


#######################
# ORBITAL INTEGRATION #
#######################

def yukawa_orbit(params):
    # Incoming set of parameters from the Markov Chain
    M_bh, D, T, Tp, a, e, inc, Omega, omega, xS0, yS0, vxS0, vyS0, v_LSR, delta, lamb = params
    

    t0 = Tp-T/2
    r_g = G_N*M_bh/(c**2)
    
    # Rescaling lambda in geometrical units
    lamb = lamb/r_g

    # Setting values for delta and lambda in the Metric object
    metric.set_constant(delta = delta, lamb = lamb)


    # Thiele-Innes elements
    A = np.cos(omega)*np.cos(Omega)-np.sin(omega)*np.sin(Omega)*np.cos(inc)
    B = np.cos(omega)*np.sin(Omega)+np.sin(omega)*np.cos(Omega)*np.cos(inc)
    C = -np.sin(omega)*np.sin(inc)
    F = -np.sin(omega)*np.cos(Omega)-np.cos(omega)*np.sin(Omega)*np.cos(inc)
    G = -np.sin(omega)*np.sin(Omega)+np.cos(omega)*np.cos(Omega)*np.cos(inc)
    H = -np.cos(omega)*np.sin(inc)

    # Initial conditions from Keplerian elements in celestial coordiantes
    M = 2*np.pi/T*(t0-Tp)
    E = np.pi
    phi = np.pi
    r = a*(1-e**2)/(1+e*np.cos(phi))
    vx_orb = -2*np.pi/T*a**2/(r)*np.sin(E)
    vy_orb = 2*np.pi/T*a**2/(r)*np.sqrt(1-e**2)*np.cos(E)

    # Conversion from celestial coordinates to orbital plane coordinates in geometrical units
    r0 = r*D/3600*np.pi/180/r_g
    phi0 = phi

    x0 = r0*np.cos(phi0)
    y0 = r0*np.sin(phi0)
    vx0 = vx_orb*D/3600*np.pi/180*1/yr_s/c
    vy0 = vy_orb*D/3600*np.pi/180*1/yr_s/c

    vr0 = (x0*vx0+y0*vy0)/r0
    vphi0 = (vy0*x0-y0*vx0)/r0**2

    # Definition of the Geodesic objects
    # We use two different geodesics,
    # one for the forward integration form apocentre,
    # the other for the backward integration

    geo_fw = pygro.Geodesic("time-like", geo_engine, verbose = False)
    geo_fw.set_starting_point(0, r0, np.pi/2, phi0)
    u0 = geo_fw.get_initial_u0(vr0, 0, vphi0)
    geo_fw.initial_u = [u0, vr0, 0, vphi0]

    geo_bw = pygro.Geodesic("time-like", geo_engine, verbose = False)
    geo_bw.set_starting_point(0, r0, np.pi/2, phi0)
    u0 = geo_bw.get_initial_u0(vr0, 0, vphi0)
    geo_bw.initial_u = [u0, vr0, 0, vphi0]

    # Numerical integration
    geo_engine.integrate(geo_fw, 4e+7, initial_step = 1e+5, PrecisionGoal = 5, AccuracyGoal = 3, direction = "fw")
    geo_engine.integrate(geo_bw, 4e+7, initial_step = 1e+5, PrecisionGoal = 5, AccuracyGoal = 3, direction = "bw")

    # Merging of the forward and backward geodesics
    geo = pygro.Geodesic("time-like", geo_engine, verbose = False)
    geo.x = np.concatenate((np.flip(geo_bw.x, axis = 0), geo_fw.x))
    geo.u = np.concatenate((np.flip(geo_bw.u, axis = 0), geo_fw.u))
    geo.tau = np.concatenate((np.flip(geo_bw.tau, axis = 0), geo_fw.tau))

    # Projection and conversion back to celestial coordinates
    t = geo.x[:,0]*r_g/(yr_s*c)+t0
    x = geo.x[:,1]*np.cos(geo.x[:,3])
    y = geo.x[:,1]*np.sin(geo.x[:,3])
    z = (C*x+H*y)*r_g
    
    vx = np.cos(geo.x[:,3])*geo.u[:,1]-geo.x[:,1]*np.sin(geo.x[:,3])*geo.u[:,3]
    vy = np.sin(geo.x[:,3])*geo.u[:,1]+geo.x[:,1]*np.cos(geo.x[:,3])*geo.u[:,3]
    
    valpha = B*vx+G*vy
    vdelta = A*vx+F*vy
    vs = -(C*vx+H*vy)*c
    
    # Conversion from kinematic radial velocity to the observed radial velocity
    # making use of the special relativistic + gravitational redshift formmulas
    VR = []
    
    for i in range(len(x)):
        VR.append((1/(np.sqrt(-metric.g_f(geo.x[i])[0,0]))*np.sqrt(1-(vs[i]**2+valpha[i]**2+vdelta[i]**2)/c**2)/(1-vs[i]/c)-1)*c)

    VR = np.array(VR)
    
    x = x*r_g/D*180/np.pi*3600
    y = y*r_g/D*180/np.pi*3600

    alpha = B*x+G*y
    delta = A*x+F*y

    alpha_int = interp1d(t, alpha)
    delta_int = interp1d(t, delta)
    z_int = interp1d(t, z)
    VR_int = interp1d(t, VR)

    # Determination of the emission time from the time of arrival, modulated by the Roemer delay 
    t_em = t_dat-z_int(t_dat)/c/yr_s
    t_em_vr = t_dat_v-z_int(t_dat_v)/c/yr_s

    # Final determination of the orbit + zero-point offset and drift of the reference frame
    alpha = alpha_int(t_em)+xS0+vxS0*(t_dat-2009.2) 
    delta = delta_int(t_em)+yS0+vyS0*(t_dat-2009.2)
    vr =  VR_int(t_em_vr)/1000-v_LSR
    
    return alpha, delta,  vr

#############################
# PROBABILITY DISTRIBUTIONS #
#############################

# Gaussian priors log-likelihood
def logprob_prior_gauss(param, start, delta):
    return -(param-start)**2/(2*delta**2)

# Flat priors log-likelihood
def logprob_prior_flat(param, start, end):
    if start < param < end:
        return 0.0
    return -np.inf

# Log-likelihood for precession
def logprob_precession(params):
    M_bh, D, T, Tp, a, e, inc, Omega, omega, xS0, yS0, vxS0, vyS0, v_LSR, delta, lamb = params
    
    # Data from Gravity et al. 2020
    fs = 1.10
    deltafs = 0.19
    
    # Conversions
    r_g = G_N*M_bh/(c**2)
    a = a*np.pi/180*D/3600

    # Precessions    
    DPhiGR = 6*np.pi*G_N*M_bh/(a*c**2*(1-e**2))
    DPhiYukawa = DPhiGR/(delta+1)*(1+2*delta*G_N**2*M_bh**2/(3*a**2*c**4*(1-e**2)**2)-2*np.pi*delta*G_N**2*M_bh**2/(a*c**4*(1-e**2)*lamb)-3*delta*G_N*M_bh/(a*c**2*(1-e**2))-delta*G_N**2*M_bh**2/(6*c**4*(delta+1)*lamb**2)+delta*G_N*M_bh/(3*lamb*c**2))

    return 0.5*(fs-DPhiYukawa/DPhiGR)**2/deltafs**2

# Total log-likelihood 
def log_prob(params):
    prior = 0

    # Priors
    for i, param in enumerate(params):
        if i >= 14:
            prior += logprob_prior_flat(param, start_flat[i], end_flat[i]) # Flat priors on delta and lambda
        else:
            prior += logprob_prior_gauss(param, start_params[i], delta_params[i]) # Gaussian priors on delta and lambda

    if not np.isfinite(prior):
        return -np.inf
        
    alpha, delta, vr = yukawa_orbit(params)
    
    # Likelihood from comparison with data
    lp = 0
    
    for i in range(len(alpha)):
        lp += 0.5*((alpha[i]-x_dat[i])**2/errx[i]**2 + (delta[i]-y_dat[i])**2/erry[i]**2)

    for i in range(len(vr)):
        lp += 0.5*(vr[i]-v_dat[i])**2/errv[i]**2
        
    lp += logprob_precession(params)

    return prior-lp

# Starting values of the parameters for the MCMC
# delta and lambda start at 0 and 10 000 AU, respectively,
# the walkers than explore the parameter space thanks to the flat priors
start_params = [M_bh_s, D_s, T_s, Tp_s, a_s, e_s, inc_s, Omega_s, omega_s, x0_s, y0_s, vx0_s, vy0_s, vLSR_s, 0, 10000*AU]

# Flat priors only for delta and lambda
start_flat = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, -0.9, 100*AU]
end_flat = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 2, 25000*AU]

# Sigmas for gaussian priors (and for the samll ball in the parameter space in which we initialize the walkers)
delta_params = [0.012*1e+6**Msun, 0.0093*kpc, 0.0013, 0.00016, 0.041, 0.000066, np.deg2rad(0.033), np.deg2rad(0.031), np.deg2rad(0.031), 0.0002, 0.0002, 0.0001, 0.0001, 1.4, 0.05, 1000*AU]

# Setting up pygro's Metric engine and Geodesic Engine 
metric = pygro.default_metrics.Yukawa()
geo_engine = pygro.GeodesicEngine(metric)

# Setting MCMM parameters
nwalkers = 32                   # 32 walkers
ndim = len(start_params)        # 16 parameters
max_n = 500000                  # Stop if not converged before 500 000 steps

# We'll track how the average autocorrelation time estimate changes

index = 0
autocorr = np.empty(max_n)

# This will be useful to test convergence
old_tau = np.inf

# Initializing the walkers in a ball around the values from Gravity
start = np.reshape(start_params*nwalkers, (nwalkers, ndim))+np.random.randn(nwalkers, ndim)*np.array(delta_params)

#############################
#        MAIN LOOP          #
#############################

if __name__ == "__main__":
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool = pool)

        # Burn-in run
        print('Burning in ...')
        nsteps = 1000
        state = sampler.run_mcmc(start, nsteps, progress=True)
        state_sample = sampler.get_chain()

        #Actual run
        sampler.reset()
        nsteps = max_n

        # Starting from the final position in the burn-in chain, sample for 1000
        # steps. (rstate0 is the state of the internal random number generator)

        print("Running MCMC ...")

        # Now we'll sample for up to max_n steps

        for sample in sampler.sample(pos, iterations=n_steps, progress=True,rstate0=state):

            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy

            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence

            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)

            if converged:
                break

            old_tau = tau
            ninteration = sampler.iteration
            n = 100 * np.arange(1, index + 1)
            y = autocorr[:index]

        final_sample = sampler.get_chain()
        
        # Save the final sample of the MCMC

        with open("final_sample_" + datetime.datetime.now().strftime("%d%m%Y_%H%M%S"), "wb") as f:
            pickle.dump(final_sample, f)
