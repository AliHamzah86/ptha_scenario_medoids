
import sys, os

from clawpack.geoclaw import topotools
from clawpack.visclaw import colormaps

import dtopotools

from scipy import stats
import KDEplots

import pandas as pd
import matplotlib.pyplot as plt
from numpy import random
from ipywidgets import interact
import ipywidgets as widgets


# Only needed if saving figures, these lines are commented out in notebook.
subdir = 'figures'
os.system('mkdir -p %s' % subdir)


def savefigp(fname):
    fname = os.path.join(subdir,fname)
    plt.savefig(fname, bbox_inches='tight')
    print ("Created ",fname)
    
# ## Function to compute "distance" between any two subfaults:

# %%
def compute_subfault_distances(fault):

    import numpy
    from numpy import pi,sqrt,sin,cos,tan
    rad = pi/180.       # conversion factor from degrees to radians
    rr = 6.378e6        # radius of earth
    lat2meter = rr*rad  # conversion factor from degrees latitude to meters

    nsubfaults = len(fault.subfaults)
    D = numpy.zeros((nsubfaults,nsubfaults))
    Dstrike = numpy.zeros((nsubfaults,nsubfaults))
    Ddip = numpy.zeros((nsubfaults,nsubfaults))
    for i, si in enumerate(fault.subfaults):
        xi = si.longitude
        yi = si.latitude
        zi = si.depth
        for j,sj in enumerate(fault.subfaults):
            xj = sj.longitude
            yj = sj.latitude
            zj = sj.depth
            dx = abs(xi-xj)*cos(0.5*(yi+yj)*pi/180.) * lat2meter
            dy = abs(yi-yj) * lat2meter
            dz = abs(zi-zj)

            # Euclidean distance:
            D[i,j] = sqrt(dx**2 + dy**2 + dz**2)
            
            # estimate distance down-dip based on depths:
            dip = 0.5*(si.dip + sj.dip)
            ddip1 = dz / sin(dip*pi/180.)
            Ddip[i,j] = ddip1 
            if Ddip[i,j] > D[i,j]:
                # should not happen...
                if 0:
                    print ("i,j,dx,dy,dz: ",i,j,dx,dy,dz)
                    print ("*** Ddip = %s, D = %s" % (Ddip[i,j], D[i,j]))

            # compute distance in strike direction to sum up properly:
            dstrike2 = max(D[i,j]**2 - Ddip[i,j]**2, 0.)
            Dstrike[i,j] = sqrt(dstrike2)
                
    return D,Dstrike,Ddip
        

import matplotlib.pyplot as plt
# Crescent City location:
xcc = -124.1838
ycc = 41.7456

column_map = {"longitude":1, "latitude":2, "depth":3, "strike":4, 
              "length":5, "width":6, "dip":7}
defaults = {'rake': 90, 'slip':1.0}
coordinate_specification = 'top center'
input_units = {'slip': 'm', 'depth': 'km', 'length': 'km', 'width': 'km'}
rupture_type = 'static'
skiprows = 1
delimiter = ','

fault = dtopotools.Fault()
fault.read('CSZe01.csv', column_map, coordinate_specification,
           rupture_type,skiprows, delimiter, input_units, defaults)
print ("There are %s subfaults" % len(fault.subfaults))

for s in fault.subfaults:
    s.longitude = s.longitude - 360.  # adjust to W coordinates
    
plt.figure(figsize=(15,6))
ax = plt.subplot(131);
fault.plot_subfaults(ax)
plt.xticks(range(-128,-123));

# Now subdivide each subfault further

new_subfaults = []  # to accumulate all new subfaults

phi_plate = 60.  # angle oceanic plate moves clockwise from north, to set rake

for subfault in fault.subfaults:
    subfault.rake = subfault.strike - phi_plate - 180.
    # subdivide into nstrike x ndip subfaults, based on the dimensions of the
    # fault:
    nstrike = int(subfault.length/12000)
    ndip = int(subfault.width/10000)
    f = dtopotools.SubdividedPlaneFault(subfault, nstrike, ndip)
    new_subfaults = new_subfaults + f.subfaults

# reset fault.subfaults to the new list of all subfaults after subdividing:
new_fault = dtopotools.Fault(subfaults = new_subfaults)
n = len(new_fault.subfaults)
print ("Subdivided fault has %s subfaults" % n)

ax = plt.subplot(132);
new_fault.plot_subfaults(ax)
plt.xticks(range(-128,-123));

topo_fname = 'etopo1min130E210E0N60N.asc'
topo_dir = os.path.join(os.getcwd(),'topo')
topo_fname = os.path.join(topo_dir, topo_fname)

topo = topotools.Topography(os.path.join(topo_dir,topo_fname), topo_type=2)

ax = plt.subplot(133)
plt.contourf(topo.X,topo.Y,topo.Z,[0,20000],colors=[[.3,1,.3]])
fault.plot_subfaults(ax)
plt.axis((-128,-122,40,50))

D, Dstrike, Ddip = compute_subfault_distances(new_fault)

import numpy as np

# make correlation matrix:
# Gaussian with correlation lengths Lstrike and Ldip:
Lstrike = 400e3
Ldip = 40e3

print ("Correlation lengths: Lstrike = %g, Ldip = %g" % (Lstrike,Ldip))
r = np.sqrt((Dstrike/Lstrike)**2 + (Ddip/Ldip)**2)
C = np.exp(-r)


lengths = np.array([s.length for s in fault.subfaults])
widths = np.array([s.width for s in fault.subfaults])
areas = lengths * widths
total_area = sum(areas)

Mw_desired = 9.0
Mo_desired = 10.**(1.5*Mw_desired + 9.05)
mean_slip = Mo_desired / (fault.subfaults[0].mu * total_area)
print ("mean_slip %g meters required for Mw %s" % (mean_slip, Mw_desired))

# Turn this into a constant vector:
mean_slip = mean_slip * np.ones(n)

alpha = 0.5
sigma_slip = alpha * mean_slip


## Lognormal:
Cov_g = np.log((sigma_slip/mean_slip) * (C*(sigma_slip/mean_slip)).T + 1.)
mean_slip_g = np.log(mean_slip) - np.diag(Cov_g)/2.

## This should be the same:
Cov_g = np.log(alpha**2 * C + 1.)

# Find eigenvalues, and eigenvector matrix.
print ("Finding eigenmodes from %s by %s matrix C" % (n,n))
lam, V = np.linalg.eig(Cov_g)
    
eigenvals = np.real(lam)  # imaginary parts should be at rounding level
V = np.real(V)

# Sort eigenvalues:
i = list(np.argsort(lam))
i.reverse()
lam = lam[i]
V = V[:,i]

plt.figure(figsize=(12,6))

ni = 1; nj = 4;
ax = plt.axes((.1,.1,.15,.8))
plt.contourf(topo.X,topo.Y,topo.Z,[0,20000],colors=[[.3,1,.3]])
plt.contour(topo.X,topo.Y,topo.Z,[0],colors='g')
fault.plot_subfaults(ax)
plt.axis((-128,-122,40,50))

cmap_slip = colormaps.make_colormap({0:'g',0.5:'w',1.:'m'})

for ii in range(ni):
    for jj in range(nj):
        pij = ii*nj + jj
        if jj<3:
            ax = plt.axes((.3 + jj*0.15,.12,.12,.76))
            shrink = 0
        else:
            ax = plt.axes((.3 + jj*0.15,.12,.17,.76))
            shrink = 0.
        V_amp = np.sqrt(sum(V[:,pij]**2))    # abs(V[:,pij]).max()
        #weight = sqrt(eigenvals[pij]) * V_amp / mean_amp
        for j,s in enumerate(new_fault.subfaults):
            s.slip = -V[j,pij] * 18.

        new_fault.plot_subfaults(ax,slip_color=True,cmin_slip=-1,cmax_slip=1,
                plot_box=0., cmap_slip=cmap_slip, colorbar_shrink=shrink)
        plt.title('Mode %s' % pij)
        plt.axis('off')

savefigp('CSZ.jpg')

topo_fname = 'etopo1min130E210E0N60N.asc'
topo_dir = os.path.join(os.getcwd(),'topo')
topo_fname = os.path.join(topo_dir, topo_fname)

topo = topotools.Topography(os.path.join(topo_dir,topo_fname), topo_type=2)

fault.subfaults = fault.subfaults[:8]

if 1:
    plt.figure(figsize=(10,4))
    ax = plt.subplot(121);
    fault.plot_subfaults(ax)
    plt.xticks(range(-126,-123));
    plt.contourf(topo.X,topo.Y,topo.Z,[0,20000],colors=[[.3,1,.3]])

# Now subdivide each subfault further

new_subfaults = []  # to accumulate all new subfaults

phi_plate = 60.  # angle oceanic plate moves clockwise from north, to set rake

for subfault in fault.subfaults:
    subfault.rake = subfault.strike - phi_plate - 180.
    # subdivide into nstrike x ndip subfaults, based on the dimensions of the
    # fault:
    nstrike = int(subfault.length/8000)
    ndip = int(subfault.width/8000)
    f = dtopotools.SubdividedPlaneFault(subfault, nstrike, ndip)
    new_subfaults = new_subfaults + f.subfaults

# reset fault.subfaults to the new list of all subfaults after subdividing:
new_fault = dtopotools.Fault(subfaults = new_subfaults)
n = len(new_fault.subfaults)
print ("Subdivided fault has %s subfaults" % n)

if 1:
    ax = plt.subplot(122);
    new_fault.plot_subfaults(ax)
    plt.xticks(range(-126,-123));


D, Dstrike, Ddip = compute_subfault_distances(new_fault)

Lstrike = 130e3
Ldip = 40e3

print ("Correlation lengths: Lstrike = %g, Ldip = %g" % (Lstrike,Ldip))
r = np.sqrt((Dstrike/Lstrike)**2 + (Ddip/Ldip)**2)
C = np.exp(-r)


lengths = np.array([s.length for s in fault.subfaults])
widths = np.array([s.width for s in fault.subfaults])
areas = lengths * widths
total_area = sum(areas)

Mw_desired = 8.8
Mo_desired = 10.**(1.5*Mw_desired + 9.05)
mean_slip = Mo_desired / (fault.subfaults[0].mu * total_area)
print ("mean_slip %g meters required for Mw %s" % (mean_slip, Mw_desired))

# Turn this into a constant vector:
mean_slip = mean_slip * np.ones(n)

alpha = 0.5
sigma_slip = alpha * mean_slip


## Lognormal:
Cov_g = np.log((sigma_slip/mean_slip) * (C*(sigma_slip/mean_slip)).T + 1.)
mean_slip_g = np.log(mean_slip) - np.diag(Cov_g)/2.

## This should be the same:
Cov_g = np.log(alpha**2 * C + 1.)

print ("Finding eigenmodes from %s by %s matrix C" % (n,n))
lam, V = np.linalg.eig(Cov_g)
    
eigenvals = np.real(lam)  # imaginary parts should be at rounding level
V = np.real(V)

# Sort eigenvalues:
i = list(np.argsort(lam))
i.reverse()
lam = lam[i]
V = V[:,i]

plt.figure(figsize=(12,8))

cmap_slip = colormaps.make_colormap({0:'g',0.5:'w',1.:'m'})

ni = 2; nj = 4;
ax = plt.axes((.1,.3,.2,.4))
plt.contourf(topo.X,topo.Y,topo.Z,[0,20000],colors=[[.3,1,.3]])
plt.contour(topo.X,topo.Y,topo.Z,[0],colors='g')
plt.plot([xcc],[ycc],'wo')
plt.plot([xcc],[ycc],'kx')
fault.plot_subfaults(ax)
plt.axis((-126,-123,40,44))

for ii in range(ni):
    for jj in range(nj):
        pij = ii*nj + jj
        if ii==0:
            ax = plt.axes((.35 + jj*0.14,.5,.12,.38))
        else:
            ax = plt.axes((.35 + jj*0.14,.1,.12,.38))
            
        V_amp = np.sqrt(sum(V[:,pij]**2))    # abs(V[:,pij]).max()
        #weight = sqrt(eigenvals[pij]) * V_amp / mean_amp
        for j,s in enumerate(new_fault.subfaults):
            s.slip = V[j,pij] * 15.

        new_fault.plot_subfaults(ax,slip_color=True,cmin_slip=-1,cmax_slip=1,
                plot_box=0., cmap_slip=cmap_slip, colorbar_shrink=0)
        plt.title('Mode %s' % pij)
        plt.axis('off')

savefigp('CSZmodes.jpg')

# Taper:
max_depth = 20000.
tau = lambda d: 1. - np.exp((d - max_depth)*20/max_depth)

def KL(z):
    KL_slip = 0.*mean_slip_g.copy()  # drop the mean slip and rescale later
    # add in the terms in the K-L expansion:  (dropping V[:,0])
    for k in range(1,len(z)):
        KL_slip += z[k] * np.sqrt(lam[k]) * V[:,k]
    
    ## Exponentiate to get Lognormal distribution:
    KL_slip = np.exp(KL_slip)
    
    # Set the fault slip for the resulting realization:
    for j,s in enumerate(new_fault.subfaults):
        s.slip = KL_slip[j] * tau(s.depth)
        
    # Rescale to have desired magnitude:
    Mo = new_fault.Mo()
    KL_slip *= Mo_desired/Mo
    for j,s in enumerate(new_fault.subfaults):
        s.slip = KL_slip[j] * tau(s.depth)
    
    return KL_slip

# Since we are using the Lognormal distribution we cannot apply Okada to each eigenmode directly.  
# Instead we apply Okada to each unit source and then take linear combinations of these for 
# any particular slip distribution. A unit source has slip 1 on one subfault and slip 0 on all others.  
# We must compute the seafloor deformation for each such unit source, 
# one for each subfault in the model (540 in the example used here.)

# %%
# grid on which to compute deformation:
nx_dtopo = 181
ny_dtopo = 361
x_dtopo = np.linspace(-126,-123,nx_dtopo)
y_dtopo = np.linspace(39,45,ny_dtopo)

n_subfaults = len(new_fault.subfaults)
dZ = np.zeros((ny_dtopo, nx_dtopo, n_subfaults)) # to store sea floor deformation corresponding to each mode V[:,j]

for j in range(n_subfaults):
    sfault = dtopotools.Fault(subfaults = [new_fault.subfaults[j]])
    sfault.subfaults[0].slip = 1.
    dtopo = sfault.create_dtopography(x_dtopo,y_dtopo,times=[1.], verbose=False)
    sys.stdout.write('%i...' % j)
    sys.stdout.flush()
    dZ[:,:,j] = dtopo.dZ[0,:,:]

def PotentialEnergy(dZr):
    dy = 1./60. * 111.e3  # m
    dx = dy * np.cos(topo.Y *np.pi/180.)  # m
    grav = 9.81  # m/s^2
    rho_water = 1000  # kg/m^3
    eta = np.ma.masked_where(topo.Z>0, dZr)
    Energy = sum(eta**2 * dx * dy) * grav * rho_water * 1e-15  # PetaJoules
    return Energy

i1cc = np.where(dtopo.x<xcc)[0].max()
j1cc = np.where(dtopo.y<ycc)[0].max()
a1cc = (xcc-dtopo.x[i1cc])/(dtopo.x[i1cc+1]-dtopo.x[i1cc])
a2cc = (ycc-dtopo.y[j1cc])/(dtopo.y[j1cc+1]-dtopo.y[j1cc])
if (a1cc<0.) or (a1cc>1.) or (a2cc<0.) or (a2cc>1.):
    print ('*** Interpolation to CC not correct!')

def dZ_CrescentCity(dZr):
    dzy1 = (1.-a1cc)*dZr[j1cc,i1cc] + a1cc*dZr[j1cc,i1cc+1]
    dzy2 = (1.-a1cc)*dZr[j1cc+1,i1cc] + a1cc*dZr[j1cc+1,i1cc+1]
    dzcc = (1.-a2cc)*dzy2 + a2cc*dzy1
    return dzcc

seed = 13579   # so random number generator gives repeatable results
random.seed(seed)

nterms = 60
nterms2 = 7
plt.figure(figsize=(10,12))
for i in range(1,6):
    z = np.random.randn(nterms)
    KL_slip = KL(z)
    ax = plt.subplot(4,5,i)
    new_fault.plot_subfaults(ax, slip_color=True, cmax_slip=20., plot_box=False, colorbar_shrink=0)
    plt.axis('off')
    plt.title('Realization %i\n %i terms' % (i,nterms), fontsize=10)
    #dtopo = new_fault.create_dtopography(x_dtopo,y_dtopo,times=[1.], verbose=False)
    dZr = np.dot(dZ,KL_slip)  # linear combination of dZ from unit sources
    ax = plt.subplot(4,5,5+i)
    dtopotools.plot_dZ_colors(dtopo.X,dtopo.Y,dZr, axes=ax, cmax_dZ = 8., \
                              dZ_interval = 1., add_colorbar=False)
    plt.ylim(39.5,44.5)
    plt.plot([xcc],[ycc],'wo')
    plt.plot([xcc],[ycc],'kx')
    # plt.title('E=%4.2f, dB=%5.2f'% (PotentialEnergy(dZr),dZ_CrescentCity(dZr)), fontsize=10)
    plt.axis('off')
    
    z = z[:nterms2]
    KL_slip = KL(z)
    ax = plt.subplot(4,5,10+i)
    new_fault.plot_subfaults(ax, slip_color=True, cmax_slip=20., plot_box=False, colorbar_shrink=0)
    plt.axis('off')
    plt.title('%i terms' % nterms2, fontsize=10)
    #dtopo = new_fault.create_dtopography(x_dtopo,y_dtopo,times=[1.], verbose=False)
    dZr = np.dot(dZ,KL_slip)  # linear combination of dZ from unit sources
    ax = plt.subplot(4,5,15+i)
    dtopotools.plot_dZ_colors(dtopo.X,dtopo.Y,dZr, axes=ax, cmax_dZ = 8., \
                              dZ_interval = 1., add_colorbar=False)
    plt.ylim(39.5,44.5)
    plt.plot([xcc],[ycc],'wo')
    plt.plot([xcc],[ycc],'kx')
    # plt.title('E=%4.2f, dB=%5.2f'% (PotentialEnergy(dZr),dZ_CrescentCity(dZr)), fontsize=10)
    plt.axis('off')
    
def test(ntrials = 10000, nterms=60):
    Energy = np.zeros(ntrials)
    Amplitude = np.zeros(ntrials)
    z_shore = np.zeros(ntrials)
    EtaMax = np.zeros(ntrials)
    
    zvals = np.zeros((ntrials,nterms+1))
    for j in range(ntrials):
        z = np.random.randn(nterms+1)  # choose random z for this realization
        zvals[j,:] = z
        KL_slip = KL(z)
        dZr = np.dot(dZ,KL_slip)  # linear combination of dZ from unit sources
        Energy[j] = PotentialEnergy(dZr)
        z_offshore = np.where(topo.Z < 0, dZr, 0.)
        Amplitude[j] = z_offshore.max() - z_offshore.min()
        z_shore[j] = dZ_CrescentCity(dZr)
        EtaMax[j] = z_offshore.max()
    return Energy, Amplitude, z_shore, EtaMax, zvals

random.seed(12345)
ntrials = 20000
# print ("Generating %s samples..." % ntrials) 

Energy, Amplitude, z_shore, EtaMax, zvals = test(ntrials = ntrials, nterms=60)

DepthProxy = EtaMax - z_shore
realizations = pd.DataFrame()
realizations['Energy'] = Energy
realizations['Amplitude'] = Amplitude
realizations['subsidence / uplift'] = z_shore
realizations['EtaMax'] = EtaMax
realizations['depth proxy'] = DepthProxy

# %%
random.seed(12345)
ntrials = 20000
nterms2 = 7
print ("Generating %s samples..." % ntrials)
Energy, Amplitude, z_shore, EtaMax, zvals = test(ntrials = ntrials, nterms=nterms2)

DepthProxy = EtaMax - z_shore
realizations2 = pd.DataFrame()
realizations2['Energy'] = Energy
realizations2['Amplitude'] = Amplitude
realizations2['subsidence / uplift'] = z_shore
realizations2['EtaMax'] = EtaMax
realizations2['depth proxy'] = DepthProxy

Q1 = 'EtaMax'
Q2 = 'subsidence / uplift'

Nx = 200
Ny = 201

x0 = 3.
x1 = 12.

y0 = -2.
y1 = 4.

x = np.linspace(x0,x1,Nx)
y = np.linspace(y0,y1,Ny)

X,Y = np.meshgrid(x,y)

xy = np.vstack((X.flatten(),Y.flatten()))


## 60 terms:

rvals = np.vstack((realizations[Q1].T, realizations[Q2].T))
kde = stats.gaussian_kde(rvals)
p = kde.pdf(xy) * (x1-x0)*(y1-y0)/float(Nx*Ny)
rho = np.reshape(p,(Ny,Nx))

KDEplots.joint_plot(X,Y,rho,xname='EtaMax (m)',yname='subsidence/uplift (m)')
savefigp('joint_Eta_DBshore_60b.jpg')

## 7 terms:

rvals = np.vstack((realizations2[Q1].T, realizations2[Q2].T))
kde2 = stats.gaussian_kde(rvals)
p2 = kde2.pdf(xy) * (x1-x0)*(y1-y0)/float(Nx*Ny)
rho2 = np.reshape(p2,(Ny,Nx))

KDEplots.joint_plot(X,Y,rho2,xname='EtaMax (m)',yname='subsidence/uplift (m)')
savefigp('joint_Eta_DBshore_7b.jpg')


Q1 = 'EtaMax'
Q2 = 'Energy'

Nx = 200
Ny = 201

x0 = 3.
x1 = 12.

y0 = 0.8
y1 = 1.6

x = np.linspace(x0,x1,Nx)
y = np.linspace(y0,y1,Ny)

X,Y = np.meshgrid(x,y)

xy = np.vstack((X.flatten(),Y.flatten()))


## 60 terms:

rvals = np.vstack((realizations[Q1].T, realizations[Q2].T))
kde = stats.gaussian_kde(rvals)
p = kde.pdf(xy) * (x1-x0)*(y1-y0)/float(Nx*Ny)
rho = np.reshape(p,(Ny,Nx))

KDEplots.joint_plot(X,Y,rho,xname='EtaMax (m)',yname='Energy (PetaJoules)')
savefigp('joint_Eta_Energy_60b.jpg')

## 7 terms:

rvals = np.vstack((realizations2[Q1].T, realizations2[Q2].T))
kde2 = stats.gaussian_kde(rvals)
p2 = kde2.pdf(xy) * (x1-x0)*(y1-y0)/float(Nx*Ny)
rho2 = np.reshape(p2,(Ny,Nx))

KDEplots.joint_plot(X,Y,rho2,xname='EtaMax (m)',yname='Energy (PetaJoules)')
savefigp('joint_Eta_Energy_7b.jpg')
