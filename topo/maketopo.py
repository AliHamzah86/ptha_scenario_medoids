"""
Simple script to download etopo1 topography/bathymetry data from
    http://www.ngdc.noaa.gov/mgg/global/global.html

The etopo1 data has 1-arcminute resolution, but you can request coarsening.
E.g. set resolution = 4./60. for 4-arcminute resolution.

"""

import os
from clawpack.geoclaw import topotools

import os,sys

try:
    CLAW = os.environ['CLAW']
except:
    raise Exception("*** Must first set CLAW enviornment variable")
scratch_dir = os.path.join(CLAW, 'geoclaw', 'scratch') 


def plot_topo(topo_fname):

    # plot the topo and save as a png file...
    import matplotlib.pyplot as plt
    
    # Scratch directory for storing topo and dtopo files:

    topo = topotools.Topography(os.path.join(scratch_dir,topo_fname), topo_type=2)
    topo.plot()
    fname = os.path.splitext(topo_fname)[0] + '.png'
    plt.savefig(fname)
    plt.close()
    print("Created ",fname)
   
if __name__=='__main__':

    topo_fname = 'etopo1min130E210E0N60N.asc'
    plot_topo(topo_fname)