import meshio
import matplotlib.pyplot as plt
import numpy as np
import glob,math,pickle,os
from scipy.stats import binned_statistic, binned_statistic_2d
import seaborn as sb
import pandas as pd
import matplotlib
from numpy import ma
from matplotlib import ticker, cm

def load_params(fname):
   with open(fname,'rb') as fob:
      data = pickle.load(fob)
   return data

def get_data( data ):
   dname = list(data.point_data.keys())[0]
   return data.point_data[dname]

def get_x(data):
   return data.points

def get_r(x):
   return np.sqrt( x[:,0]**2 + x[:,1]**2 + x[:,2]**2 )

def p75( x ):
   return np.percentile( x, 75)

def p25( x ):
   return np.percentile( x, 25)

def p50(x):
   return np.percentile(x,50)

def get_midpoints(edges):
   return edges[1:]-np.diff(edges)

prefix = "diffusion_gaussian/"
parallel = ''#MAke empty string if not parallel

params = load_params('params.pickle')

fig, axs = plt.subplots(2, 1, figsize=(7,3))

titles = ["$L_G$=$w_0$","$L_G$=4$w_0$"]
titles = ["Small $L_g$","Large $L_g$"]

for idx,Lsi in enumerate( params['Ls'] ):

   idxstr = f"{idx:06d}"
   axi = axs[idx]

   dfile = prefix+f'diffusivity{idxstr}.{parallel}vtu'
   rfile = prefix+f'absorption_rate{idxstr}.{parallel}vtu'
   cfile = prefix+f"concentration{idxstr}.{parallel}vtu"
   afile = prefix+f"absorbed{idxstr}.{parallel}vtu"
   sfile = prefix+f"source{idxstr}.{parallel}vtu"

   adata = meshio.read(afile)
   ddata = meshio.read(dfile)
   #sdata = meshio.read(sfile)
   #rdata = meshio.read(rfile)

   D = get_data(ddata)
   #s = get_data(sdata)
   #ar = get_data(rdata)
   armax = params['r_eps']#Make sure it's a constant for consistent normalization!
   a = get_data(adata)/armax

   x = get_x(ddata)
   xb,yb,zb = sorted(list(set(x[:,0]))),sorted(list(set(x[:,1]))),sorted(list(set(x[:,2])))
   r = get_r( x )
   mask = (D < 0.99) * (r < 0.95*np.pi)
   x = x[mask]
   r = r[mask]
   a = a[mask]
   Nr = 30
   Ny = 13
   rmin = 0.05*r.max()
   rbins = np.linspace(rmin, r.max(), Nr)
   A = []

   for i in range(1,len(rbins)):
      shell = ( r >= rbins[i-1] ) * ( r < rbins[i])
      ri = r[shell]
      ai = a[shell]

      srt = np.argsort(ai)[::-1]

      ri = ri[srt]
      ai = ai[srt]

      gidx = np.linspace(0, len(ri)-1, Ny)

      aint = np.interp( gidx, list(range(len(ai))), ai )
      A.append(aint)

   ya = np.linspace(0,1,Ny)
   A = np.array(A)

   levels = np.logspace( -3,0,7 )

   cs = axi.contourf( get_midpoints(rbins), ya, A.transpose(), locator=ticker.LogLocator(), cmap=cm.BuGn, levels=levels)

   axi.set_yticks([])

   if idx == 0:
      cbar = fig.colorbar(cs, ax=axs, shrink=0.8, location='right',aspect=10)

      cbar.set_label('Relative dose',fontsize=14, rotation=270,labelpad=20)
      #cbar.ax.set_label('Relative dose',fontsize=14)
      axi.set_xticks([])
   else:
      axi.set_xlabel("Relative distance to NP", fontsize=14)
   axi.set_ylabel(titles[idx],fontsize=14)

   #axi.margins(x=0.25,y=0.25)

plt.subplots_adjust(bottom=0.17, right=0.75)
fname = "dose_radius_Lg.png"
plt.savefig(fname,dpi=300)
os.system(f"convert -trim {fname} {fname}")


