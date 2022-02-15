from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

Gaia = fits.open('/root/下载/data_exercise4_1.fits')
Gaia.info()
#fit_file = get_pkg_data_filename('/root/下载/data_exercise4_1.fits')
#data = fits.getdata(fit_file)
#51113*32
#Gaia[1].header

#Question 1. Plot the colour-magnitude(BP-RP vs G) diagram of the full dataset.
def density_calc(x, y, radius):
    """
    Scatter density calculation (for color rendering of scatter density in scatter plots)
    :param x:
    :param y:
    :param radius:
    :return: data sampels' density
    """
    res = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        print(i)
        res[i] = np.sum((x > (x[i] - radius)) & (x < (x[i] + radius))
                        & (y > (y[i] - radius)) & (y < (y[i] + radius)))
    return res

G=Gaia[1].data.field('phot_g_mean_mag') #label for column 12
BP_RP=Gaia[1].data.field('bp_rp') #label for column 22 --> Gaia[1].data.field(21)

# ----------- Define Parameters ------------
radius = 0.17  # radius
colormap = plt.get_cmap("jet") # color bar
marker_size = 1  #size of point
yrange = [24,8] 
xrange = [-2, 5] 
ylabel = "phot_g_mean_mag"
xlabel = "bp_rp/mag"
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 7}
xticks = np.linspace(24, 8, 1)
yticks = np.linspace(-2, 5, 1)
cbar_ticks = [10**0, 10**1, 10**2, 10**3, 10**4, 10**5]
# ----------------- Plot -----------------
fig = plt.figure (1,facecolor="grey")
Z1 = density_calc( BP_RP, G, radius)
plt.scatter(BP_RP, G, c=Z1, cmap=colormap, marker=".", s=marker_size,
            norm=colors.LogNorm(vmin=500, vmax=35500))
plt.xlim(xrange)
plt.ylim(yrange)
plt.xticks(xticks, fontproperties='Times New Roman', size=7)
plt.yticks(yticks, fontproperties='Times New Roman', size=7)

plt.xlabel(xlabel, fontdict=font)
plt.ylabel(ylabel, fontdict=font)
plt.title('BP-RP vs G diagram')
plt.grid(linestyle='--', color="grey")
plt.plot(xrange, yrange, color="k", linewidth=0.8, linestyle='--')
plt.rc('font', **font)
fig.tight_layout()
cbar = plt.colorbar(orientation='horizontal', extend="both", pad=0.1)
cbar.set_label("Scatter Density", fontdict=font)
cbar.set_ticks(cbar_ticks)
cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=6) 
cbar.ax.tick_params(which="minor", direction="in", length=0) 
plt.show()

#------------------------------------------------------------
# Question 2.Select only objects which have positive parralaxes and parallax erroe smaller than 20%


parallaxes = Gaia[1].data.field('parallax')   #field(9)
parallaxes_error = Gaia[1].data.field('parallax_error')  #field(17)

def select_parrallax(num):
  num = str(num)
  pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
  result = pattern.match(num)
  if result:
    return True
  else:
    return False

#slected the data with values
cleaned_data_1=[]
for i in range(len(parallaxes)):
    if ( select_parrallax(parallaxes[i]) == True & select_parrallax(parallaxes_error[i] )== True ):
       print(i)
       cleaned_data_1.append(Gaia[1].data[i])

#select the parallax errors smaller than 20%
cleaned_phot_g_mean_mag=[]
cleaned_bp_rp=[]
for i in range(40442):
    if ((0<cleaned_data_1[i][16]< 0.2) & (0<cleaned_data_1[i][8])):
       cleaned_phot_g_mean_mag.append(cleaned_data_1[i][11])  # colume[11] and column[12] nealry
       cleaned_bp_rp.append(cleaned_data_1[i][21])

plt.scatter(cleaned_bp_rp, cleaned_phot_g_mean_mag)
plt.show()

# ----------- Define Parameters ------------
radius2=0.4  # radius
colormap = plt.get_cmap("jet") # color bar
marker_size = 1  #size of point
yrange = [24,8] 
xrange = [-3, 6] 
ylabel = "phot_g_mean_mag"
xlabel = "bp_rp/mag"
font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 7}
xticks = np.linspace(24, 8, 1)
yticks = np.linspace(-3, 6, 1)
cbar_ticks = [10**0, 10**1, 10**2, 10**3, 10**4, 10**5]

#-----plot fig2------------
fig = plt.figure (1,facecolor="grey")
Z1 = density_calc(cleaned_bp_rp, cleaned_phot_g_mean_mag, radius2)
plt.scatter(cleaned_bp_rp, cleaned_phot_g_mean_mag, c=Z1, cmap=colormap, marker=".", s=marker_size, norm=colors.LogNorm(vmin=100, vmax=25500))
#plt.scatter(cleaned_bp_rp, cleaned_phot_g_mean_mag, c=Z1, cmap=colormap, marker=".", s=marker_size,     norm=colors.LogNorm(vmin=Z1.min(), vmax=0.5 * Z1.max()))
plt.xlim(xrange)
plt.ylim(yrange)
plt.xticks(xticks, fontproperties='Times New Roman', size=7)
plt.yticks(yticks, fontproperties='Times New Roman', size=7)

plt.xlabel(xlabel, fontdict=font)
plt.ylabel(ylabel, fontdict=font)
plt.title('cleaned BP-RP vs G diagram')
plt.grid(linestyle='--', color="grey")
plt.plot(xrange, yrange, color="k", linewidth=0.8, linestyle='--')
plt.rc('font', **font)
fig.tight_layout()

cbar = plt.colorbar(orientation='horizontal', extend="both", pad=0.1)
cbar.set_label("Scatter Density", fontdict=font)
cbar.set_ticks(cbar_ticks)
cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=6) 
cbar.ax.tick_params(which="minor", direction="in", length=0) 
plt.show()















