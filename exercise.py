import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit

#task: select the planets in the habitable zone -> din<d<dout
#din=1.68*10e-8*R*Teff**2
#dout=3.12*10e-8*R*Teff**2


data = pd.read_csv('/root/下载/exoplanet.eu_catalog.csv', index_col=0)
# print(type(data))

data = data[data['planet_status'] == 'Confirmed']
cfg=int(len(data))
mass = data['mass']
star_teff = data['star_teff']
star_radius = data['star_radius']
radius=data['radius']
semi_major_axis = data['semi_major_axis']
din=1.68*10e-8*star_radius*star_teff**2
dout=3.12*10e-8*star_radius*star_teff**2
habitable_planet=[]
semi_major_axis_2=[]
star_teff_2=[]
density=[]
mass2=[]
radius2=[]

#question 3
for i in range(cfg):
    if semi_major_axis[i] > din[i] and semi_major_axis[i] < dout[i]:
       semi_major_axis_2.append(semi_major_axis[i])
       star_teff_2.append(star_teff[i])

xdata=semi_major_axis_2
ydata=star_teff_2
plt.xlabel('Semi-Major axis[AU]')
plt.ylabel('Effective temperature of host star[K]')
plt.title('Semi-Major axis-Effetive Temperatures of their host stars diagram')
plt.scatter(xdata, ydata, alpha=0.5,c='b')
plt.show()

#quesstion4 filter the desity of planet which>2.5g/cm^3

for i in range(cfg):
    density.append(mass[i]/(4*np.pi/3*radius[i]**3))

for i in range(cfg):
    if density[i] > 2.5:
       density.append(density)
       mass2.append(mass)
       radius2.append(radius)

xdata=radius2
ydata=mass2
plt.xlabel('Radius[m]')
plt.ylabel('Mass[kg]')
plt.title('Smass-radius diagram')
plt.scatter(xdata, ydata, alpha=0.5,c='b')
plt.show()

# Another parameter could be essetial for habitable planet is Tne magnetic field. Because the magnetic fields can reduce the size of planetary magnetospheres to such an extent that a significant fraction of the planet's atmosphere may be exposed to erosion by the stellar wind. The magnetic fields of M dwarf (dM) stars on potentially habitable Earth-like planets. Then we can select the habitable planets by certain range of intensity of magnetic field.

#quesstion6 filter the intensity of magnetic field of planet which belong to [5G, 7000G](assumption, stll need precise modeling process before determine values)

star_magnetic_field = data['star_magnetic_field'] 
star_magnetic_field_2=[]

for i in range(cfg):
    #if star_magnetic_field[i] > 5 and star_magnetic_field[i]<7000
       star_magnetic_field_2.append(star_magnetic_field[i])
       radius2.append(radius)

xdata=radius2
ydata=star_magnetic_field_2
plt.xlabel('radius[m]')
plt.ylabel('star_magnetic_field_2[G]')
plt.title('star_magnetic_field-radius diagram')
plt.scatter(xdata, ydata, alpha=0.5,c='b')
plt.show()

          
            


