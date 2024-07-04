#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Font

# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('default')
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : "Computer Modern Serif"}
plt.rcParams.update(params)


# # Read FITS

# In[ ]:


import numpy as np
from astropy.io import fits


# In[ ]:


# eventually change this (create directory)
path = "/Users/users/stoica/DATA/new/new"


# In[ ]:


# eventually change this as well
pathfits = "/Users/users/stoica/DATA/L2/LWVE_00582288+2651523_01_BR_L1_P0000_APS.fits"
hdulist = fits.open(pathfits)


# In[ ]:


# PATCH_BINSPEC

header3 = hdulist[3].header
data3 = hdulist[3].data


# In[ ]:


# PATCH_TABLE

data2 = hdulist[2].data

rid = [] # ID
ID2 = [] # bin_ID
SNR = [] # SNR
xx = [] # x-coordinate
yy = [] # y-coordinate
zz = [] # redshift
for i in range(0, len(data2)):
    if data2[i][1] > 0:
        rid.append(data2[i][0])
        ID2.append(data2[i][1])
        SNR.append(float(data2[i][13]))
        xx.append(-data2[i][5]) # the x-coordinate is switched
        yy.append(data2[i][6])
        zz.append(data2[i][7])

ixx = np.argsort(SNR)
ixx = list(reversed(ixx)) # true indices of highest to lowest SNR
SNR.sort(reverse = True) # highest SNR first


# In[ ]:


# bin ID
# log(lambda)
# spectra
# error in spectra

ID = []
wave = []
spectra = []
err = []
for i in range(0, len(data3)):
    if data3[i][0] > 0:
        ID.append(data3[i][0])
        wave.append(data3[i][1]/np.log(10)) # base change
        spectra.append(data3[i][2])
        err.append(data3[i][3])


# In[ ]:


rSNR = []
for i in range(0, len(wave)):
    rSNR.append(np.mean(spectra[i]/(err[i])))


# In[ ]:


rixx = np.argsort(rSNR)
rixx = list(reversed(rixx)) # true indices of highest to lowest SNR
rSNR.sort(reverse = True) # highest SNR first


# In[ ]:


# x- and y-coordinates corresponding to the highest to lowest SNR
X = []
Y = []
ZZ = [] # redshift
for i in ixx:
    X.append(xx[i])
    Y.append(yy[i])
    ZZ.append(zz[i])


# In[ ]:


#ordt = [] # order of highest SNR bins from spectra/err with respect to fits header values
#for j in rixx:
    #ordt.append(ixx.index(j))


# # Index

# In[ ]:


# bin ID's
ix = ixx

number_of_files = 2*len(ix) + 1
even_files = 2*len(ix) + 2

number = []
for odd in range(1, number_of_files, 2):
    number.append(odd)

#pp = [] # added for "real" SNR indices
#for j in ordt:
    #pp.append(number[j])

index = [str(value) for value in number] # change pp to number for SNR from header
#print(index) # index of main files


number2 = []
for even in range(2, even_files, 2):
    number2.append(even)

#pp2 = [] # added for "real" SNR indices
#for j in ordt:
    #pp2.append(number2[j])

even = [str(value) for value in number2] # change pp to number2 for SNR from header
#print(even) # index of error files


# In[ ]:


# ix has all the bins
# divide the bins into 3 objects

bins1 = 391 # last bin of first object
bins2 = 4260 # last bin of main object
bins3 = 4443 # last bin of third object


BIN1 = [] # bins from highest to lowest SNR
X1 = []
Y1 = []
rSNR1 = []
for j in ix:
    if j < bins1:
        BIN1.append(j)
        X1.append(xx[j])
        Y1.append(yy[j])
        rSNR1.append(np.mean(spectra[j]/(err[j])))
rSNR1.sort(reverse = True)

BIN2 = [] # bins from highest to lowest SNR
X2 = []
Y2 = []
rSNR2 = []
for j in ix:
    if j >= bins1 and j < bins2:
        BIN2.append(j)
        X2.append(xx[j])
        Y2.append(yy[j])
        rSNR2.append(np.mean(spectra[j]/(err[j])))
rSNR2.sort(reverse = True)

BIN3 = [] # bins from highest to lowest SNR
X3 = []
Y3 = []
rSNR3 = []
for j in ix:
    if j >= bins2 and j<= bins3:
        BIN3.append(j)
        X3.append(xx[j])
        Y3.append(yy[j])
        rSNR3.append(np.mean(spectra[j]/(err[j])))
rSNR3.sort(reverse = True)


# In[ ]:


# finding the number of the new FITS files that corresponds to each bin

ordt1 = []
for j in BIN1:
    ordt1.append(ixx.index(j))
    
ordt2 = []
for j in BIN2:
    ordt2.append(ixx.index(j))
    
ordt3 = []
for j in BIN3:
    ordt3.append(ixx.index(j))


pb1 = []
for j in ordt1:
    pb1.append(number[j])
bin1 = [str(value) for value in pb1] # images of first object

pb2 = []
for j in ordt2:
    pb2.append(number[j])
bin2 = [str(value) for value in pb2] # images of main object

pb3 = []
for j in ordt3:
    pb3.append(number[j])
bin3 = [str(value) for value in pb3] # images of third object


# In[ ]:


# highest SNR bin and bin halfway through

jj = ['1', '4447']
kk = ['2', '4448']


# In[ ]:


midd = index.index('4447') # index of bin halfway through bin1 with respect to all bins in index


# In[ ]:





# # Write FITS

# In[ ]:


for j, i, k in zip(ix, index, even):
    hdr = fits.Header()
    hdr['NAXIS'] = 1
    hdr['BITPIX'] = -64
    hdr['CTYPE1'] = 'AWAVLOG'
    hdr['CRPIX1'] = 1.0
    hdr['CRVAL1'] = wave[j][0]
    hdr['CD1_1'] = (wave[j][1]-wave[j][0])
    hdr['CDELT1'] = (wave[j][1]-wave[j][0])
    hdr['DC-FLAG'] = 1
    hdr['SNR'] = rSNR[j]
    hdr['REDSHIFT'] = ZZ[j]
    empty_primary1 = fits.PrimaryHDU(header = hdr, data = spectra[j])
    empty_primary2 = fits.PrimaryHDU(header = hdr, data = err[j])
    hdul = fits.HDUList([empty_primary1])
    hdule = fits.HDUList([empty_primary2])
    hdul.writeto((path+i+'.fits'), overwrite = True)
    hdule.writeto((path+k+'.fits'), overwrite = True)


# # STECKMAP

# In[ ]:


#print(len(ix)) # this is the number of files
idd = len(ix) # Yorick counts from 1


# In[ ]:


# this only works for a small number of files

# efs = []
# for j in even:
#     efs.append('"'+path+j+".fits"+'"')

# efs = []
# for ef in evenfiles:
#     efs.append('"'+ef+'"')

# evenfiles = ""
# evenfiles += "["

# for e in efs:
#     evenfiles += e
#     evenfiles += ","
# evenfiles = evenfiles[:-1]
# evenfiles += "]"


# ofs = []
# for i in index:
#     ofs.append('"'+path+i+".fits"+'"')

# oddfiles = ""
# oddfiles += "["

# for o in ofs:
#     oddfiles += o
#     oddfiles += ","
# oddfiles = oddfiles[:-1]
# oddfiles += "]"


# In[ ]:


# this works for a large amount of files
# create filename.txt, efilename.txt and ma-ta.py files first

with open('/Users/users/stoica/DATA/efilename.txt', 'w') as the_file:
    for j in even:
        the_file.write(path+j+".fits"+'\n')

with open('/Users/users/stoica/DATA/filename.txt', 'w') as the_file:
    for i in index:
        the_file.write(path+i+".fits"+'\n')


# In[ ]:


# change parameters here

with open('/Users/users/stoica/Yorick/ma-ta.py', 'r') as file:
    data = file.readlines()

data[0] = 'include, "STECKMAP/Pierre/POP/pop_paths.i"\n'
data[1] = 'include, "STECKMAP/Pierre/POP/sfit.i"\n'

data[3] = f'idd = {idd}\n'

data[5] = 'ages = [5E8,13.6E9]\n'
data[6] = 'wavel = [3600.,6000.]\n'
data[7] = 'basisfile = "XSL"\n'
data[8] = 'nbins = 30\n'
data[9] = 'kin = 1\n'
data[10] = 'epar = 3\n'
data[11] = 'nde = 15\n'
data[12] = 'vlim = [-1000.,1000.]\n'
data[13] = 'meval = 500\n'
data[14] = 'RMASK = [[4400.,4410.], [4730.,4740.], [5235.,5335.], [5610.,5630.], [5900.,6000.]]\n'
data[15] = 'mux = 0.01\n'
data[16] = 'muz = 100\n'
data[17] = 'muv = 100\n'
data[18] = 'L1 = "D2"\n'
data[19] = 'L3 = "D1"\n'
data[20] = 'f = open("/Users/users/stoica/DATA/filename.txt", "r")\n'
data[21] = 'lines = rdline(f, 250000)\n'
data[22] = 'file = lines(where(lines))\n'

data[23] = 'ef = open("/Users/users/stoica/DATA/efilename.txt", "r")\n'
data[24] = 'lines = rdline(ef, 250000)\n'
data[25] = 'efile = lines(where(lines))\n'

data[26] = 'b = bRbasis3(ages, 4.6, nbins = nbins, R = 10000, wavel = wavel, basisfile = basisfile)\n'
data[27] = 'ws; plb, b.flux, b.wave\n'
data[28] = 'for (i=1; i<=idd; ++i) x = sfit((convert_all(file(i), log = 1, z0 = 0, errorfile = efile(i))), b, kin = kin, epar = epar, RMASK = RMASK, noskip = 1, vlim = vlim, meval = meval, nde = nde, mux = mux, muv = muv, muz = muz, L3 = L3, L1 = L1, sav = 1)\n' # change start if incomplete run
data[29] = "quit"

with open('/Users/users/stoica/Yorick/ma-ta.py', 'w') as file:
    file.writelines(data)


# In[ ]:


# use this for running in a terminal

#import os
#os.system("cd ~; export STECKMAPROOTDIR=$HOME/Yorick/; cd $STECKMAPROOTDIR; yorick -batch ma-ta.py")


# In[ ]:


# use this for running in a notebook

get_ipython().system('cd ~; export STECKMAPROOTDIR=$HOME/Yorick/; cd $STECKMAPROOTDIR; yorick -batch ma-ta.py')


# # Results files

# In[ ]:


import sys

def getpct(x, q, axis = 0):
    x = np.asarray(x)
    if len(x.shape) > 1: x = np.sort(x, axis)
    else: x = np.sort(x)
    j = min([len(x) - 1,int(0.01*q*len(x))])
    
    if len(x) > 2:
        if axis == 0: return x[j]
        else: return x[::, j]
    elif len(x): return np.sum(x)/(1.0*len(x))
    else: return 0.0

def readsection(f, startLine, numEntires = 30, nPerLine = 5):
    # assumes that f is a list constructed as f = file.readlines() on some file
    nLines = int(numEntries/nPerLine)
    if (1.*numEntries/nPerLine)!=nLines: nLines = nLines + 1
    l = []
    for j in range(startLine, startLine + nLines):
        s = (f[j].strip()).split()
        [l.append(float(s[i])) for i in range(len(s))]
    return np.array(l)

MCc = []
for k in index: # change to jj for MC run
    for filename in [path+k+'.res'+i for i in ['-AMR.txt', '-MASS.txt', '-SAD.txt', '-SFR.txt', '-LOSVD.txt']]:
        print(filename)
        f = open(filename).readlines()
        g = open(filename.replace('txt', 'dat'), 'w')
        numEntries = int((f[0].split())[-1])
        nPerLine = len(f[1].split())
        nLines = int(numEntries/nPerLine)
        if (numEntries%nPerLine)!=0: nLines = int(nLines) + 1
        ages = readsection(f, 1, numEntries, nPerLine)
        q = readsection(f, 2 + nLines, numEntries, nPerLine)
        
        # if there are MC simulations
        if len(f) > 2*nLines + 2:
            MC = []
            for i in range(int((f[2*nLines + 2].strip()).split()[-1])):
                MC.append(readsection(f, (2 + i)*(nLines + 1) + 1, numEntries, nPerLine))
            MC = np.array(MC)
            MCc.append(np.copy(MC))
            # print(MCc)
            mid = getpct(MC, 50)
            low1sigma = getpct(MC, 16)
            high1sigma = getpct(MC, 84)
            
            for j in range(len(ages)):
                ostr = '%12.6g %12.6g %12.6g %12.6g %12.6g\n' % (ages[j], q[j], low1sigma[j], high1sigma[j], mid[j])
                g.write(ostr)
        else:
            for j in range(len(ages)):
                ostr = '%12.6g %12.6g\n' % (ages[j], q[j])
                g.write(ostr)
        g.close()


for k in index: # change to jj for MC run
    filename = str(path+k+'.res'+'-spectra.txt')
    print(filename)
    f = open(filename).readlines()
    g = open(filename.replace('txt', 'dat'), 'w')
    numEntries = int((f[0].split())[-1])
    nPerLine = len(f[1].split())
    nLines = int(numEntries/nPerLine)
    
    if (numEntries%nPerLine)!=0: nLines = int(nLines) + 1
    wavelengths = readsection(f, 1, numEntries, nPerLine)
    data = readsection(f, 2 + nLines, numEntries, nPerLine)
    fit = readsection(f, 2*(nLines + 1) + 1, numEntries, nPerLine)
    weights = readsection(f, 3*(nLines + 1) + 1,numEntries, nPerLine)
    extinct = readsection(f, 4*(nLines + 1) + 1, numEntries, nPerLine)
    
    for j in range(len(wavelengths)):
        ostr = '%7.2f %8.6f %8.6f %8.6f %8.6f\n' % (wavelengths[j], data[j], fit[j], weights[j], extinct[j])
        g.write(ostr)
    g.close()


# # Plots

# In[ ]:


import pandas as pd


# # Objects

# In[ ]:


# choose object to be visualized

#objectx, X, Y, rSNR = index, X, Y, rSNR # all three
#objectx, X, Y, rSNR = bin1, X1, Y1, rSNR1 # first object
objectx, X, Y, rSNR = bin2, X2, Y2, rSNR2 # main object
#objectx, X, Y, rSNR = bin3, X3, Y3, rSNR3 # third object


# In[ ]:


rSNRc, Xc, Yc, core = [], [], [], [] # SNR and coordinates of the two cores
rSNRc1, Xc1, Yc1, core1 = [], [], [], [] # SNR and coordinates of the first core
rSNRc2, Xc2, Yc2, core2 = [], [], [], [] # SNR and coordinates of the second core

Y_stop = 3 # y-coordinate that separates the two bins
SNR_stop = 29 # SNR that separates the two bins

for i, j, k, l in zip(rSNR2, X2, Y2, bin2): # change "string1<<1>>" for core 1 and delete "1" for both cores
    if i > SNR_stop and k > Y_stop:
        rSNRc.append(i)
        Xc.append(j)
        Yc.append(k)
        core.append(l)

for i, j, k, l in zip(rSNR2, X2, Y2, bin2):
    if i > SNR_stop and k < Y_stop:
        rSNRc2.append(i)
        Xc2.append(j)
        Yc2.append(k)
        core2.append(l)


# In[ ]:


#objectx, X, Y, rSNR = core1, Xc1, Yc1, rSNRc1 # main core
objectx, X, Y, rSNR = core2, Xc2, Yc2, rSNRc2 # companion core


# # Metallicity

# In[ ]:


Z_sun = 0.02

az = []
for j in objectx:
    list1 = []
    list1.append(pd.read_csv(path+j+".res-AMR.dat", delim_whitespace = True, names = ["a", "b", "c", "d", "e"]))
    az.append(list1)
az = np.array(az)


ages = np.log10(az[..., 0]) # ages are the same for all parameters
ul_ages = az[..., 0] # unlogged

metallicity = np.log10(az[..., 1])
ul_metallicity = az[..., 1] # unlogged


# In[ ]:


metallicityL = az[..., 2] # unlogged
metallicityH = az[..., 3] # unlogged
metallicitymid = az[..., 4] # unlogged


# # Mass

# In[ ]:


am = []
for j in objectx:
    list1 = []
    list1.append(pd.read_csv(path+j+".res-MASS.dat", delim_whitespace = True, names = ["a", "b", "c", "d", "e"]))
    am.append(list1)
am = np.array(am)


ages = np.log10(am[..., 0])
ul_ages = am[..., 0] # unlogged

mass = np.log10(am[..., 1])
ul_mass = am[..., 1] # unlogged


# In[ ]:


massL = am[..., 2] # unlogged
massH = am[..., 3] # unlogged
massmid = am[..., 4] # unlogged


# # Flux

# In[ ]:


af = []
for j in objectx:
    list1 = []
    list1.append(pd.read_csv(path+j+".res-SAD.dat", delim_whitespace = True, names = ["a", "b", "c", "d", "e"]))
    af.append(list1)
af = np.array(af)


ages = np.log10(af[..., 0])
ul_ages = af[..., 0] # unlogged

flux = np.log10(af[..., 1])
ul_flux = af[..., 1] # unlogged


# In[ ]:


fluxL = af[..., 2] # unlogged
fluxH = af[..., 3] # unlogged
fluxmid = af[..., 4] # unlogged


# # SFR

# In[ ]:


asf = []
for j in objectx:
    list1 = []
    list1.append(pd.read_csv(path+j+".res-SFR.dat", delim_whitespace = True, names = ["a", "b", "c", "d", "e"]))
    asf.append(list1)
asf = np.array(asf)


ages = np.log10(asf[..., 0]/1000) # coverted from Myr to Gyr
ul_ages = asf[..., 0]/1000 # unlogged
r_ages = list(reversed(ul_ages[0][0])) # reversed order

SFR = np.log10(asf[..., 1])
ul_SFR = asf[..., 1] # unlogged


# In[ ]:


SFRL = asf[..., 2] # unlogged
SFRH = asf[..., 3] # unlogged
SFRmid = asf[..., 4] # unlogged


# In[ ]:


dt = [] # time steps used to calculate SFR
for j in range(0, len(objectx)):
    listt = []
    for i in range(1, len(ages[j][0])-1):
        listt.append((ages[j][0][i]+ages[j][0][i+1])/2 - (ages[j][0][i]+ages[j][0][i-1])/2)
    dt.append(listt)


# # Velocity

# In[ ]:


velocities = []
gv = []
for j, i in zip(objectx, range(0, len(objectx))):
    velocities.append([float(x.split()[0]) for x in open(path+j+".res-LOSVD.dat").readlines()])
    gv.append([float(x.split()[1]) for x in open(path+j+".res-LOSVD.dat").readlines()])


# # Fit

# In[ ]:


wv = 10 # weight vector amplifier so that one can see it clearly

wavel = []
data = []
bestfit = []
weightvector = []
extinctioncurve = []
for j in objectx[:10]: # just the first bin with highest SNR
    print(j)
    wavel.append([float(x.split()[0]) for x in open(path+j+".res-spectra.dat").readlines()])
    data.append([float(x.split()[1]) for x in open(path+j+".res-spectra.dat").readlines()])
    bestfit.append([float(x.split()[2]) for x in open(path+j+".res-spectra.dat").readlines()])
    weightvector.append([wv*float(x.split()[3]) for x in open(path+j+".res-spectra.dat").readlines()])
    extinctioncurve.append([float(x.split()[4]) for x in open(path+j+".res-spectra.dat").readlines()])


# # Results

# In[ ]:


# means of paramters over age

metallicity_m = np.mean(metallicity, axis = 0)
mass_m = np.mean(mass, axis = 0)
flux_m = np.mean(flux, axis = 0)
SFR_m = np.mean(SFR, axis = 0)


# In[ ]:


# age to redshift conversion

import astropy.units as u
from astropy.cosmology import Planck13, z_at_value

new_tick_locations = np.array([r_ages[-2], r_ages[20], r_ages[13], r_ages[12],  r_ages[10], r_ages[8], r_ages[7], r_ages[3], r_ages[1]])

def tick_function(r_ages):
    z_red = []
    for i in r_ages:
        z_red.append(float(z_at_value(Planck13.age, (13.6-i) * u.Gyr)))
    z_red = z_red
    return ["%.3f" % z for z in z_red]


# In[ ]:





# In[ ]:


from matplotlib.pyplot import cm

SNRe = np.stack((rSNR, rSNR)) # colorbar

color = iter(cm.rainbow(np.linspace(0, 1, len(rSNR)))) # colors span rSNR (add _r delete reverse to plot highest SNR bin last)

figure, axis = plt.subplots(2, 2, figsize = [20, 10])

for j, i in zip(reversed(range(0, len(objectx))), reversed(range(0, len(objectx)))): # loop over odd files
    c = next(color)
    figure.suptitle("AMR, SAD, MASS, SFR")

    axis[0, 0].plot(ages[j][0], -np.log10(Z_sun) + metallicity[j][0], color = c, marker = ".", alpha = 0.1) # , label = "AMR "+str(i+1))
    axis[0, 0].set_title("Age-Metallicity Relation "+str(len(ix))+" Bins")
    axis[0, 0].set_xlabel("Log Age (Gyr)")
    axis[0, 0].set_ylabel("$log_{10}(Z/Z_{\odot})$")
    #axis[0, 0].legend()

    axis[1, 0].plot(ages[j][0], mass[j][0], color = c, marker = ".", alpha = 0.1) #, label = "MASS "+str(i+1))
    axis[1, 0].set_title("Age-Mass Relation "+str(len(ix))+" Bins")
    axis[1, 0].set_xlabel("Log Age (Gyr)")
    axis[1, 0].set_ylabel("Mass (normalized solar masses) $log_{10}(M/M_{\odot})$")
    #axis[1, 0].legend()

    axis[0, 1].plot(ages[j][0], flux[j][0], color = c, marker = ".", alpha = 0.1)# , label = "SAD "+str(i+1))
    axis[0, 1].set_title("Age-Flux Relation "+str(len(ix))+" Bins")
    axis[0, 1].set_xlabel("Log Age (Gyr)")
    axis[0, 1].set_ylabel("Log Flux (normalized)")
    #axis[0, 1].legend()

    axis[1, 1].plot(ages[j][0], SFR[j][0], color = c, marker = ".", alpha = 0.1) #, label = "SFR "+str(i+1))
    axis[1, 1].set_title("Age-Star Formation Rate Relation "+str(len(ix))+" Bins")
    axis[1, 1].set_xlabel("Log Age (Gyr)")
    axis[1, 1].set_ylabel("Log SFR (unnormalized solar masses/yr)")
    #axis[1, 1].legend()

# means
axis[0, 0].grid()
axis[0, 0].plot(ages[0][0], -np.log10(Z_sun) + metallicity_m[0], color = "black", marker = ".", alpha = 1, label = "Mean AMR")
axis[0, 0].legend()

axis[1, 0].grid()
axis[1, 0].plot(ages[0][0], mass_m[0], color = "black", marker = ".", alpha = 1, label = "Mean MASS")
axis[1, 0].legend()

axis[0, 1].grid()
axis[0, 1].plot(ages[0][0], flux_m[0], color = "black", marker = ".", alpha = 1, label = "Mean SAD")
axis[0, 1].legend()

axis[1, 1].grid()
axis[1, 1].plot(ages[0][0], SFR_m[0], color = "black", marker = ".", alpha = 1, label = "Mean SFR")
axis[1, 1].legend()

invisible_axis = figure.add_axes([0, 0, 0.001, 0.001], visible = False)
im = invisible_axis.imshow(SNRe, cmap = "rainbow", interpolation = "bilinear", origin = "upper", vmin = rSNR[0], vmax = rSNR[-1])
cbar_ax = figure.add_axes([0.92, 0.15, 0.02, 0.7])
figure.colorbar(im, cax = cbar_ax, label = "SNR", shrink = 1)


plt.savefig("/Users/users/stoica/Pictures/Plotsf_P0001.pdf", dpi = 300, facecolor = "white")
plt.show()


# In[ ]:





# In[ ]:


r_metallicity, r_mass, r_flux, r_SFR = [], [], [], []
for j in range(0, len(metallicity)):
    list1, list2, list3, list4 = [], [], [], []
    for i in range(0, len(metallicity[j][0])):
        list1.append(list(reversed(ul_metallicity[j][0]))) # reversed order
        list2.append(list(reversed(ul_mass[j][0])))
        list3.append(list(reversed(ul_flux[j][0])))
        list4.append(list(reversed(ul_SFR[j][0])))

r_metallicity.append(list1)
r_mass.append(list2)
r_flux.append(list3)
r_SFR.append(list4)


# In[ ]:


# cumulative parameters

T_metallicity = np.cumsum(r_metallicity[0], axis = 1)
T_mass = np.cumsum(r_mass[0], axis = 1)
T_flux = np.cumsum(r_flux[0], axis = 1)
T_SFR = np.cumsum(r_SFR[0], axis = 1)


# In[ ]:


# summed over all bins

cummet = np.sum(T_metallicity, axis = 0)
cummass = np.sum(T_mass, axis = 0)
cumflux = np.sum(T_flux, axis = 0)
cumSFR = np.sum(T_SFR, axis = 0)


# In[ ]:


figure, axis = plt.subplots(2, 2, figsize = [25, 15])

figure.suptitle("Cumulative Parameters")

axis[0, 0].plot(r_ages, cummet, color = "black", marker = ".", alpha = 1, label = "Cumulative AMR")
axis[0, 0].set_title("Age-Metallicity Relation")
axis[0, 0].set_xlabel("Lookback Time (Gyr)") # reversed age so right is present and left is Big Bang
axis[0, 0].set_ylabel("$Z/Z_{\odot}$")
axis[0, 0].legend()
axis[0, 0].grid()

ax21 = axis[0, 0].twiny()
ax21.set_xlim(axis[0, 0].get_xlim())
ax21.set_xticks(new_tick_locations)
ax21.set_xticklabels(tick_function(new_tick_locations))
ax21.set_xlabel("Redshift z")


axis[1, 0].plot(r_ages, cummass, color = "black", marker = ".", alpha = 1, label = "Cumulative MASS")
axis[1, 0].set_title("Age-Mass Relation")
axis[1, 0].set_xlabel("Lookback Time (Gyr)")
axis[1, 0].set_ylabel("Mass (normalized solar masses) $\Delta M/(M_{tot}/M_{\odot})$")
axis[1, 0].legend()
axis[1, 0].grid()

ax22 = axis[1, 0].twiny()
ax22.set_xlim(axis[1, 0].get_xlim())
ax22.set_xticks(new_tick_locations)
ax22.set_xticklabels(tick_function(new_tick_locations))
ax22.set_xlabel("Redshift z")


axis[0, 1].plot(r_ages, cumflux, color = "black", marker = ".", alpha = 1 , label = "Cumulative SAD")
axis[0, 1].set_title("Age-Flux Relation")
axis[0, 1].set_xlabel("Lookback Time (Gyr)")
axis[0, 1].set_ylabel("Flux (normalized)")
axis[0, 1].legend()
axis[0, 1].grid()

ax23 = axis[0, 1].twiny()
ax23.set_xlim(axis[0, 1].get_xlim())
ax23.set_xticks(new_tick_locations)
ax23.set_xticklabels(tick_function(new_tick_locations))
ax23.set_xlabel("Redshift z")


axis[1, 1].plot(r_ages, cumSFR, color = "black", marker = ".", alpha = 1, label = "Cumulative SFR")
axis[1, 1].set_title("Age-Star Formation Rate Relation")
axis[1, 1].set_xlabel("Lookback Time (Gyr)")
axis[1, 1].set_ylabel("SFR (unnormalized solar masses/yr)")
axis[1, 1].legend()
axis[1, 1].grid()

ax24 = axis[1, 1].twiny()
ax24.set_xlim(axis[1, 1].get_xlim())
ax24.set_xticks(new_tick_locations)
ax24.set_xticklabels(tick_function(new_tick_locations))
ax24.set_xlabel("Redshift z")


plt.tight_layout()
plt.savefig("/Users/users/stoica/Pictures/Cumulativef_P0001.svg", dpi = 300, facecolor = "white")
plt.show()


# In[ ]:





# In[ ]:


# fractional parameters

F_metallicity = []
F_mass = []
F_flux = []
F_SFR = []
for i in range(0, len(cummet)):
    F_metallicity.append(cummet[i]/max(cummet))
    F_mass.append(cummass[i]/max(cummass))
    F_flux.append(cumflux[i]/max(cumflux))
    F_SFR.append(cumSFR[i]/max(cumSFR))


figure, axis = plt.subplots(2, 2, figsize = [25, 15])

figure.suptitle("Fractional Parameters")

axis[0, 0].plot(r_ages, F_metallicity, color = "black", marker = ".", alpha = 1, label = "Fractional AMR")
axis[0, 0].set_title("Age-Metallicity Relation")
axis[0, 0].set_xlabel("Lookback Time (Gyr)")
axis[0, 0].set_ylabel("$Z/Z_{\odot}$")
axis[0, 0].legend()
axis[0, 0].grid()

ax21 = axis[0, 0].twiny()
ax21.set_xlim(axis[0, 0].get_xlim())
ax21.set_xticks(new_tick_locations)
ax21.set_xticklabels(tick_function(new_tick_locations))
ax21.set_xlabel("Redshift z")


axis[1, 0].plot(r_ages, F_mass, color = "black", marker = ".", alpha = 1, label = "Fractional MASS")
axis[1, 0].axhline(y = 0.9, color = "blue", linestyle = '--', label = f"90\% of mass")
axis[1, 0].set_title("Age-Mass Relation")
axis[1, 0].set_xlabel("Lookback Time (Gyr)")
axis[1, 0].set_ylabel("Mass (normalized solar masses) $M/M_{\odot}$")
axis[1, 0].fill_between(x = r_ages, y1 = F_mass, where = (r_ages[11] < r_ages)&(r_ages <= r_ages[0]), color = "slateblue", alpha = 0.2)
axis[1, 0].legend()
axis[1, 0].grid()

ax22 = axis[1, 0].twiny()
ax22.set_xlim(axis[1, 0].get_xlim())
ax22.set_xticks(new_tick_locations)
ax22.set_xticklabels(tick_function(new_tick_locations))
ax22.set_xlabel("Redshift z")


axis[0, 1].plot(r_ages, F_flux, color = "black", marker = ".", alpha = 1 , label = "Fractional SAD")
axis[0, 1].axhline(y = 0.9, color = "blue", linestyle = '--', label = f"90\% of flux")
axis[0, 1].set_title("Age-Flux Relation")
axis[0, 1].set_xlabel("Lookback Time (Gyr)")
axis[0, 1].set_ylabel("Flux (normalized)")
axis[0, 1].fill_between(x = r_ages, y1 = F_flux, where = (r_ages[17] < r_ages)&(r_ages <= r_ages[0]), color = "slateblue", alpha = 0.2)
axis[0, 1].legend()
axis[0, 1].grid()

ax23 = axis[0, 1].twiny()
ax23.set_xlim(axis[0, 1].get_xlim())
ax23.set_xticks(new_tick_locations)
ax23.set_xticklabels(tick_function(new_tick_locations))
ax23.set_xlabel("Redshift z")


axis[1, 1].plot(r_ages, F_SFR, color = "black", marker = ".", alpha = 1, label = "Fractional SFR")
axis[1, 1].axhline(y = 0.9, color = "blue", linestyle = '--', label = f"90\% of SFR")
axis[1, 1].set_title("Age-Star Formation Rate Relation")
axis[1, 1].set_xlabel("Lookback Time (Gyr)")
axis[1, 1].set_ylabel("SFR (unnormalized solar masses/yr)")
axis[1, 1].fill_between(x = r_ages, y1 = F_SFR, where = (r_ages[16] < r_ages)&(r_ages <= r_ages[0]), color = "slateblue", alpha = 0.2)
axis[1, 1].legend()
axis[1, 1].grid()

ax24 = axis[1, 1].twiny()
ax24.set_xlim(axis[1, 1].get_xlim())
ax24.set_xticks(new_tick_locations)
ax24.set_xticklabels(tick_function(new_tick_locations))
ax24.set_xlabel("Redshift z")


plt.tight_layout()
plt.savefig("/Users/users/stoica/Pictures/Fractionalf_P0001.svg", dpi = 300, facecolor = "white")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:


# re-run bins with new object without overwriting the previous one


# In[ ]:


F_mass2 = []
F_flux2 = []
for i in range(0, len(cummass)):
    F_mass2.append(cummass[i]/max(cummass))
    F_flux2.append(cumflux[i]/max(cumflux))


# In[ ]:


figure, axis = plt.subplots(2, 2, figsize = [25, 15])

figure.suptitle("Fractional Parameters")

axis[0, 0].plot(r_ages, F_mass, color = "black", marker = ".", alpha = 1, label = "Fractional AMR")
axis[0, 0].axhline(y = 0.9, color = "blue", linestyle = '--', label = f"90\% of mass")
axis[0, 0].set_title("Age-Mass Relation of Core 1")
axis[0, 0].set_xlabel("Lookback Time (Gyr)")
axis[0, 0].set_ylabel("$M/M_{\odot}$")
axis[0, 0].fill_between(x = r_ages, y1 = F_mass, where = (r_ages[9] < r_ages)&(r_ages <= r_ages[0]), color = "slateblue", alpha = 0.2)
axis[0, 0].legend()
axis[0, 0].grid()

ax21 = axis[0, 0].twiny()
ax21.set_xlim(axis[0, 0].get_xlim())
ax21.set_xticks(new_tick_locations)
ax21.set_xticklabels(tick_function(new_tick_locations))
ax21.set_xlabel("Redshift z")


axis[0, 1].plot(r_ages, F_flux, color = "black", marker = ".", alpha = 1, label = "Fractional MASS")
axis[0, 1].axhline(y = 0.9, color = "blue", linestyle = '--', label = f"90\% of mass")
axis[0, 1].set_title("Age-Flux Relation of Core 1")
axis[0, 1].set_xlabel("Lookback Time (Gyr)")
axis[0, 1].set_ylabel("Flux (normalized)")
axis[0, 1].fill_between(x = r_ages, y1 = F_flux, where = (r_ages[14] < r_ages)&(r_ages <= r_ages[0]), color = "slateblue", alpha = 0.2)
axis[0, 1].legend()
axis[0, 1].grid()

ax22 = axis[0, 1].twiny()
ax22.set_xlim(axis[0, 1].get_xlim())
ax22.set_xticks(new_tick_locations)
ax22.set_xticklabels(tick_function(new_tick_locations))
ax22.set_xlabel("Redshift z")



axis[1, 0].plot(r_ages, F_mass2, color = "black", marker = ".", alpha = 1 , label = "Fractional SAD")
axis[1, 0].axhline(y = 0.9, color = "blue", linestyle = '--', label = f"90\% of flux")
axis[1, 0].set_title("Age-Mass Relation of Core 2")
axis[1, 0].set_xlabel("Lookback Time (Gyr)")
axis[1, 0].set_ylabel("$M/M_{\odot}$")
axis[1, 0].fill_between(x = r_ages, y1 = F_mass2, where = (r_ages[8] < r_ages)&(r_ages <= r_ages[0]), color = "slateblue", alpha = 0.2)
axis[1, 0].legend()
axis[1, 0].grid()

ax23 = axis[1, 0].twiny()
ax23.set_xlim(axis[1, 0].get_xlim())
ax23.set_xticks(new_tick_locations)
ax23.set_xticklabels(tick_function(new_tick_locations))
ax23.set_xlabel("Redshift z")


axis[1, 1].plot(r_ages, F_flux2, color = "black", marker = ".", alpha = 1, label = "Fractional SFR")
axis[1, 1].axhline(y = 0.9, color = "blue", linestyle = '--', label = f"90\% of SFR")
axis[1, 1].set_title("Age-Flux Relation of Core 2")
axis[1, 1].set_xlabel("Lookback Time (Gyr)")
axis[1, 1].set_ylabel("Flux (normalized)")
axis[1, 1].fill_between(x = r_ages, y1 = F_flux2, where = (r_ages[13] < r_ages)&(r_ages <= r_ages[0]), color = "slateblue", alpha = 0.2)
axis[1, 1].legend()
axis[1, 1].grid()

ax24 = axis[1, 1].twiny()
ax24.set_xlim(axis[1, 1].get_xlim())
ax24.set_xticks(new_tick_locations)
ax24.set_xticklabels(tick_function(new_tick_locations))
ax24.set_xlabel("Redshift z")


plt.tight_layout()
plt.savefig("/Users/users/stoica/Pictures/Fractionalf1-21.svg", dpi = 300, facecolor = "white")
plt.show()


# In[ ]:





# # Weighted Parameters

# In[ ]:


# time stepts for integration (equal in logarithmic space)

dtt = []
for j in range(0, len(objectx)):
    listt = []
    for i in range(1, len(ages[j][0])-1):
        listt.append((ages[j][0][i+1]-ages[j][0][i]))
    dtt.append(listt)

dt_m = np.mean(dtt)


# In[ ]:


t_m = [] # mass weighted age
for j in range(0, len(objectx)):
    t_m.append(np.log10((np.sum(10**mass[j][0] * 10**ages[j][0] * 10**dt_m))/(np.sum(10**mass[j][0] * 10**dt_m))))


t_f = [] # flux weighted age
for j in range(0, len(objectx)):
    t_f.append(np.log10((np.sum(10**flux[j][0] * 10**ages[j][0] * 10**dt_m))/(np.sum(10**flux[j][0] * 10**dt_m))))


Z_sun = 0.02
Z_m = [] # mass weighted metallicity
for j in range(0, len(objectx)):
    Z_m.append(-np.log10(Z_sun) + np.log10((np.sum(10**metallicity[j][0] * 10**mass[j][0] * 10**dt_m))/(np.sum(10**mass[j][0] * 10**dt_m))))


# In[ ]:


from math import pi

xm = X
ym = Y
zm = rSNR

#a = 2
#b = 4
#t = np.linspace(0, 2*pi, 100)


plt.figure(figsize = (15, 15))
plt.scatter(xm, ym, c = zm, cmap = "turbo", s = 100, marker = ",")
#plt.plot(X2[0] + a*np.cos(t) , Y2[0] + b*np.sin(t), color = "white", linewidth = 5)
plt.title("Signal-to-Noise Ratio")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "SNR")
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/SNR_P0001f.svg", dpi = 300, facecolor = "white")
plt.show()
################


# In[ ]:





# In[ ]:


xm = Xc
ym = Yc
zm = rSNRc


plt.figure(figsize = (8, 9))
plt.scatter(xm, ym, c = zm, cmap = "turbo", s = 500, marker = ",")
plt.title("Signal-to-Noise Ratio of the Cores")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "SNR")
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/SNR_Cores.svg", dpi = 300, facecolor = "white")
plt.show()
################


# In[ ]:





# In[ ]:


xm = X
ym = Y
zm = Z_m

plt.figure(figsize = (15, 15))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 100, marker = ",")
plt.title("Mass Weighted Metallicity")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "$log(Z)$") # added log(Z_sun) to be left with log(Z)
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/MassZ_P0001f.svg", dpi = 300, facecolor = "white")
plt.show()
#################


# In[ ]:





# In[ ]:


Z_mc = []
for i, j in zip(rSNR2, Z_m):
    if i > SNR_stop:
        Z_mc.append(j)

xm = Xc
ym = Yc
zm = Z_mc

plt.figure(figsize = (8, 9))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 500, marker = ",")
plt.title("Mass Weighted Metallicity of the Cores")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "$log(Z)$") # added log(Z_sun) to be left with log(Z)
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/MassZ_P0001Cores.svg", dpi = 300, facecolor = "white")
plt.show()
#################


# In[ ]:





# In[ ]:


xm = X
ym = Y
zm = t_m

plt.figure(figsize = (15, 15))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 100, marker = ",")
plt.title("Mass Weighted Age")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "Log Gyr")
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/MassAge_P0001f.svg", dpi = 300, facecolor = "white")
plt.show()
################


# In[ ]:





# In[ ]:


t_mc = []
for i, j in zip(rSNR2, t_m):
    if i > SNR_stop:
        t_mc.append(j)

xm = Xc
ym = Yc
zm = t_mc

plt.figure(figsize = (8, 9))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 500, marker = ",")
plt.title("Mass Weighted Age of the Cores")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "Log Gyr")
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/MassAge_P0001Cores.svg", dpi = 300, facecolor = "white")
plt.show()
################


# In[ ]:





# In[ ]:


xm = X
ym = Y
zm = t_f

plt.figure(figsize = (15, 15))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 100, marker = ",")
plt.title("Flux Weighted Age")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "Log Gyr")
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/FluxAge_P0001f.svg", dpi = 300, facecolor = "white")
plt.show()
################


# In[ ]:





# In[ ]:


t_fc = []
for i, j in zip(rSNR2, t_f):
    if i > SNR_stop:
        t_fc.append(j)

xm = Xc
ym = Yc
zm = t_fc

plt.figure(figsize = (8, 9))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 500, marker = ",")
plt.title("Flux Weighted Age of the Cores")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "Log Gyr")
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/FluxAge_P0001Cores.svg", dpi = 300, facecolor = "white")
plt.show()
################


# In[ ]:





# In[ ]:


# velocity plots of both cores

v = []
for j in range(0, len(objectx)):
    v.append(np.linspace(velocities[j][0], velocities[j][-1]))

velocitiesc1, gvc1, vc1 = [], [], [] # first core
velocitiesc2, gvc2, vc2 = [], [], [] # second core

for i, j, k, l, n in zip(rSNR2, Y2, velocities, gv, v):
    if i > 0:# and j > Y_stop:
        velocitiesc1.append(k)
        gvc1.append(l)
        vc1.append(n)

for i, j, k, l, n in zip(rSNR2, Y2, velocities, gv, v):
    if i > SNR_stop and j < Y_stop:
        velocitiesc2.append(k)
        gvc2.append(l)
        vc2.append(n)


# In[ ]:


from scipy.optimize import curve_fit
from scipy.signal import argrelextrema


max11 = []
for i in range(0, len(gvc1)):
    max11.append(argrelextrema(np.array(gvc1[i]), np.greater, order = 10))

max1 = argrelextrema(np.array(gvc1[0]), np.greater, order = 10)

p1 = []
t1 = []
for j in range(0, len(max11)):
    pc = []
    tc = []
    for i in max11[j][0]:
        pc.append(gvc1[j][i])
        tc.append(velocitiesc1[j][i])
    p1.append(pc)
    t1.append(tc)

p = []
t = []
for i in max1[0]:
    p.append(gvc1[0][i])
    t.append(velocitiesc1[0][i])

z1 = []
for i in range(0, len(t1)):
    z1.append(np.linspace(t1[i][0], t1[i][-1]))
    
z = np.linspace(t[0], t[-1])

popt1 = []
pcov1 = []
popt11 = []
for i in range(0, len(t1)):
    def Gaussian1(z1, A1, mu1, sigma1):
        return A1 * np.exp(-0.5*((z1-mu1)**2)/(sigma1**2))
    p01 = [np.max(p1[i]), np.mean(t1[i]), np.std(t1[i])]
    popt1, pcov1 = curve_fit(Gaussian1, t1[i], p1[i], p0 = p01, maxfev = 1000000000)
    popt11.append(popt1)

def Gaussian12(z, A1, mu1, sigma1):
    return A1 * np.exp(-0.5*((z-mu1)**2)/(sigma1**2))
p01 = [0.005, 500, 500]
popt12, pcov12 = curve_fit(Gaussian12, t, p, p0 = p01, maxfev = 1000000000)


# In[ ]:


sigma_m = []
for i in range(0, len(popt11)):
    sigma_m.append(popt11[i][-1])

for j in range(0, len(sigma_m)):
        if sigma_m[j] > 1000:
            sigma_m[j] = 0

mean_m = []
for i in range(0, len(popt11)):
    mean_m.append(popt11[i][-2])

for j in range(0, len(mean_m)):
        if mean_m[j] > 1000 or mean_m[j] < -1000:
            mean_m[j] = 0


# In[ ]:


xm = X
ym = Y
zm = mean_m

plt.figure(figsize = (15, 15))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 100, marker = ",")
plt.title("Velocity Dispersion")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "v (km/s)") # added log(Z_sun) to be left with log(Z)
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/MEAN.svg", dpi = 300, facecolor = "white")
plt.show()
#################


# In[ ]:





# In[ ]:


xm = X
ym = Y
zm = sigma_m

plt.figure(figsize = (15, 15))
plt.scatter(xm, ym, c = zm, cmap = "inferno", s = 100, marker = ",")
plt.title("Velocity Dispersion")
plt.xlabel("X (arcsec)")
plt.ylabel("Y (arcsec)")
plt.colorbar(label = "$\sigma (km/s)$") # added log(Z_sun) to be left with log(Z)
plt.grid()
plt.savefig("/Users/users/stoica/Pictures/SIGMA.svg", dpi = 300, facecolor = "white")
plt.show()
#################


# In[ ]:





# In[ ]:


max2 = argrelextrema(np.array(gvc2[0]), np.greater, order = 10)

p2 = []
t2 = []
for i in max2[0]:
    p2.append(gvc2[0][i])
    t2.append(velocitiesc2[0][i])

z2 = np.linspace(t2[0], t2[-1])

def Gaussian2(z2, A2, mu2, sigma2):
    return A2 * np.exp(-0.5*((z2-mu2)**2)/(sigma2**2))
p02 = [0.005, 0, 500]
popt2, pcov2 = curve_fit(Gaussian2, t2, p2, p0 = p02, maxfev = 1000000000)


# In[ ]:


figure, axis = plt.subplots(1, 2, figsize = [20, 5])

for j, k in zip(range(0, len(velocitiesc1)), range(0, len(velocitiesc2))):
    
    figure.suptitle("Velocities of Two Cores")
    
    axis[0].plot(velocitiesc1[j], gvc1[j], color = "tomato", marker = ".")
    
    axis[1].plot(velocitiesc2[k], gvc2[k], color = "tomato", marker = ".")

axis[0].plot(velocitiesc1[0], gvc1[0], color = "tomato", marker = ".", label = "LOSVD")
axis[0].plot(z, Gaussian1(z, *popt1), color = "black", label = f"Mean Velocity = {popt1[1]:.2f} km/s; Velocity Dispersion = {popt1[2]:.2f} km/s")
axis[0].set_title("Velocity-Broadening Function Relation of Main Core")
axis[0].set_xlabel("Velocity (km/s)")
axis[0].set_ylabel("Broadening Function g(v) (normalized)")
axis[0].grid()
axis[0].legend()

axis[1].plot(velocitiesc2[0], gvc2[0], color = "tomato", marker = ".", label = "LOSVD")
axis[1].plot(z2, Gaussian1(z2, *popt2), color = "black", label = f"Mean Velocity = {popt2[1]:.2f} km/s; Velocity Dispersion = {popt2[2]:.2f} km/s")
axis[1].set_title("Velocity-Broadening Function Relation of Secondary Core")
axis[1].set_xlabel("Velocity (km/s)")
axis[1].set_ylabel("Broadening Function g(v) (normalized)")
axis[1].grid()
axis[1].legend()

plt.savefig("/Users/users/stoica/Pictures/VelocityD.svg", dpi = 300, facecolor = "white")
plt.show()


# In[ ]:





# In[ ]:


# fit

figure, axis = plt.subplots(1, 1, figsize = [20, 5])

for j in range(0, len(objectx[:1])): # just the first bin with highest SNR

    #figure.suptitle(f"L2 = {pathfits} (Plot {j+1})")

    axis.plot(wavel[j], data[j], color = "firebrick", label = "spectra")
    axis.plot(wavel[j], bestfit[j], color = "limegreen", label = "Bestfit")
    axis.plot(wavel[j], weightvector[j], color = "blue", alpha = 0.5, label = f"Weight Vector (times {wv})")
    axis.plot(wavel[j], extinctioncurve[j], color = "black", label = "Extinction Curve")
    axis.set_title("Wavelength-Data Relation"+" Bin ("+str(ix[j])+")")

axis.set_xlabel("Wavelength (Å)")
axis.set_ylabel("Data")
axis.grid()
#axis.axvline(x = 3969, color = "orange", linestyle = '--', label = "Ca H")
#axis.axvline(x = 3934, color = "red", linestyle = '--', label = "Ca K")
axis.legend()

plt.savefig("/Users/users/stoica/Pictures/Fit_P0001f.svg", dpi = 300, facecolor = "white")
plt.show()


# In[ ]:





# # Monte Carlo

# In[ ]:


objectx = jj # first and middle bins


# In[ ]:


from statistics import median, mode

v_mc = MCc[4]
metallicity_mc = MCc[0]
mass_mc = MCc[1]
flux_mc = MCc[2]
SFR_mc = MCc[3]


# In[ ]:


plt.figure(figsize = [20, 8])
for i in range(0, len(mass_mc)):
    plt.plot(ul_ages[0][0], mass_mc[i], color = "orange", alpha = 0.5)

plt.plot(ul_ages[0][0], mass_mc[0], color = "orange", alpha = 0.5, label = "MC Simulation")
plt.plot(ul_ages[0][0], ul_mass[0][0], color = "blue", label = "STECKMAP")
#plt.plot(ul_ages[0][0], massL[0][0], color = "gold", label = "MC Simulation (Low sigma)")
#plt.plot(ul_ages[0][0], massH[0][0], color = "crimson", label = "MC Simulation (High sigma)")
#plt.plot(ul_ages[0][0], massmid[0][0], color = "green", label = "MC Simulation (Mid sigma)")
plt.title("Monte Carlo Simulation for Mass of Highest SNR Bin")
plt.xlabel("Lookback Time (Gyr)")
plt.ylabel("$M/M_{\odot}$")
plt.grid()
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize = [20, 8]) # change range and parameter
for i in range(22, 23):
    plt.hist(flux_mc.T[i], color = "black", alpha = 0.5, label = "Distribution of MC Simulated Values")
    plt.axvline(ul_flux[0][0][i], color = "crimson", label = "MAP")
    plt.axvline(np.mean(flux_mc.T[i]), color = "blue", linestyle = "--", label = "Mean")
    plt.axvline(median(flux_mc.T[i]), color = "limegreen", linestyle = "-.", label = "Median")

plt.title("Monte Carlo Simulation for Parameter of Highest SNR Bin")
plt.xlabel("Parameter")
plt.ylabel("Bins")
plt.grid(alpha = 0.2)
plt.legend()
plt.show()


# In[ ]:


figure, axis = plt.subplots(2, 3, figsize = [25, 15])

figure.suptitle("Monte Carlo Simulations of Highest SNR Bin")

axis[0, 0].hist(flux_mc.T[0], color = "mediumslateblue", alpha = 0.5, label = "Distribution of MC Simulated Values")
axis[0, 0].axvline(ul_flux[0][0][0], color = "red", label = "MAP")
axis[0, 0].axvline(mode(flux_mc.T[0]), color = "blue", linestyle = "--", label = "Mode")
axis[0, 0].axvline(median(flux_mc.T[0]), color = "limegreen", linestyle = "-.", label = "Median")
axis[0, 0].axvline(np.mean(flux_mc.T[0]), color = "darkorange", linestyle = ":", label = "Mean")
axis[0, 0].set_title(f"{(ul_ages[0][0][0])*1000} Myr")
axis[0, 0].set_xlabel("Flux")
axis[0, 0].set_ylabel("Bins")
axis[0, 0].grid(alpha = 0.2)
axis[0, 0].legend()

axis[0, 1].hist(flux_mc.T[15], color = "mediumslateblue", alpha = 0.5, label = "Distribution of MC Simulated Values")
axis[0, 1].axvline(ul_flux[0][0][15], color = "red", label = "MAP")
axis[0, 1].axvline(mode(flux_mc.T[15]), color = "blue", linestyle = "--", label = "Mode")
axis[0, 1].axvline(median(flux_mc.T[15]), color = "limegreen", linestyle = "-.", label = "Median")
axis[0, 1].axvline(np.mean(flux_mc.T[15]), color = "darkorange", linestyle = ":", label = "Mean")
axis[0, 1].set_title(f"{(ul_ages[0][0][15])} Gyr")
axis[0, 1].set_xlabel("Flux")
axis[0, 1].set_ylabel("Bins")
axis[0, 1].grid(alpha = 0.2)
axis[0, 1].legend()

axis[0, 2].hist(flux_mc.T[29], color = "mediumslateblue", alpha = 0.5, label = "Distribution of MC Simulated Values")
axis[0, 2].axvline(ul_flux[0][0][29], color = "red", label = "MAP")
axis[0, 2].axvline(mode(flux_mc.T[29]), color = "blue", linestyle = "--", label = "Mode")
axis[0, 2].axvline(median(flux_mc.T[29]), color = "limegreen", linestyle = "-.", label = "Median")
axis[0, 2].axvline(np.mean(flux_mc.T[29]), color = "darkorange", linestyle = ":", label = "Mean")
axis[0, 2].set_title(f"{(ul_ages[0][0][29])} Gyr")
axis[0, 2].set_xlabel("Flux")
axis[0, 2].set_ylabel("Bins")
axis[0, 2].grid(alpha = 0.2)
axis[0, 2].legend()


axis[1, 0].hist(mass_mc.T[0], color = "mediumslateblue", alpha = 0.5, label = "Distribution of MC Simulated Values")
axis[1, 0].axvline(ul_mass[0][0][0], color = "red", label = "MAP")
axis[1, 0].axvline(mode(mass_mc.T[0]), color = "blue", linestyle = "--", label = "Mode")
axis[1, 0].axvline(median(mass_mc.T[0]), color = "limegreen", linestyle = "-.", label = "Median")
axis[1, 0].axvline(np.mean(mass_mc.T[0]), color = "darkorange", linestyle = ":", label = "Mean")
axis[1, 0].set_title(f"{(ul_ages[0][0][0])*1000} Myr")
axis[1, 0].set_xlabel("Mass")
axis[1, 0].set_ylabel("Bins")
axis[1, 0].grid(alpha = 0.2)
axis[1, 0].legend()

axis[1, 1].hist(mass_mc.T[15], color = "mediumslateblue", alpha = 0.5, label = "Distribution of MC Simulated Values")
axis[1, 1].axvline(ul_mass[0][0][15], color = "red", label = "MAP")
axis[1, 1].axvline(mode(mass_mc.T[15]), color = "blue", linestyle = "--", label = "Mode")
axis[1, 1].axvline(median(mass_mc.T[15]), color = "limegreen", linestyle = "-.", label = "Median")
axis[1, 1].axvline(np.mean(mass_mc.T[15]), color = "darkorange", linestyle = ":", label = "Mean")
axis[1, 1].set_title(f"{(ul_ages[0][0][15])} Gyr")
axis[1, 1].set_xlabel("Mass")
axis[1, 1].set_ylabel("Bins")
axis[1, 1].grid(alpha = 0.2)
axis[1, 1].legend()

axis[1, 2].hist(mass_mc.T[29], color = "mediumslateblue", alpha = 0.5, label = "Distribution of MC Simulated Values")
axis[1, 2].axvline(ul_mass[0][0][29], color = "red", label = "MAP")
axis[1, 2].axvline(mode(mass_mc.T[29]), color = "blue", linestyle = "--", label = "Mode")
axis[1, 2].axvline(median(mass_mc.T[29]), color = "limegreen", linestyle = "-.", label = "Median")
axis[1, 2].axvline(np.mean(mass_mc.T[29]), color = "darkorange", linestyle = ":", label = "Mean")
axis[1, 2].set_title(f"{(ul_ages[0][0][29])} Gyr")
axis[1, 2].set_xlabel("Mass")
axis[1, 2].set_ylabel("Bins")
axis[1, 2].grid(alpha = 0.2)
axis[1, 2].legend()


plt.tight_layout()
plt.savefig("/Users/users/stoica/Pictures/MC.svg", dpi = 300, facecolor = "white")
plt.show()


# In[ ]:




