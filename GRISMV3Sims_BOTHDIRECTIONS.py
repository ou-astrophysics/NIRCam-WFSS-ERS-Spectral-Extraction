# Import Packages

import numpy as np
import pandas as pd
import astropy.io.fits as apfits
import astropy.visualization as apvis
import astropy.nddata as apnndd
import astropy.table as aptable
import astropy.wcs as apwcs
import astropy.units as apunits
import astropy.coordinates as apcoords
import matplotlib.pyplot as mplplot
import matplotlib.patches as mplpatches
import glob
import os
import scipy as sp
import scipy.optimize as spopt
import scipy.interpolate as spinterp
import copy
import grismconf
import sys 

file_name = sys.argv[1]

fits_file = apfits.open(file_name)
data = fits_file["SCI"].data

filter_ = fits_file['PRIMARY'].header['FILTER'].strip()
# [-1] needed to extract just either R or C
# CHANGE DIRECTION TO GRISM FOR CLARITY
direction = fits_file['PRIMARY'].header['PUPIL'].strip()[-1]
module = fits_file['PRIMARY'].header['MODULE'].strip()
dither = fits_file['PRIMARY'].header['PATT_NUM']
# SUBPIXNUM Missing from new sims!!!
# subpixel = Frame['PRIMARY'].header['SUBPXNUM']

# As sub-pix missing, use exposure number to distinguish frames
exposure = fits_file['PRIMARY'].header['EXPOSURE']

# All 192 simulated frames produced by MIRAGE - total = 192 frames as 4 observations each have 2 Grisms, 2 modules, 3 Primary Dithers each with 4 subpixel dithers.
# Of the 4 observations 2 are with the F322W2 (Water) Filter and F444W (CO+CO2 filter), making 96 frames per filter. This is how the folders are split.
frameFitsDir = f"/STEM/scratch.san/zx446701/SimsV2/{filter_}/{filter_}_grism/{filter_}_Level15_Frames/"
# Source list for every frame within the simulation - WANT TO CREATE OUR OWN SOURCE FROM DIRECT IMAGES
frameListDir = f"/STEM/scratch.san/zx446701/SimsV2/{filter_}/{filter_}_grism/{filter_}_List"

# Use source code to look into utils etc. - Ask Hugh how he knows when to look for these libraries!
from webbpsf.utils import to_griddedpsfmodel
from webbpsf.gridded_library import display_psf_grid
# How do we justify the model number (1-5??)
# Take specific instrument, module, filter and detector PSF fits files
# and turn into a grid of how the PSF changes with position on detector array
# NUMBER OF DETECTOR IS IN NRCA5 - 5 IS LW DETECTOR 1-4 IS SW
# ONLY 5 REQUIRED FOR WFSS AS ONLY DETECTOR ABLE TO DO THIS
if filter_ == 'F322W2':
    grid = to_griddedpsfmodel( f"/STEM/scratch.san/zx446701/mirage_data/nircam/gridded_psf_library/nircam_nrc{module.lower()}5_{filter_.lower()}_clear_fovp47_samp5_npsf16_requirements_realization0.fits"
    )
else:
    grid = to_griddedpsfmodel( f"/STEM/scratch.san/zx446701/mirage_data/nircam/gridded_psf_library/nircam_nrc{module.lower()}5_{filter_.lower()}_clear_fovp61_samp5_npsf16_requirements_realization0.fits"
    )

## Function extracting Sky Coordinates from source list

# - Identify all the values within col 3 and 4 of pandas df
#         sourceList.loc[:, [3, 4]]
# - Turns this into a numpy array
#         sourceList.loc[:, [3, 4]].to_numpy()
# - Finally tranpose the the 2D array to a list of RA values and Dec values
#         sourceList.loc[:, [3, 4]].to_numpy().T
# - Starred list indicates 0 and 1 lists within of 2d array ([[0],[1]]) inputted here so every sources RA and Dec can be inputted and coverted into astropy SkyCoord form
#         apcoords.SkyCoord(*sourceList.loc[:, [3, 4]].to_numpy().T, frame=apcoords.ICRS, unit=apunits.deg)

def getSourceListForImage(image, frameListDir):
    listPath = os.path.join(
        frameListDir,
        f"{'_'.join(os.path.basename(image).split('_')[:4])}_uncal_pointsources.list",
    )
    return pd.read_csv(listPath, delim_whitespace=True, comment="#", header=None)

def getSourceCoordsForImage(image, frameListDir):
    sourceList = getSourceListForImage(image, frameListDir) # Previous Function
    coords = apcoords.SkyCoord(
        *sourceList.loc[:, [3, 4]].to_numpy().T, frame=apcoords.ICRS, unit=apunits.deg
    )
    return coords

def getSourcePixelsForImage(image, frameListDir):
    sourceList = getSourceListForImage(image, frameListDir)
    pixels = sourceList.loc[:, [5, 6]].to_numpy()
    
    return pixels

def computeTrace(pixels, fac=100, filter_="F322W2", module="A", direction="R", simYDisp=False, order=1):
    # Locate config File for the module and grism direction 
    confFile = f"/STEM/scratch.san/zx446701/mirage_data/nircam/gridded_psf_library/nircam_nrc{module.lower()}5_{filter_.lower()}_clear_fovp47_samp5_npsf16_requirements_realization0.fits"
    # Class to read and hold GRISM configuration info
    conf = grismconf.Config(confFile)
    # Found from GRISMCONF README file - see link above
    # Middle section - number of pixels from end to start in X direction
    # 1/ middle = slighting trace by number of pixels
    # /fac is for splitting by subpixel amounts and oversampling
    dt = np.abs(1 / (1 + conf.DISPX(f'+{order}', *pixels, 1) - conf.DISPX(f'+{order}', *pixels, 0)) / fac)
    # t is the trace and how much of it is covered (0 to 1 is the full trace)
    t = np.arange(0, 1, dt)

    # DISP(X,Y,L) = DISPERSION POLYNOMIAL (X direction, Y, Full Length)
    # order, x0, y0, steps along dispersion between 0 and 1
    # X disp polynomial
    dxs = conf.DISPX(f"+{order}" *pixels, t)
    # Y disp polynomial 
    dys = conf.DISPY(f"+{order}" *pixels, t)
    # Compute wavelength of each pixel
    wavs = conf.DISPL(f"+{order}" *pixels, t)

    return (
        pixels[0] + dxs,
        pixels[1] + dys if simYDisp else np.full_like(dys, pixels[1]),
        wavs,
    )

def computeTraceWLR(pixels, filter_="F322W2", module="A", simYDisp=False, order=1):
    # Locate config File for the filter, module and grism direction 
    confFile = f"/STEM/scratch.san/zx446701/GRISM_NIRCAM/V3/NIRCAM_{filter_}_mod{module}_R.conf"
    # Class to read and hold GRISM configuration info
    conf = grismconf.Config(confFile)

    # dt = 1/(conf.DISPX(f'+{order}', *pixels,0)-conf.DISPX(f'+{order}', *pixels,1)) / fac
    # t = np.arange(0,1,dt)

    # 
    dxs0 = conf.DISPX(f'+{order}',*pixels, 0)

    #
    dxs1 = conf.DISPX(f'+{order}', *pixels, 1)
    
    # np.floor rounds to lower value
    # np.ceil rounds to higher value
    # Want this so full trace is picked up in all pixels
    if module == "A":
    # FOR MODULE A IN DIRECTION R
    # +1 REQUIRED DUE TO ARANGE NEEDING TO INCLUDE LAST TRACE PIXEL NUMBER
        dxs = np.arange(np.floor(dxs0),np.ceil(dxs1)+1).astype(int)

    else:
        # if dxs is decreasing then switch floor and ceil
        dxs = np.arange(np.floor(dxs1),np.ceil(dxs0)+1).astype(int)
    
    ts = conf.INVDISPX(f'+{order}',*pixels,dxs)
    
    dys = conf.DISPY(f'+{order}',*pixels,ts)

    wavs = conf.DISPL(f'+{order}',*pixels,ts)
    
    return (
        pixels[0] + dxs,
        pixels[1] + dys,
        wavs,
    )

def computeTraceWLC(
    pixels,
    filter_="F322W2",
    module="A",
    simYDisp=False,
    order=1
):
    
    # Locate config File for the filter, module and grism direction 
    confFile = f"/STEM/scratch.san/zx446701/GRISM_NIRCAM/V3/NIRCAM_{filter_}_mod{module}_C.conf"
    # Class to read and hold GRISM configuration info
    conf = grismconf.Config(confFile)
    # Found from GRISMCONF README file - see link above


    # dt = 1/(conf.DISPX(f'+{order}', *pixels,0)-conf.DISPX(f'+{order}', *pixels,1)) / fac
    # t = np.arange(0,1,dt)

    dys0 = conf.DISPY(f'+{order}',*pixels, 0)

    dys1 = conf.DISPY(f'+{order}', *pixels, 1)
    
    # np.floor rounds to lower value
    # np.ceil rounds to higher value
    # Want this so full trace is picked up in all pixels
    
    dys = np.arange(np.floor(dys0),np.ceil(dys1)+1).astype(int)

    ts = conf.INVDISPY(f'+{order}',*pixels,dys)
    
    dxs = conf.DISPX(f'+{order}',*pixels,ts)

    wavs = conf.DISPL(f'+{order}',*pixels,ts)

    
    return (
        pixels[0] + dxs,
        pixels[1] + dys,
        wavs,
    )

def computeTraceWL(
    pixels,
    filter_="F322W2",
    module="A",
    direction="R",
    simYDisp=False,
    order=1
):
    if direction == "R":
        return computeTraceWLR(pixels=pixels,filter_=filter_,module=module,simYDisp=simYDisp,order=order)
    else:
        return computeTraceWLC(pixels=pixels,filter_=filter_,module=module,simYDisp=simYDisp,order=order)
    
def computeTraceBoxR(
    # Pixels of source being traced (x0, y0)
    pixels,
    # Not sure this is needed
    fac=100,
    # Change filter depending on filter used for observation
    filter_="F322W2", 
    # Change module depending on module used for observation
    module="A",
    # Is this needed?
    simYDisp=False,
    # Box around expected trace 
    returnRect=True,
    # Height 50 pixels as PSF modelled 50x50 pixels
    cross_disp_size=50,
    # Set Order desired to be computed for
    order=1,
    # Need some guidance on what this is !
    **patchkws,
):
    confFile = f"/STEM/scratch.san/zx446701/GRISM_NIRCAM/V3/NIRCAM_{filter_}_mod{module}_R.conf"
    conf = grismconf.Config(confFile)
    # X and Y disp polynomials with 2 steps, the start [0] and end [1] of the trace
    dxs = conf.DISPX(f'+{order}', *pixels, np.array([0, 1]))
    # Keep in in case the JWST dispersion is curved and we need to trace the change in the curve
    dys = conf.DISPY(f'+{order}', *pixels, np.array([0, 1]))
    
    # Locating the centre of the trace in dispersion direction [0.5]
    centrePix_X = conf.DISPX(f'+{order}', *pixels, np.array([0.5]))
    
    # Locating Cross-dispersion centre of the trace
    centrePix_Y = conf.DISPY(f'+{order}', *pixels, np.array([0.5]))

    # Slight Rotation of tracebox due to slant in trace
    angle = np.rad2deg(np.arctan((dys[1]-dys[0])/(dxs[1]-dxs[0])))
    
    if returnRect:
#         mplplot.scatter(pixels[0]+centrePix[0], pixels[1],c='green')
        return mplpatches.Rectangle(
            # x0,y0 in bottom left of rectangle
            (pixels[0] + dxs[0], pixels[1] + dys[0] - (cross_disp_size // 2)),
            # width of rectangle 
            dxs[1] - dxs[0],
            # height of box (PSF width 50 pixels)
            cross_disp_size,
            # Slight Rotation of tracebox due to slant in trace
            angle=angle,
            **patchkws,
        )
    # Returns Central x and y of trace and dimensions of tracebox (height, width)
    return (pixels[0]+centrePix_X[0], pixels[1]+centrePix_Y[0]), (cross_disp_size, abs(dxs[1] - dxs[0]))

def computeTraceBoxC(
    # Pixels of source being traced (x0, y0)
    pixels,
    # Not sure this is needed
    fac=100,
    # Change filter depending on filter used for observation
    filter_="F322W2",
    # Change module depending on module used for observation
    module="A",
    # Is this needed?
    simYDisp=False,
    # Box around expected trace 
    returnRect=True,
    # Height 50 pixels as PSF modelled 50x50 pixels
    cross_disp_size=50,
    # Set Order desired to be computed for
    order=1,
    # Need some guidance on what this is !
    **patchkws,
):
    confFile = f"/STEM/scratch.san/zx446701/GRISM_NIRCAM/V3/NIRCAM_{filter_}_mod{module}_C.conf"
    conf = grismconf.Config(confFile)
    # X and Y disp polynomials with 2 steps, the start and end of the trace
    dxs = conf.DISPX(f'+{order}', *pixels, np.array([0, 1]))
    # Keep in in case the JWST dispersion is curved and we need to trace the change in the curve
    dys = conf.DISPY(f'+{order}', *pixels, np.array([0, 1]))
    
    # Locating the centre of the trace in dispersion direction
    centrePix_Y = conf.DISPY(f'+{order}', *pixels, np.array([0.5]))
    
    # Locating the centre of the trace in CROSS-dispersion direction
    centrePix_X = conf.DISPX(f'+{order}', *pixels, np.array([0.5]))
    
    # Slight Rotation of tracebox due to slant in trace
    angle = np.rad2deg(np.arctan((dxs[1]-dxs[0])/(dys[1]-dys[0])))
    
    if returnRect:
#         mplplot.scatter(pixels[0]+centrePix[0], pixels[1],c='green')
        return mplpatches.Rectangle(
            # x0,y0 in bottom left of rectangle
            (pixels[0] + dxs[0] - cross_disp_size // 2, pixels[1] + dys[0]),
            # width of rectangle 
            cross_disp_size,
            # height of box (PSF width 50 pixels)
            dys[1] - dys[0],
            # Slight Rotation of tracebox due to slant in trace
            angle=angle,
            **patchkws,
        )
    # source Y pos + central Y pix of trace length, same but for X, PSF width (50 pix), trace length (~1365)
    return (pixels[1]+centrePix_Y[0], pixels[0]+centrePix_X[0]), (cross_disp_size, abs(dys[1] - dys[0]))

def computeTraceBox(
    # Pixels of source being traced (x0, y0)
    pixels,
    # Not sure this is needed
    fac=100,
    # Change filter depending on filter used for observation
    filter_="F322W2",
    # Change module depending on module used for observation
    module="A",
    # Change Direction depending on disperser used for observation
    direction="R",
    # Is this needed?
    simYDisp=False,
    # Box around expected trace 
    returnRect=True,
    # Height 50 pixels as PSF modelled 50x50 pixels
    cross_disp_size=50,
    # Set Order desired to be computed for
    order=1,
    # Need some guidance on what this is !
    **patchkws,
):
    if direction == "R":
        return computeTraceBoxR(pixels=pixels,fac=fac,filter_=filter_,module=module,simYDisp=simYDisp,returnRect=returnRect,cross_disp_size=cross_disp_size,order=order,**patchkws)
    else:
        return computeTraceBoxC(pixels=pixels,fac=fac,filter_=filter_,module=module,simYDisp=simYDisp,returnRect=returnRect,cross_disp_size=cross_disp_size,order=order,**patchkws)
    
def compute1stOrderTraceBox(
    # Pixels of source being traced (x0, y0)
    pixels,
    # Not sure this is needed
    fac=100,
    # Change filter depending on filter used for observation
    filter_="F322W2",
    # Change module depending on module used for observation
    module="A",
    # Change Direction depending on disperser used for observation
    direction="R",
    # Is this needed?
    simYDisp=False,
    # Box around expected trace 
    returnRect=True,
    # Height 50 pixels as PSF modelled 50x50 pixels
    cross_disp_size=50,
    # Need some guidance on what this is !
    **patchkws,
):
    return computeTraceBox(pixels,fac,filter_,module,direction,simYDisp,returnRect,cross_disp_size,1,**patchkws)

def compute2ndOrderTraceBox(
    # Pixels of source being traced (x0, y0)
    pixels,
    # Not sure this is needed
    fac=100,
    # Change filter depending on filter used for observation
    filter_="F322W2",
    # Change module depending on module used for observation
    module="A",
    # Change Direction depending on disperser used for observation
    direction="R",
    # Is this needed?
    simYDisp=False,
    # Box around expected trace 
    returnRect=True,
    # Height 50 pixels as PSF modelled 50x50 pixels
    cross_disp_size=50,
    # Need some guidance on what this is !
    **patchkws,
):
    if filter_ == "F322W2":
        return computeTraceBox(pixels,fac,filter_,module,direction,simYDisp,returnRect,cross_disp_size,2,**patchkws)
    else:
        print("You have made an error. 2nd Order spectra only occur in F332W2 frames. Please comment out patch.")
        
# Get sources from all images loaded into notebook
directPixels = getSourcePixelsForImage(file_name, frameListDir)
DPdf = pd.DataFrame(directPixels)
sourceCoords = getSourceCoordsForImage(file_name, frameListDir)

disp_ends_pix_1st = []
disp_ends_pix_2nd = []

for directPixel in directPixels:
    
    confFile = f"/STEM/scratch.san/zx446701/GRISM_NIRCAM/V3/NIRCAM_{filter_}_mod{module}_{direction}.conf"
    conf = grismconf.Config(confFile)
    # X and Y disp polynomials with 2 steps, the start and end of the trace for 1st and 2nd order spectra
    # Because the grismconf.config() function knows which filter is used from the file name,
    # so the conf.DISPX and conf.DISPY functions work as needed
    if direction == "R":
        # Returns first and last pixel in disp axis of trace for 1st order trace in R direction
        # In our original function (traceWL) we floor and ceil the trace values to ensure all the 
        # pixels with signal in at either end were included so must be done here
        
        dxs0 = conf.DISPX(f'+1',*directPixel, 0)

        dxs1 = conf.DISPX(f'+1', *directPixel, 1)
        
        # We are now "floor"ing and "ceil"ing pixels we want to include in trace
        # at same position as our function, therefore there is no longer a mismatch
        ends1st0 = directPixel[0] + np.floor(dxs0)
        ends1st1 = directPixel[0] + np.ceil(dxs1)
        ends1st = [int(ends1st0),int(ends1st1)]

        disp_ends_pix_1st.append(ends1st)
        
        # Same but 2nd order
        
        dxs02nd = conf.DISPX(f'+2',*directPixel, 0)
        
        dxs12nd = conf.DISPX(f'+2',*directPixel, 1)
        
        ends2nd0 = directPixel[0] + np.floor(dxs02nd)
        ends2nd1 = directPixel[0] + np.ceil(dxs12nd)
        ends2nd = [int(ends2nd0),int(ends2nd1)]

        disp_ends_pix_2nd.append(ends2nd)
    
    elif direction == "C":
        dxs0 = conf.DISPY(f'+1',*directPixel, 0)

        dxs1 = conf.DISPY(f'+1', *directPixel, 1)
        
        # We are now "floor"ing and "ceil"ing pixels we want to include in trace
        # at same position as our function, therefore there is no longer a mismatch
        ends1st0 = directPixel[1] + np.floor(dxs0)
        ends1st1 = directPixel[1] + np.ceil(dxs1)
        ends1st = [int(ends1st0),int(ends1st1)]

        disp_ends_pix_1st.append(ends1st)
        
        # Same but 2nd order
        
        dxs02nd = conf.DISPY(f'+2',*directPixel, 0)
        
        dxs12nd = conf.DISPY(f'+2',*directPixel, 1)
        
        ends2nd0 = directPixel[1] + np.floor(dxs02nd)
        ends2nd1 = directPixel[1] + np.ceil(dxs12nd)
        ends2nd = [int(ends2nd0),int(ends2nd1)]

        disp_ends_pix_2nd.append(ends2nd)

    else:
        print("You have made an error. Direction must be R or C.")

# Turn the list into a dataframe
df_disp_ends_pix_1st = pd.DataFrame(disp_ends_pix_1st)

# Rename the columns for pixel trace start and endpoints
df_disp_ends_pix_1st = df_disp_ends_pix_1st.rename(columns = {0 : "low_WL_trace_end_pixel", 1 : "high_WL_trace_end_pixel"})

# The X and Y pixels coordinates of the sources
df_disp_ends_pix_1st["Direct_X_pixel"] = DPdf[0]
df_disp_ends_pix_1st["Direct_Y_pixel"] = DPdf[1]

# The Sky Coordinates of sources
df_disp_ends_pix_1st["RA"] = sourceCoords.ra
df_disp_ends_pix_1st["Dec"] = sourceCoords.dec

# Order of Traces
df_disp_ends_pix_1st["Order"] = 1

# Position of source within Fengwu file - CHECK THIS!!!
df_disp_ends_pix_1st["FengwuID"] = df_disp_ends_pix_1st.index

# Dataframe sorted by trace starting furthest to the left
df_disp_ends_pix_1st = df_disp_ends_pix_1st.sort_values(by="low_WL_trace_end_pixel")

# True or false to come next
if filter_ == "F322W2":
    df_disp_ends_pix_2nd = pd.DataFrame(disp_ends_pix_2nd)
    # Rename the columns for pixel trace start and endpoints
    df_disp_ends_pix_2nd = df_disp_ends_pix_2nd.rename(columns = {0 : "low_WL_trace_end_pixel", 1 : "high_WL_trace_end_pixel"})
    # The X and Y pixels coordinates of the sources
    df_disp_ends_pix_2nd["Direct_X_pixel"] = DPdf[0]
    df_disp_ends_pix_2nd["Direct_Y_pixel"] = DPdf[1]
    # The Sky Coordinates of sources
    df_disp_ends_pix_2nd["RA"] = sourceCoords.ra
    df_disp_ends_pix_2nd["Dec"] = sourceCoords.dec
    # Order of Trace
    df_disp_ends_pix_2nd["Order"] = 2
    # Position of source with Fengwu file
    df_disp_ends_pix_2nd["FengwuID"] = df_disp_ends_pix_2nd.index

    # Dataframe sorted by trace starting furthest to the left
    df_disp_ends_pix_2nd = df_disp_ends_pix_2nd.sort_values(by="low_WL_trace_end_pixel")
    # df_disp_ends_pix_2nd


    source_data = pd.concat([df_disp_ends_pix_2nd,df_disp_ends_pix_1st])
    
else:
    source_data = df_disp_ends_pix_1st
    
def TrimOffsetFunc(conf,direction):
    if direction =="R":
        return np.amax([abs(conf._DISPY_data[f"+{o}"]) for o in range(1,3)])
    else:
        return np.amax([abs(conf._DISPX_data[f"+{o}"]) for o in range(1,3)])
    
# CONDITIONS CUT OFF SOURCES WITH TRACES OUTSIDE OF IMAGE BOUNDS
# Currently, the additional shifts are a full trace length off the end which is incorrect but works 
# Reference for out-of-field sources in grism images: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-wide-field-slitless-spectroscopy#NIRCamWideFieldSlitlessSpectroscopy-OutOfFieldOut-of-fieldsources

#
max_CD_trace_offset = TrimOffsetFunc(conf,direction)


if filter_ == "F322W2" and direction == "R" and module == "A":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset) & (source_data['high_WL_trace_end_pixel']>0) & (source_data['low_WL_trace_end_pixel']<2047) 
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel

elif filter_ == "F322W2" and direction == "R" and module == "B":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset) & (source_data['low_WL_trace_end_pixel']>0) & (source_data['high_WL_trace_end_pixel']<2047) 
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel
    
elif filter_ == "F322W2" and direction == "C":
    mask = (source_data.Direct_X_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_X_pixel < fits_file["SCI"].data.shape[0]-30-max_CD_trace_offset) & (source_data['high_WL_trace_end_pixel']>0) & (source_data['low_WL_trace_end_pixel']<2047) 
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_X_pixel

elif filter_ == "F444W" and direction == "R" and module == "A":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset) & (source_data['high_WL_trace_end_pixel']>0) & (source_data['low_WL_trace_end_pixel']<2047) 
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel

elif filter_ == "F444W" and direction == "R" and module == "B":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset) & (source_data['low_WL_trace_end_pixel']>0) & (source_data['high_WL_trace_end_pixel']<2047)
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel

elif filter_ == "F444W" and direction == "C":
    mask = (source_data.Direct_X_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_X_pixel < fits_file["SCI"].data.shape[0]-30-max_CD_trace_offset) & (source_data['high_WL_trace_end_pixel']>0) & (source_data['low_WL_trace_end_pixel']<2047) 
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_X_pixel
    
else:
    print("You have made an error.")
    
# Created array for results

# Mask selects the sources that are within a 1/2 PSF width from edge (25 pixels) as we cannot fit sources closer in than this 
# Might need to revisit this choice for potential recovery of source's data for combination with other grism dispersions

# USED FOR CREATING FLUX CALIBRATED PLOTS

trace_x_pixels = []
trace_y_pixels = []
trace_WL = []
for index, row in in_frame_source_data.iterrows():
    # print(row.Direct_X_pixel)

    trace_pixels = computeTraceWL((row.Direct_X_pixel,row.Direct_Y_pixel),
    filter_=filter_,
    module=module,
    direction=direction,
    simYDisp=False,
    order=row.Order.astype(int),
    )
    # Need as to floor or ceiling it 
    # I have used np.floor as originally the 0th element of the array was carrying 2 values 
    # Thought of writing code to immediately define dispersion and cross-dispersion pixels prior to fit running
    
    if direction == "R":
        # trace_x_pixels.append(np.floor(trace_pixels[0]))#astype(int))
        trace_x_pixels.append(trace_pixels[0][(np.floor(trace_pixels[0])>=0) & (trace_pixels[0]<2048)])#.astype(int))
        trace_y_pixels.append(trace_pixels[1][(np.floor(trace_pixels[0])>=0) & (trace_pixels[0]<2048)])
        trace_WL.append(trace_pixels[2][(np.floor(trace_pixels[0])>=0) & (trace_pixels[0]<2048)])
    else:
        trace_x_pixels.append(trace_pixels[0][(np.floor(trace_pixels[1])>=0) & (trace_pixels[1]<2048)])
        trace_y_pixels.append(trace_pixels[1][(np.floor(trace_pixels[1])>=0) & (trace_pixels[1]<2048)])#.astype(int))
        trace_WL.append(trace_pixels[2][(np.floor(trace_pixels[1])>=0) & (trace_pixels[1]<2048)])

# Chi squared function to test parameters fit

# norms = initial parameters
# cutoutSlice = column we are fitting
# errorSlice = errors associated with the column pixels
# psfs = 
# means = mean position of psfs (Dispersion Direction Direct Pixel)

# TO ADD ADDITIONAL FITTED PARAMETERS, 1ST THING IN FUCTION HAS TO BE THE FITTED PARAMS SO IF YOU HAVE 2 DIFFERENT VALUES
# YOU NEED TO INPUT THEM TOGETHER AND THEN UNPACK THEN IN THE FUNCTION?

# def fitFunction(IPs, cutoutSlice, errorSlice, psfs, means,

'''
Interpolated PSF Algorithm

Discrete Data Need X's and Y's - X's = Range of Y's or Real values in cross-dispersion direction (PSF mean -24 to PSF mean +25), Y's = Intensity of 1D PSF for of given source within column

Make interpolation object using Discrete Data - spinterp.interp1d(np.arange(mean-24,mean+25),psf)

Interpolate over PSF mean -24 to PSF mean +25 - np.arange()
np.arange(psf_means_float.loc[trace_index] - 24, psf_means_float.loc[trace_index] + 25)[:49]

interploated PSFs where x values aren't integers


'''

def fitFunction(norms, cutoutSlice, errorSlice, psfs, means,
#                 offsets,
                plotme=False,
                returnModel=False):
    # norms = IPs[:int(len(IPs/2))]
    # offsets = IPs[int(len(IPs/2)):]
    # Error Mask to simulate likely errors when checking for what was causing incorrect fits
#     error_mask = np.ones_like(cutoutSlice)*1000
    selection = errorSlice > 0
    model = np.zeros_like(cutoutSlice)
    
#     for norm, psf, mean, offset in zip(norms, psfs, means, offsets):
    for index, (norm, psf, mean) in enumerate(zip(norms, psfs, means)):
        # May be better interpolated below
        
        # Model for a single column with potentially many traces passing through
        # Artificially making mean an integer - need a better approach.
        # WHAT SHOULD WE DO INSTEAD?

        try:
            # These conditions ensure that the sources less than half a PSF wdith from the edge are able to be fitting too 
            # i.e if the psf was to leak over the edge of the image, the data is still usable
            
            # If we wanted to add cross-disp shift, would need to ensure this interpolation occurs within function
#             psf_interp = spinterp.interp1d(np.arange(mean-24,mean+25)[:49],psf,bounds_error=False,fill_value='extrapolate') 
            
            # x_range = range(max(0, int(mean) - psf.size // 2), min(cutoutSlice.size, int(mean) + psf.size // 2)+1)
        
            model[max(0, int(mean) - psf.size // 2) : min(cutoutSlice.size, int(mean) + psf.size // 2)+1] += norm * psf#_interp(x_range)
        except ValueError as e:
            print(index, norm, psf, mean, e)
                
    
    if plotme:
        fig, ((ax1),(ax2)) = mplplot.subplots(figsize=(10,20),nrows = 2, ncols =1)
        ax1.step(cutoutSlice,np.arange(cutoutSlice.size), label="Data")
        ax1.plot(model,np.arange(model.size),label="Model")
        ax1.legend()
        ax2.plot(errorSlice,label="Errors")
    if returnModel:
        return model
    return np.nansum(
        ((model[selection] - cutoutSlice[selection]) / errorSlice[selection]) ** 2
    )

## FITTING CODE

# %%time
# Result Array Setup - 2D array 
# Shape required is no of sources in the column by the no of pixels in the column (2048)
# Rows: Start with shape of full source list though not every column will have every trace in
# Columns: Start with full column length so any source with trace pixels within any column can have fits inputted
result = np.zeros(shape=(in_frame_source_data.shape[0],fits_file["SCI"].data.shape[0]))
result_err = np.zeros(shape=(in_frame_source_data.shape[0],fits_file["ERR"].data.shape[0]))

result_opt_box = np.zeros(shape=(in_frame_source_data.shape[0],fits_file["SCI"].data.shape[0]))
result_err_opt_box = np.zeros(shape=(in_frame_source_data.shape[0],fits_file["ERR"].data.shape[0]))

# Start list for PSF's used for each trace giving the source's direct image position
psfs1DForTraces = {}
# psfs2DForTraces = []
 
# Start list of full initial paramaters for all 2048 column
all_columns_initial_params = []
all_columns_initial_params_opt_box = []

# COMMENT THIS
all_columns_psf_list = []
all_columns_psf_means = []

# Start list to hold full fit result outputs
fit_results = []
fit_results_opt_box = []

traces_for_columns = []

all_psf_data = []
all_column_im_data = []

# Optimal Weighted Box Method Results
OptWeightBoxRes = np.full(shape=(in_frame_source_data.shape[0],fits_file["SCI"].data.shape[0]),fill_value=1000.0)
OptWeightBoxResErr = np.full(shape=(in_frame_source_data.shape[0],fits_file["SCI"].data.shape[0]),fill_value=1000.0)
no_err_OptWeightBoxRes = np.full(shape=(in_frame_source_data.shape[0],fits_file["SCI"].data.shape[0]),fill_value=1000.0)
BoxMethod = np.full(shape=(in_frame_source_data.shape[0],fits_file["SCI"].data.shape[0]),fill_value=1000.0)

# FITS FORMATTING IS Y,X SO:
# .T SWAPS DATA TO X,Y


if direction == 'R':
    SCI = fits_file["SCI"].data.T
    ERR = fits_file["ERR"].data.T
elif direction == 'C':
    SCI = fits_file["SCI"].data
    ERR = fits_file["ERR"].data


for column_number, (column, err_col) in enumerate(zip(SCI, ERR)):
    
    # INTERPOLATING IMAGE DATA - NEEDED FOR MULTIPLYING BY PSF 
    # FOR OPT BOX METHOD
    interp_col = spinterp.interp1d(range(len(column)),column)
    interp_col_err = spinterp.interp1d(range(len(err_col)),err_col)
    
    # if column_number > 100:
    #     break
    """
    # Step 1: Work out which and how many traces there are in column to define params
    """
    # Want all traces with a start < 0
    # How many traces are with sensitive pixels overlap the column as a boolean list
    
    # Applying this mask restricts the fits to only traces within the column 
    # This results in no fit before a trace has started and after it has finished
    # This is advantagous to the fitting as it prevents parameters being fitted to noise 
    # where no signal is present. This can adversely effect the fitting of real data points 
    # as 

    def mask_conditions(column_number):
        
        # In this mask, the wavelengths follow go low to high WITH the pixel numbers
        mask1 = (in_frame_source_data.low_WL_trace_end_pixel <= column_number) & (np.round(in_frame_source_data.high_WL_trace_end_pixel) >= column_number)


        # In this mask, the wavelengths follow go low to high AGAINST the pixel numbers
        mask2 = (in_frame_source_data.high_WL_trace_end_pixel <= column_number) & (np.round(in_frame_source_data.low_WL_trace_end_pixel) >= column_number)
        # display(in_frame_source_data[["high_WL_trace_end_pixel","low_WL_trace_end_pixel"]])
        return {"F444W":
                     # modules
                         {"A": 
                          #directions
                              {"C": mask1, "R": mask1}, 
                          "B":  
                            #directions
                              {"C": mask1, "R": mask2}
                         },
                "F322W2":
                     # modules
                         {"A": 
                          #directions
                              {"C": mask1, "R": mask1}, 
                          "B":  
                            #directions
                              {"C": mask1, "R": mask2}
                         }
                    }

    mask = mask_conditions(column_number)[f"{filter_}"][f"{module}"][f"{direction}"]
    # Number of traces that overlap column - NOT USED BUT COULD BE USEFUL SOMEWHERE IN FUTURE?
    # n_params = mask.sum()
    traces_for_columns.append(mask)

    
    """
    # Step 2: Create required number of parameters for fitting function
    """
    
    col_index_matches = []
    col_CD_pixel = []

    if direction == "R":
        data = zip(trace_x_pixels,trace_y_pixels)
    else:
        data = zip(trace_y_pixels,trace_x_pixels)
        
    for i, (disp_pix, cross_d_pix) in enumerate(data):
        if column_number >= np.floor(np.min(disp_pix)) and column_number <= np.ceil(np.max(disp_pix)):
            col_CD_pixel.append(np.array(spinterp.interp1d(disp_pix,cross_d_pix,bounds_error=False,fill_value="extrapolate")(column_number)))
        else:
            col_CD_pixel.append(np.empty((0,)))
            
            # col_match = np.where(disp_pix==column_number)
            # col_index_matches.append(col_match[0])
            # CD_shift_pixel = cross_d_pix[col_index_matches[i]]
            # col_CD_pixel.append(CD_shift_pixel.astype(int))

#     else:
#         for i, (disp_pix, cross_d_pix) in enumerate(zip(trace_y_pixels,trace_x_pixels)):
#             if column_number >= np.floor(np.min(disp_pix)) and column_number <= np.ceil(np.max(disp_pix)):
#                 col_CD_pixel.append(np.array(spinterp.interp1d(disp_pix,cross_d_pix)(column_number)))
#             else:
#                 col_CD_pixel.append(np.empty((0,)))
            
#             # col_match = np.where(disp_pix==column_number)
#             # col_index_matches.append(col_match[0])
#             # CD_shift_pixel = cross_d_pix[col_index_matches[i]]
#             # col_CD_pixel.append(CD_shift_pixel.astype(int))

    # For a given column, a list of the psf means (C-D axis values) for all traces within that column (mask ensures this)
    # psf_means = pd.Series(col_CD_pixel)[mask.to_numpy()].astype(int)
    
    psf_means_float = pd.Series(col_CD_pixel)[mask.to_numpy()] #'''!!!!!!!!!!!!!'''
    
    all_columns_psf_means.append(psf_means_float)

    """
    # Step 1.5 Extract PSF for each Source
    """
    # Setting up grid for PSF to be plotted into around direct image pixels of source 
    # The PSF is only given 25 pixels in each direction (N,E,S,W) of direct image source pixels
    
    # Create 1D PSF list of all sources within the mask dependent on their position in detector
    psf_list = []
    interp_psf_list = []
    
    col_fluxes_box = []
    col_fluxes_box_err = []
    no_err_col_fluxes_box = []
    fluxes_box = []

    all_psf_data.append(dict())
    all_column_im_data.append(dict())
    # Working from an index which orders the traces by lowest wavelength trace source first
    # Use direct imaging position as this is where the light will have hit the CCD
    for trace_loc, (trace_index, (Direct_X_pixel, Direct_Y_pixel)) in enumerate(in_frame_source_data.loc[mask.to_numpy(), ["Direct_X_pixel", "Direct_Y_pixel"]].iterrows()):
        # print(trace_loc,trace_index)
        # If a PSF has already been extracted for a given source and its trace, the "try" will add this to the psf_list
        # This will only continue as long as there is a trace for a given source going through that column, cutting off 
        # when the trace finishes
        # The pre-saving of PSFs and cutting off the list once the trace has ended should save us lots of time (hopefully)

        # try:
        if trace_index in psfs1DForTraces:
            # psf_temp = psfs1DForTraces[trace_loc]
            psf_list.append(psfs1DForTraces[trace_index])

        # If trace psf has not already been created and added to psf list, it creates the psf.
        # except IndexError as e:
        else:
            # print(e)
            
            # Fits files use 4TRAN notation for grids (y,x) not C notation (x,y) hence 
            y,x = np.mgrid[
                int(Direct_Y_pixel - 24) : int(Direct_Y_pixel + 25),
                int(Direct_X_pixel - 24) : int(Direct_X_pixel + 25),
            ]
    #             y,x = (y,x) if direction == "R" else (x,y) if direction == "C" else (0,0)
            # Need to ask where .evaluate came from!! 
            # Seems to create 2d PSF within given parameters around central point with a given flux
            # Flux seems to be arbitrary 
            # 2D array of Y axis slices divided into X axis length list
            # Shape is (70,50) so 70 arrays with 50 flux values each!

            # grid from PSF grid from webbpsf file - "Import PSF model grid modules" cell
            psf2d = grid.evaluate(
                x=x, y=y, flux=1, x_0=int(Direct_X_pixel), y_0=int(Direct_Y_pixel)
            )

            # "axis=_" is the cross dispersion
            psf1d = psf2d.sum(axis=1) if direction == "R" else psf2d.sum(axis=0) if direction == "C" else np.zeros(psf2d.shape[0])

            psfs1DForTraces[trace_index] = psf1d
    #         psfs2DForTraces.append(psf2d)

    #         # print(psfs1DForTraces[trace_loc])

            psf_list.append(psf1d)
        all_psf_data[-1][trace_index] = psf_list[-1]
        # print(column_number,np.shape(psf_list))
        
        psf1d = psf_list[trace_loc]

        '''INTERPOLATED PSF PRE-COMPUTATION'''
        psf_interp = spinterp.interp1d(np.arange(psf_means_float.loc[trace_index]-24,psf_means_float.loc[trace_index]+25)[:49],psf1d,bounds_error=False,fill_value='extrapolate') 

        col_range = range(max(0, int(psf_means_float.loc[trace_index]) - psf1d.size // 2), min(column.size, int(psf_means_float.loc[trace_index]) + psf1d.size // 2)+1)
        
        interp_psf_list.append(psf_interp(col_range))
        # print("Precomp Done")
        
        """
        Optimally Weighted Box Method set up for inital params
        """
        
        
        cross_disp_values = np.arange(psf_means_float.loc[trace_index] - 24, psf_means_float.loc[trace_index] + 25)[:49]

        column_im_data = interp_col(cross_disp_values)
        # print(column_im_data.shape)
        # print(psf1d.shape)
        # column_im_data = column[psf_means[trace_index]-24 : psf_means[trace_index]+25]
#         # print(trace_index, len(psf_means), column_number, column_im_data)
        # column_im_err = err_col[psf_means[trace_index]-24 : psf_means[trace_index]+25]
        column_im_err = interp_col_err(cross_disp_values)

#         # Assumes monochromatic PSF - though probably not true (i.e wavelength dependent psfs)
        col_flux_box = np.sum(column_im_data*(psf1d/(column_im_err**2)))/np.sum((psf1d/column_im_err)**2)
        col_fluxes_box.append(col_flux_box)
        
        col_flux_box_err = np.sqrt(np.sum(column_im_data*psf1d)/np.sum((psf1d/column_im_err)**2))
        col_fluxes_box_err.append(col_flux_box_err)
        
        no_err_col_flux_box = np.mean(column_im_data*psf1d)/np.sum(psf1d**2)
        no_err_col_fluxes_box.append(no_err_col_flux_box)
        
        box_flux = np.mean(column_im_data)
        fluxes_box.append(box_flux)
    
        all_column_im_data[-1][trace_index] = column_im_data*psf1d
    
    OptWeightBoxRes[mask.to_numpy(), column_number] = col_fluxes_box
    OptWeightBoxResErr[mask.to_numpy(), column_number] = col_fluxes_box_err
    
    # no_err_OptWeightBoxRes[mask.to_numpy(), column_number] = no_err_col_fluxes_box
    BoxMethod[mask.to_numpy(), column_number] = fluxes_box
    
    all_columns_psf_list.append(psf_list)
    
        # Creates initial parameters to fit from for each pixel along a trace using the data at each pixel
    """ ADDING VARIATION TO THE INITIAL PARAMETERS RATHER THAN BEING SO CLOSE TO THE TRUE SOLUTION """
    mean_int = np.round(psf_means_float.to_numpy().astype(float)).astype(int)
    #mean_int)
    init_params = column[mean_int] + np.random.randn(column[mean_int].size) * (0.1*column[mean_int].mean())
    col_fluxes_box_scattered = col_fluxes_box + np.random.randn(len(col_fluxes_box)) * (0.1*np.mean(col_fluxes_box))
    # print(col_fluxes_box)
    # print(col_fluxes_box_scattered)
    # print(len(column[psf_means]))
    # print(column[psf_means])
    # print(column[psf_means] + (np.random.randn(column[psf_means].size) * (0.1*column[psf_means].mean())))
    # init_params = col_fluxes_box

    # Creating list of lists of each columns intitial parameters for each trace that the column contains
    # Row shape = 2048 by number of traces in column
    # all_columns_initial_params.append(init_params)
    all_columns_initial_params_opt_box.append(col_fluxes_box_scattered)

    
    """
    # Step 3: Optimise parameters of the function given the data - Returns optimised params (Flux of each trace in column), Mean (location of source) is fixed (FOR NOW!)
    # ACTUAL PSF METHOD TAKING PLACE! 
    # THIS IS USING THE OPTIMISED BOX METHOD FLUXES AS INITIAL PARAMETERS
    """
#     if column_number == 1002:
#         _ = fitFunction(col_fluxes_box, column, err_col, interp_psf_list, psf_means_float, plotme=True)

#     # norms, cutoutSlice, errorSlice, psfs, means
#     # Optimised parameters
    try:
        opt_result_opt_box = spopt.minimize(
                fitFunction,

                # This is the 'norm' within fitFunction
                # Why deepcopy???
                copy.deepcopy(col_fluxes_box_scattered),
                # copy.deepcopy(init_params),

                # Arguments - the constants 
                (column, err_col, interp_psf_list, psf_means_float),
                method='BFGS',

                # HAD TO CHANGE STEP SIZE FOR MODEL TO MOVE ENOUGH TO BEGIN FITTING PROPERLY
                # MUST HAVE BEEN TOO SMALL INITIALLY - use 0.1 for step size
                # gtol = the amount to which the fit needs to be close to the local minima to count as successful
                options = dict(eps = 0.1,gtol=10e-3),
                # bounds=[[0, None]] * range(len(col_fluxes_box))
            )
    except ValueError as e:
        print(e)
        _ = fitFunction(col_fluxes_box, column, err_col, interp_psf_list, psf_means_float, plotme=True)

    fit_results_opt_box.append(opt_result_opt_box)

    opt_params_opt_box = opt_result_opt_box.x

    opt_params_err_opt_box = np.diagonal(opt_result_opt_box.hess_inv)
    
    """
    Step 4: Store Flux parameter outputs - associate flux with trace and column it belongs to!!!
    """
    # SAVE SEPARATE FILE WITH OBJECT IDENTICATION PER TRACE
    
    # Optimised Box initial parameters Results array
    result_opt_box[mask.to_numpy(), column_number] = opt_params_opt_box

    result_err_opt_box[mask.to_numpy(), column_number] = opt_params_err_opt_box


    """
    Repeat Step 3 and 4 for Original initial paramters
    """
    # if column_number == 1000:
    #     _ = fitFunction(init_params, column, err_col, interp_psf_list, psf_means_float, plotme=True)

    # norms, cutoutSlice, errorSlice, psfs, means
    # Standard Initial parameters
    try:
        opt_result = spopt.minimize(
                fitFunction,

                # This is the 'norm' within fitFunction
                # Why deepcopy???
                copy.deepcopy(init_params),

                # Arguments - the constants 
                (column, err_col, interp_psf_list, psf_means_float),
                method='BFGS',

                # HAD TO CHANGE STEP SIZE FOR MODEL TO MOVE ENOUGH TO BEGIN FITTING PROPERLY
                # MUST HAVE BEEN TOO SMALL INITIALLY - use 0.1 for step size
                # gtol = the amount to which the fit needs to be close to the local minima to count as successful
                options = dict(eps = 0.1,gtol=10e-3),
            )
    except ValueError as e:
        print(e)
        _ = fitFunction(init_params, column, err_col, interp_psf_list, psf_means_float, plotme=True)

    fit_results.append(opt_result)

    opt_params = opt_result.x

    opt_params_err = np.diagonal(opt_result.hess_inv)

    # Step 4: Append optimised parameters to results array

    result[mask.to_numpy(), column_number] = opt_params

    result_err[mask.to_numpy(), column_number] = opt_params_err

#     # Step 5: Next Column please

# Trace X, Y and wavelength arrays for all sources in each frame
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Trace_Parameters", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_trace_x_pixels"),trace_x_pixels)

np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Trace_Parameters", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_trace_y_pixels"),trace_y_pixels)

np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Trace_Parameters", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_trace_WL"),trace_WL)

    
# Results from Standard pixel flux value initial parameters
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Standard_IP_Extraction_Results_BOTHDIRECTIONS", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_standIP_result_array"),result)
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Standard_IP_Extraction_Results_BOTHDIRECTIONS", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_standIP_result_error_array"),result_err)

# Results from Optimised Box Method Initial Parameters
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Optimal_Box_IP_Extraction_Results_BOTHDIRECTIONS", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_OptBoxIP_result_array"),result_opt_box)
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Optimal_Box_IP_Extraction_Results_BOTHDIRECTIONS", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_OptBoxIP_result_error_array"),result_err_opt_box)

# Results from Optimised Box Method 
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Optimal_Box_Extraction_Results_BOTHDIRECTIONS", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_OptBox_result_array"),OptWeightBoxRes)
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Optimal_Box_Extraction_Results_BOTHDIRECTIONS", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_OptBox_result_error_array"),OptWeightBoxResErr)

# Box Method
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Box_Extraction_Results_BOTHDIRECTIONS", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_Box_result_array"),BoxMethod)

# Source Confusion Maps
# Leave for now!

# Extracting source input data
# np.save(os.path.join(os.path.dirname(os.path.dirname(file_name))), "Extraction_Results", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_fitted_dataframe"),in_frame_source_data)
in_frame_source_data.to_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Standard_IP_Extraction_Results", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_fitted_dataframe.pkl"))
in_frame_source_data.to_pickle(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Optimal_Box_IP_Extraction_Results", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_fitted_dataframe.pkl"))
