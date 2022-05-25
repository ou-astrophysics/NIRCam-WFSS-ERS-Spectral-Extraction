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
import copy
import grismconf
# import spectres
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

# Define function to extract the source list of sources within an uncalibrated image
def getSourceListForImage(image, frameListDir):
    listPath = os.path.join(
        frameListDir,
        f"{'_'.join(os.path.basename(image).split('_')[:4])}_uncal_pointsources.list",
    )
    return pd.read_csv(listPath, delim_whitespace=True, comment="#", header=None)


# Define function to extract the source sky coordinates of sources within an uncalibrated image
def getSourceCoordsForImage(image, frameListDir):
    sourceList = getSourceListForImage(image, frameListDir) # Use previous function within this one
    coords = apcoords.SkyCoord(
        *sourceList.loc[:, [3, 4]].to_numpy().T, frame=apcoords.ICRS, unit=apunits.deg
    )
    return coords


# Define function to extract the source pixel coordinates of sources within an uncalibrated image
def getSourcePixelsForImage(image, frameListDir):
    sourceList = getSourceListForImage(image, frameListDir)
    pixels = sourceList.loc[:, [5, 6]].to_numpy()
    # pixels are the
    return pixels

### This function is not used

# Just calculates the length and width of a trace from a given x,y pixel position 

def computeTrace(pixels, fac=100, filter_="F322W2", module="A", direction="R", simYDisp=False, order=1):
    # Locate config File for the module and grism direction 
    confFile = f"/STEM/scratch.san/zx446701/GRISM_NIRCAM/V3/NIRCAM_{filter_}_mod{module}_{direction}.conf"
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

### Function returning the pixels and wavelengths of those pixels of a source dispersed in the R direction

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

### Function returning the pixels and wavelengths of those pixels of a source dispersed in the C direction

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

### Function to choose between R or C wavelengths required

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
    
### Function to calculate the trace box of a source dispersed in the R direction

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

### Function to calculate the trace box of a source dispersed in the C direction

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
    return (pixels[1]+centrePix_Y[0], pixels[0]+centrePix_X[0]), (cross_disp_size, abs(dys[1] - dys[0]))

### Function to choose between R or C trace box required

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
    
### Function returning Trace Box of 1st Order dispersions 

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

### Function returning Trace Box of 2nd Order dispersions 

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

        
# Set up required variables for code
# Retrieve each sources direct pixel and sky coordinates and create a Pandas dataframe of the direct pixel coords for manipulation later.

# Get sources from all images loaded into notebook
directPixels = getSourcePixelsForImage(file_name, frameListDir)
DPdf = pd.DataFrame(directPixels)
sourceCoords = getSourceCoordsForImage(file_name, frameListDir)

## Create dataframe of source information
# - First and last Dispersion axis pixels for each source
#     - In 1st and 2nd order for F322W2
# - Direct Image Pixel coordinates
# - RA and Dec
# - Fengwu ID from original source catalogue

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
    

### Trim Offset Function

# Calculates the maximum cross-dispersion offset for a trace in either the first or 2nd order to allow for tarces of both orders to be included within the bounds of the in-frame sources dataframe.

# Absolute values required to be taken as offsets can be minus numbers

def TrimOffsetFunc(conf,direction):
    if direction =="R":
        return np.amax([abs(conf._DISPY_data[f"+{o}"]) for o in range(1,3)])
    else:
        return np.amax([abs(conf._DISPX_data[f"+{o}"]) for o in range(1,3)])

## Trim Dataframe to sources only within image
# - See https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-wide-field-slitless-spectroscopy#NIRCamWideFieldSlitlessSpectroscopy-OutOfFieldOut-of-fieldsources for regions where source may be off field but within GRISM data

# CONDITIONS CUT OFF SOURCES WITH TRACES OUTSIDE OF IMAGE BOUNDS
# Currently, the additional shifts are a full trace length off the end which is incorrect but works 
# Reference for out-of-field sources in grism images: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-observing-modes/nircam-wide-field-slitless-spectroscopy#NIRCamWideFieldSlitlessSpectroscopy-OutOfFieldOut-of-fieldsources

#
max_CD_trace_offset = TrimOffsetFunc(conf,direction)

if filter_ == "F322W2" and direction == "R" and module == "A":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset)  & (source_data.Direct_X_pixel >= 0) & (source_data.Direct_X_pixel <= 2047+1743)
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel

elif filter_ == "F322W2" and direction == "R" and module == "B":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset)  & (source_data.Direct_X_pixel >= -1743) & (source_data.Direct_X_pixel <= 2047)
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel

elif filter_ == "F444W" and direction == "R" and module == "A":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset)  & (source_data.Direct_X_pixel >= -1366) & (source_data.Direct_X_pixel <= 2047)
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel

elif filter_ == "F444W" and direction == "R" and module == "B":
    mask = (source_data.Direct_Y_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_Y_pixel < fits_file["SCI"].data.shape[1]-30-max_CD_trace_offset)  & (source_data.Direct_X_pixel >= 0) & (source_data.Direct_X_pixel <= 2047+1366)
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_Y_pixel

elif filter_ == "F322W2" and direction == "C":
    mask = (source_data.Direct_X_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_X_pixel < fits_file["SCI"].data.shape[0]-30-max_CD_trace_offset) & (source_data.Direct_Y_pixel >= 0) & (source_data.Direct_Y_pixel <= 2047+1743)
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_X_pixel

elif filter_ == "F444W" and direction == "C":
    mask = (source_data.Direct_X_pixel >= 40+max_CD_trace_offset) & (source_data.Direct_X_pixel < fits_file["SCI"].data.shape[0]-30-max_CD_trace_offset) & (source_data.Direct_Y_pixel >= -1366) & (source_data.Direct_Y_pixel <= 2047)
    in_frame_source_data = source_data[mask.to_numpy()]
    in_frame_source_data = in_frame_source_data.reset_index(drop=True)
    initial_trace_location = in_frame_source_data.Direct_X_pixel

else:
    print("You have made an error.")
    
## Creating x, y and wavelength arrays for each source from GRISMCONF functions

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
        trace_x_pixels.append(trace_pixels[0][(np.floor(trace_pixels[0])>=0) & (trace_pixels[0]<2048)].astype(int))
        trace_y_pixels.append(trace_pixels[1][(np.floor(trace_pixels[0])>=0) & (trace_pixels[0]<2048)])
        trace_WL.append(trace_pixels[2][(np.floor(trace_pixels[0])>=0) & (trace_pixels[0]<2048)])
    else:
        trace_x_pixels.append(trace_pixels[0][(np.floor(trace_pixels[1])>=0) & (trace_pixels[1]<2048)])
        trace_y_pixels.append(trace_pixels[1][(np.floor(trace_pixels[1])>=0) & (trace_pixels[1]<2048)].astype(int))
        trace_WL.append(trace_pixels[2][(np.floor(trace_pixels[1])>=0) & (trace_pixels[1]<2048)])

# Trace X, Y and wavelength arrays for all sources in each frame
np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Trace_Parameters", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_trace_x_pixels"),trace_x_pixels)

np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Trace_Parameters", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_trace_y_pixels"),trace_y_pixels)

np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_name))), "Trace_Parameters", f"{os.path.basename(file_name)[:-5]}_{filter_}_{module}_{direction}_Primary{dither}_Exp{exposure}_trace_WL"),trace_WL)