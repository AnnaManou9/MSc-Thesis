import constants as c
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from astropy.wcs import WCS
from reproject import reproject_interp
from spectral_cube import SpectralCube
import astropy.units as u
import pywt
import warnings
warnings.filterwarnings('ignore')

## Convert ghz to hz 
def ghz_to_hz(ghz):
    return ghz * 1e9

def einstein(logAij):
    return 10**logAij

## Convert Jy/beam to Kelvin
def jytokelvin(file, i, j):
    intensity=file[0].data[0,:,i,j]
    header=file[0].header
    xaxis=(np.arange(header['NAXIS1'])-(header['CRPIX1']-1))*header['CDELT1']*60.0**2
    yaxis=(np.arange(header['NAXIS2'])-(header['CRPIX2']-1))*header['CDELT2']*60.0**2
    freq=(np.arange(header['NAXIS3'])-(header['CRPIX3']-1))*header['CDELT3']/1000.0+header['CRVAL3']/1000.0
    theta_maj = header['BMAJ']*3600 #Major axis converted to arcsec
    theta_min = header['BMIN']*3600 #Minor axis converted too arcsec
    temp_data = (1.222e6*(intensity))/(((freq/1.e6)**2)*(theta_maj*theta_min))
    plt.figure(figsize=(15,7))
    plt.step(freq/1.e6, temp_data, c='black', label='Observed Spectrum')
    plt.axhline(y=0,linestyle='--', c='orange')
    plt.xlabel('Freq [GHz]')
    plt.ylabel('$T_B [K]$')
    plt.legend()
    plt.show()
    
## Calculate column density
def colden_cm(v,w,alpha):
    return (8*np.pi*c.k*v**2*w)/(c.h*c.c_cm**3*alpha)

def colden_km(v,w,alpha):
    return (8*np.pi*c.k*v**2*w)/(c.h*c.c_km**3*alpha)

## Plot spectrum 
def plot_spectrum(file, i, j):
    intensity=file[0].data[0,:,i,j]
    header=file[0].header
    xaxis=(np.arange(header['NAXIS1'])-(header['CRPIX1']-1))*header['CDELT1']*60.0**2
    yaxis=(np.arange(header['NAXIS2'])-(header['CRPIX2']-1))*header['CDELT2']*60.0**2
    freq=(np.arange(header['NAXIS3'])-(header['CRPIX3']-1))*header['CDELT3']/1000.0+header['CRVAL3']/1000.0
    X2, Y2 = np.meshgrid(xaxis, yaxis)
    plt.figure(figsize=(15,7))
    plt.step(freq/1e6, intensity, c='black', label='Observed Spectrum')
    plt.axhline(y=0,linestyle='--', c='orange')
    plt.xlabel('Freq [GHz]')
    plt.ylabel('Intensity [Jy/beam]')
    
## File info 
def file_info(file):
    header=file[0].header
    xaxis=(np.arange(header['NAXIS1'])-(header['CRPIX1']-1))*header['CDELT1']*60.0**2
    yaxis=(np.arange(header['NAXIS2'])-(header['CRPIX2']-1))*header['CDELT2']*60.0**2
    freq=(np.arange(header['NAXIS3'])-(header['CRPIX3']-1))*header['CDELT3']/1000.0+header['CRVAL3']/1000.0
    
## Selection of the peak and plot 
def int_region(file, i, j, fmin, fmax):
    intensity=file[0].data[0,:,i,j]
    header=file[0].header
    xaxis=(np.arange(header['NAXIS1'])-(header['CRPIX1']-1))*header['CDELT1']*60.0**2
    yaxis=(np.arange(header['NAXIS2'])-(header['CRPIX2']-1))*header['CDELT2']*60.0**2
    freq=(np.arange(header['NAXIS3'])-(header['CRPIX3']-1))*header['CDELT3']/1000.0+header['CRVAL3']/1000.0
    theta_maj = header['BMAJ']*3600 #Major axis converted to arcsec
    theta_min = header['BMIN']*3600 #Minor axis converted too arcsec
    temp_data = (1.222e6*(intensity))/(((freq/1.e6)**2)*(theta_maj*theta_min))    
    sel=((freq/1.0e6 > fmin)*(freq/1.0e6 < fmax))
    plt.figure(figsize=(15,7))
    plt.step(freq/1.e6, temp_data, c='black', label='Observed Spectrum')
    plt.step(freq[sel]/1.0e6, temp_data[sel], c='green', label='Observed Spectrum')
    plt.axhline(y=0,linestyle='--', c='orange')
    plt.xlabel('Freq [GHz]')
    plt.ylabel('$T_B [K]$')
    plt.show()

## Integration of the peak
def peak_integration(file, i, j, fmin, fmax):
    intensity=file[0].data[0,:,i,j]
    header=file[0].header
    xaxis=(np.arange(header['NAXIS1'])-(header['CRPIX1']-1))*header['CDELT1']*60.0**2
    yaxis=(np.arange(header['NAXIS2'])-(header['CRPIX2']-1))*header['CDELT2']*60.0**2
    freq=(np.arange(header['NAXIS3'])-(header['CRPIX3']-1))*header['CDELT3']/1000.0+header['CRVAL3']/1000.0
    theta_maj = header['BMAJ']*3600 #Major axis converted to arcsec
    theta_min = header['BMIN']*3600 #Minor axis converted too arcsec
    temp_data = (1.222e6*(intensity))/(((freq/1.e6)**2)*(theta_maj*theta_min))    
    sel=((freq/1.0e6 > fmin)*(freq/1.0e6 < fmax))    
    integrated_peak = np.sum(temp_data[sel])
    return integrated_peak
    
## Calculate the column density
def column_density(hdulist, excel, i, j, fmin, fmax, sheet_name, column, value):
    intensity=hdulist[0].data[0,:,i,j]
    header=hdulist[0].header
    xaxis=(np.arange(header['NAXIS1'])-(header['CRPIX1']-1))*header['CDELT1']*60.0**2
    yaxis=(np.arange(header['NAXIS2'])-(header['CRPIX2']-1))*header['CDELT2']*60.0**2
    freq=(np.arange(header['NAXIS3'])-(header['CRPIX3']-1))*header['CDELT3']/1000.0+header['CRVAL3']/1000.0
    theta_maj = header['BMAJ']*3600 #Major axis converted to arcsec
    theta_min = header['BMIN']*3600 #Minor axis converted too arcsec
    temp_data = (1.222e6*(intensity))/(((freq/1.e6)**2)*(theta_maj*theta_min))    
    sel=((freq/1.0e6 > fmin)*(freq/1.0e6 < fmax))
    integrated_peak = np.sum(temp_data[sel])
    file = pd.read_excel(excel, sheet_name=sheet_name)
    columns = file.columns
    logAij = file.loc[file[column] == value, 'logAij'].values[0]
    rest_freq = file.loc[file[column] == value, 'Frequency (GHz)'].values[0]
    Aij = einstein(logAij)
    freq_hz = ghz_to_hz(rest_freq)
    final_dens = colden_cm(freq_hz, integrated_peak, Aij)
    print('The final calculated column density is', final_dens, 'cm^-2')
    
## Moment maps 
def moment_maps(hdulist, freq, fmin, fmax, lower, upper, moment):
    # header=hdulist[0].header
    cube = SpectralCube.read(hdulist)
    slab = cube.spectral_slab(fmin * u.GHz, fmax * u.GHz)
    rest_freq = freq * u.GHz
    cube_with_velocity = slab.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=rest_freq)
    lower_bound = lower * u.Jy / u.beam 
    upper_bound = upper * u.Jy / u.beam  
    mask_bounds = (cube_with_velocity >= lower_bound) & (cube_with_velocity <= upper_bound)
    cube_masked = cube_with_velocity.with_mask(mask_bounds)
    moment_map = cube_masked.with_spectral_unit(u.km/u.s).moment(order=moment) 
    return moment_map
    
def linewidth_map(hdulist, rest_freq, lower, upper):
    header=hdulist[0].header
    cube = SpectralCube.read(hdulist)
    rest_freq = rest_freq * u.GHz
    cube_with_velocity = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=rest_freq)
    lower_bound = lower * u.Jy / u.beam 
    upper_bound = upper * u.Jy / u.beam  
    mask_bounds = (cube_with_velocity >= lower_bound) & (cube_with_velocity <= upper_bound)
    cube_masked = cube_with_velocity.with_mask(mask_bounds)
    linewidth = cube_masked.linewidth_fwhm()
    return linewidth
    
    
def column_density_map(hdulist, rest_freq, lower, upper):
    header=hdulist[0].header
    cube = SpectralCube.read(hdulist)
    rest_freq = rest_freq * u.GHz
    cube_with_velocity = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio', rest_value=rest_freq)
    lower_bound = lower * u.Jy / u.beam 
    upper_bound = upper * u.Jy / u.beam  
    mask_bounds = (cube_with_velocity >= lower_bound) & (cube_with_velocity <= upper_bound)
    cube_masked = cube_with_velocity.with_mask(mask_bounds)
    column_density = cube_masked.moment(order=0) * 1.82 * 10**18 / (u.cm * u.cm) * u.s / u.K / u.km 
    return column_density
 
## Plot beams in map 
def beam_ellipse(hdulist, x_center, y_center):
    header=hdulist[0].header
    beam_size_major_arcsec = header['BMAJ']*3600  # Major axis size in arcseconds
    beam_size_minor_arcsec = header['BMIN']*3600   # Minor axis size in arcseconds
    pixels_per_arcsec = header['BSCALE']  # Replace this with the pixel scale from your WCS
    beam_size_major_pixels = beam_size_major_arcsec * pixels_per_arcsec
    beam_size_minor_pixels = beam_size_minor_arcsec * pixels_per_arcsec
    beam_ellipse = patches.Ellipse((x_center, y_center),  # Center position (in axis fraction)
                                beam_size_major_pixels,  # Width of ellipse
                                beam_size_minor_pixels,  # Height of ellipse
                                edgecolor='black', facecolor='none', linewidth=1, angle=0)  # Angle can be adjusted
    return beam_ellipse

def major_axis(hdulist, x_center):
    header=hdulist[0].header
    beam_size_major_arcsec = header['BMAJ']*3600  # Major axis size in arcseconds
    pixels_per_arcsec = header['BSCALE']  # Replace this with the pixel scale from your WCS
    beam_size_major_pixels = beam_size_major_arcsec * pixels_per_arcsec
    return x_center - beam_size_major_pixels / 2, x_center + beam_size_major_pixels / 2

def minor_axis(hdulist, y_center):
    header=hdulist[0].header
    beam_size_minor_arcsec = header['BMIN']*3600   # Minor axis size in arcseconds
    pixels_per_arcsec = header['BSCALE']  # Replace this with the pixel scale from your WCS
    beam_size_minor_pixels = beam_size_minor_arcsec * pixels_per_arcsec
    return y_center - beam_size_minor_pixels / 2, y_center + beam_size_minor_pixels / 2

def denoised_spectrum(hdulist, i, j):
    data=hdulist[0].data
    intensity=data[0,:,i,j]
    coeff = pywt.wavedec(intensity, 'db1', level=2) #*wavelet decomposition of the signal with db1 wavelet and X levels of decomposition
    sigma = np.median(np.abs(coeff[-1])) / 0.6745 #*estimate the noise level using the MAD estimator (sigma = 1.4826*MAD) 
    uthresh = sigma * np.sqrt(2 * np.log(len(intensity))) #*universal threshold for the signal coefficients 
    coeff = [pywt.threshold(c, value=uthresh, mode='hard') for c in coeff] #*thresholding of the coefficients 
    denoised_spectrum = pywt.waverec(coeff, 'db1') #*reconstruction of the signal from the thresholded coefficients
    return denoised_spectrum

def z(shift, freq):
    return shift/freq

def doppler(z):
    return z*2.99e8   

def v_lsr(shift, rest_freq):
    return (shift/rest_freq)*c.c_km

def rms(intensity):
    return np.sqrt(np.mean(intensity**2))

def sn(signal, noise):
    return signal/noise

def combine_7m_tp(hdulist_7m, hdulist_tp):
    for i in range(0,len(hdulist_7m[0].data[0,:,:,:])):
        all_data_7m = hdulist_7m[0].data[0,:,:,:]
        all_data_tp = hdulist_tp[0].data[0,:,:,:]
        header_7m=hdulist_7m[0].header
        header_tp=hdulist_tp[0].header
        wcs_7m = WCS(header_7m, naxis=2)
        wcs_tp = WCS(header_tp, naxis=2)
        data_7m_reprojected, footprint = reproject_interp((all_data_7m[i], wcs_7m), wcs_tp, shape_out=all_data_tp[i].shape)
        return data_7m_reprojected + all_data_tp

def partition_function(gu, E, T):
    return np.sum(gu*np.exp(-E/T))

def temperature_conversion(intensity, theta_maj, theta_min, freq_hz):
    return (1.222e6*(intensity))/(((freq_hz/1.e9)**2)*(theta_maj*theta_min))

def doppler_velocity_cm(freq, rest_freq):
    return ((freq - rest_freq)/rest_freq) * c.c_cm

def total_column_density(nu, q, gu, Eu, T):
    return (nu*q/gu)*(np.exp(Eu/T))

def upper_column_density(freq, w, alpha):
    rest_freq = freq * 1e9
    return (8*np.pi*c.k*rest_freq**2*w)/(c.h*c.c_cm**3*(alpha))

def h2_column_density(lamda, temperature, kappa, Sv, theta):
    return 2.02e20*(np.exp(1.439/((temperature/10)*lamda)) - 1)*(kappa/0.01)*Sv*(theta/10)**(-2)*(lamda)**3

def read_partition(file, sheet_name, molecule_name):
    # Load the Excel sheet into a DataFrame
    df = pd.read_excel(file, sheet_name=sheet_name)
    
    # Ensure the necessary columns exist
    required_columns = ['Q(2.725K)', 'Q(5K)', 'Q(9.375K)', 'Q(18.75K)', 'Q(37.5K)', 'Q(75K)', 'Q(150K)', 'Q(225K)', 'Q(300K)', 'Q(500K)', 'Q(1000K)', 'Q(2000K)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"One or more required columns are missing from the sheet '{sheet_name}'.")

    # Filter the DataFrame for the specified molecule and transition
    filtered_df = df[(df['Molecule'] == molecule_name)]
    
    # Check if the result is empty
    if filtered_df.empty:
        raise ValueError(f"No data found for Molecule '{molecule_name}' in sheet '{sheet_name}'.")

    # Extract the values of interest
    Q_2_725 = filtered_df['Q(2.725K)'].values[0]
    Q_5 = filtered_df['Q(5K)'].values[0]
    Q_9_375 = filtered_df['Q(9.375K)'].values[0]
    Q_18_75 = filtered_df['Q(18.75K)'].values[0]
    Q_37_5 = filtered_df['Q(37.5K)'].values[0]
    Q_75 = filtered_df['Q(75K)'].values[0]
    Q_150 = filtered_df['Q(150K)'].values[0]
    Q_225 = filtered_df['Q(225K)'].values[0]
    Q_300 = filtered_df['Q(300K)'].values[0]
    Q_500 = filtered_df['Q(500K)'].values[0]
    Q_1000 = filtered_df['Q(1000K)'].values[0]
    Q_2000 = filtered_df['Q(2000K)'].values[0]
    
    return {'Q(2.725K)': Q_2_725, 'Q(5K)': Q_5, 'Q(9.375K)': Q_9_375, 'Q(18.75K)': Q_18_75, 'Q(37.5K)': Q_37_5, 'Q(75K)': Q_75, 'Q(150K)': Q_150, 'Q(225K)': Q_225, 'Q(300K)': Q_300, 'Q(500K)': Q_500, 'Q(1000K)': Q_1000, 'Q(2000K)': Q_2000}


def read_excel(file, sheet_name, molecule_name, transition, n_custom_value):
    # Load the Excel sheet into a DataFrame
    df = pd.read_excel(file, sheet_name=sheet_name)
    
    # Ensure the necessary columns exist
    required_columns = ['Molecule', 'Resolved QNs', 'Aij', 'Nu [cm-2]', 'Frequency (GHz)', 'Q(75K)', 'Q(150K)', 'Q(225K)', 'Q(300K)', 'Filling factor 7m', 'Filling factor TP']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"One or more required columns are missing from the sheet '{sheet_name}'.")

    # Filter the DataFrame for the specified molecule and transition
    filtered_df = df[(df['Molecule'] == molecule_name) & (df['Resolved QNs'] == transition)]
    
    # Check if the result is empty
    if filtered_df.empty:
        raise ValueError(f"No data found for Molecule '{molecule_name}' with Transition '{transition}' in sheet '{sheet_name}'.")

    # Extract the values of interest
    Aij = filtered_df['Aij'].values[0]
    rest_freq = filtered_df['Frequency (GHz)'].values[0]
    gu = filtered_df['gu'].values[0]
    Eu = filtered_df['Eu (K)'].values[0]
    Q_75 = filtered_df['Q(75K)'].values[0]
    Q_150 = filtered_df['Q(150K)'].values[0]
    Q_225 = filtered_df['Q(225K)'].values[0]
    Q_300 = filtered_df['Q(300K)'].values[0]
    ff_7m = filtered_df['Filling factor 7m'].values[0]
    ff_tp = filtered_df['Filling factor TP'].values[0]
    ff_comb = filtered_df['Filling factor combined'].values[0]
    name = filtered_df['Molecule'].values[0]
    
    return {"frequency": rest_freq, 'Aul': Aij, 'gu': gu, 'Eu': Eu, 'N': n_custom_value, 'Q_75': Q_75, 'Q_150': Q_150, 'Q_225': Q_225, 'Q_300': Q_300, 'ff_7m': ff_7m, 'ff_tp': ff_tp, 'name':name, 'ff_comb': ff_comb}

def intensity_kelvin(Aul, N, gu, Eu, temperature, freq, Q, tau, filling):
    boltzmann_factor = np.exp(-Eu /  temperature)
    depth_factor = tau / (1 - np.exp(-tau))
    intensity_value = ((Aul * c.h * c.c_cm**3 * N / (8 * np.pi * c.k * freq**2 * Q * depth_factor)) *
                       (gu * filling * boltzmann_factor))
    return intensity_value

def gaussian_profile(amplitude, freq, rest_freq, sigma):
    return amplitude * np.exp(-((freq - rest_freq)**2 / (2*sigma**2)))

def intensity(Aul, N, gu, Eu, temperature, freq, Q, tau, filling, theta_maj, theta_min):
    boltzmann_factor = np.exp(-Eu / temperature)
    temp_factor = ((freq/1.e9)**2 * theta_maj * theta_min) / (1.222e6)
    depth_factor = tau / (1 - np.exp(-tau))
    Nu = (N/Q)*gu*boltzmann_factor
    intensity_value = ((Aul * c.h * c.c_cm**3) / (8 * np.pi * c.k * freq**2 * depth_factor)) * Nu *  filling * temp_factor
    return intensity_value
    
def tau(fwhm_hz, N, gu, Q, Eu, Aul, freq, temperature):
    Bul = (Aul * c.c_cm**3) / (8 * np.pi * c.h * freq**3)
    delta_u = ((fwhm_hz / freq) * c.c_cm ) 
    Nu = (N/Q)*(gu*np.exp(-Eu/temperature))
    tau = (c.h/delta_u)*Nu*Bul*(np.exp((c.h*freq)/(c.k*temperature))-1)
    return tau

def line_temperature(ratio, g_one, g_two, A_one, A_two, E_one, E_two):
    in_ln = ratio*((g_two*A_two)/(g_one*A_one))
    log = np.log(in_ln)
    energy = E_two - E_one
    temperature = energy / log
    return temperature

def read_excel_rotational(file, sheet_name, molecule_name, transition):
    # Load the Excel sheet into a DataFrame
    df = pd.read_excel(file, sheet_name=sheet_name)
    
    # Ensure the necessary columns exist
    required_columns = ['Molecule', 'Resolved QNs', 'Aij sum', 'Eu average', 'Freq average', 'gu sum', 'Q(75K)', 'Q(150K)', 'Q(225K)', 'Q(300K)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"One or more required columns are missing from the sheet '{sheet_name}'.")

    # Filter the DataFrame for the specified molecule and transition
    filtered_df = df[(df['Molecule'] == molecule_name) & (df['Resolved QNs'] == transition)]
    
    # Check if the result is empty
    if filtered_df.empty:
        raise ValueError(f"No data found for Molecule '{molecule_name}' with Transition '{transition}' in sheet '{sheet_name}'.")

    # Extract the values of interest
    Aij = filtered_df['Aij sum'].values[0]
    rest_freq = filtered_df['Freq average'].values[0]
    gu = filtered_df['gu sum'].values[0]
    Eu = filtered_df['Eu average'].values[0]
    Q_75 = filtered_df['Q(75K)'].values[0]
    Q_150 = filtered_df['Q(150K)'].values[0]
    Q_225 = filtered_df['Q(225K)'].values[0]
    Q_300 = filtered_df['Q(300K)'].values[0]
    name = filtered_df['Molecule'].values[0]
    
    return {"frequency": rest_freq, 'Aul': Aij, 'gu': gu, 'Eu': Eu, 'Q_75': Q_75, 'Q_150': Q_150, 'Q_225': Q_225, 'Q_300': Q_300, 'name':name}

def ln_rotational_diagram(freq_hz, int_intensity, Aij, gu):
    int_intensity_cms = int_intensity * (c.c_cm/freq_hz) 
    up = 8*np.pi*c.k*freq_hz**2*int_intensity_cms
    down = c.h*(c.c_cm**3)*Aij*gu
    return np.log(up/down)