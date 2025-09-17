from astropy.convolution import Gaussian2DKernel, convolve_fft
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Ellipse

def add_beam(ax, bmaj_pix, bmin_pix, bpa_deg, xy_offset=(10, 10), color='white', crosshair=True):
    """
    Add a beam ellipse to a matplotlib Axes.
    
    Parameters:
    - ax: matplotlib Axes
    - bmaj_pix: major axis (pixels)
    - bmin_pix: minor axis (pixels)
    - bpa_deg: position angle in degrees (counter-clockwise from x-axis)
    - xy_offset: offset from bottom-left corner (pixels)
    - color: color of the beam ellipse and crosshair
    - crosshair: whether to draw crosshairs along major/minor axes
    """

    # Get limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Coordinates of bottom left, adjusted by offset
    x0 = xlim[0] + xy_offset[0]
    y0 = ylim[0] + xy_offset[1]

    # Beam ellipse
    beam = Ellipse((x0, y0),
                   width=bmin_pix,
                   height=bmaj_pix,
                   angle=bpa_deg,
                   edgecolor=color,
                   facecolor='none',
                   linewidth=1)
    ax.add_patch(beam)

    # Optional crosshairs
    if crosshair:
        # Convert angle to radians
        theta = np.deg2rad(90+bpa_deg)
        dx_major = 0.5 * bmaj_pix * np.sin(theta)
        dy_major = 0.5 * bmaj_pix * np.cos(theta)
        dx_minor = 0.5 * bmin_pix * np.cos(theta)
        dy_minor = -0.5 * bmin_pix * np.sin(theta)

        # Major axis line
        ax.plot([x0 - dx_major, x0 + dx_major], [y0 - dy_major, y0 + dy_major],
                color=color, linewidth=1, alpha = 0.5)
        # Minor axis line
        ax.plot([x0 - dx_minor, x0 + dx_minor], [y0 - dy_minor, y0 + dy_minor],
                color=color, linewidth=1, alpha=0.5)
        


def create_circular_aperture_mask(cube, R_e, beam_width_px):
    """
    Creates a circular mask, and expands it if beam causes significant flux to leak outside.

    Parameters:
    - cube: 3D numpy array (n_channels, nx, ny)
    - offset: max offset of source from center (in pixels)
    - init_grid_size: defines R_e
    - k: multiplier for how many R_e to include beyond offset
    - beam_fwhm: beam FWHM in same units as pixel_scale
    - pixel_scale: angular size per pixel (e.g., arcsec/px)
    - flux_threshold: fraction of total flux allowed to lie outside the mask

    Returns:
    - mask: 2D boolean array of shape (nx, ny)
    """

    D_e = 2*R_e

    beam_px = beam_width_px

    if D_e >= beam_px:
        D_aper = 2*D_e
        #print(f'D_e {D_e} is larger than beam size {beam_px}')
    elif beam_px>D_e:
        D_aper = 2*beam_px
        #print(f'Beam_px {beam_px} is larger than D_e {D_e}')


    n_channels, nx, ny = cube.shape
    x_center, y_center = nx // 2, ny // 2

    Y, X = np.ogrid[:nx, :ny]
    dist_from_center = np.sqrt((X - x_center)**2 + (Y - y_center)**2)

    '''# Base aperture radius
    aperture_radius = offset + k * R_e'''

    mask = dist_from_center <= D_aper/2


    mask_3d = np.asarray([mask for i in range(n_channels)])
    return mask_3d,  D_aper/2


def convolve_beam(spectral_cube, beam_width_px):
    convolved_spectral_cube_px = np.zeros_like(spectral_cube)
    
    bmaj_avg_px = beam_width_px
    bmin_avg_px = beam_width_px

    x_stddev = bmaj_avg_px / (2 * np.sqrt(2 * np.log(2)))
    y_stddev = bmin_avg_px / (2 * np.sqrt(2 * np.log(2)))


    # (BPA should be counterclockwise from north)
    # Finding the angle with respect to positive X axis
    bpa = 0
    #converting to radians

    theta = np.deg2rad(bpa)

    beam = Gaussian2DKernel(x_stddev, y_stddev, theta)
    beam.normalize()
    beam_area = (np.pi * bmaj_avg_px * bmin_avg_px)/(4 * np.log(2))


    # Convolve each slice (2D image) of the spectral cube with the PSF
    for i in range(spectral_cube.shape[0]):  # Iterate over the spectral dimension (N)
        
        # Convolve each 2D slice with the PSF

        convolved_spectral_cube_px[i, :, :] = convolve_fft(spectral_cube[i, :, :], beam)

    convolved_spectral_cube_beam = (convolved_spectral_cube_px * beam_area)

    return convolved_spectral_cube_beam


def apply_and_convolve_noise(spectral_cube, peak_snr, beam_width_px):
    # Step 1: Estimate peak flux
    peak_flux = np.max(spectral_cube)
    target_noise_rms = peak_flux / peak_snr

    # Step 2: Generate uncorrelated Gaussian noise
    white_noise = np.random.normal(0,1.0, spectral_cube.shape)


    # Step 3: Convolve with beam
    convolved_noise = convolve_beam(white_noise, beam_width_px)
    
    # Step 4: Measure RMS in background region only
    current_rms = np.std(convolved_noise)

    # Step 5: Scale to match target_noise_rms
    scaled_noise = convolved_noise * (target_noise_rms / current_rms)

    #print(np.std(white_noise), np.std(scaled_noise), peak_flux/np.std(scaled_noise))

    # Step 6: Add to signal
    noisy_cube = spectral_cube + scaled_noise


    return noisy_cube




def apply_noise(spectral_cube, peak_snr):
    signal_max = np.max(spectral_cube)
    mock_noise = signal_max / peak_snr

    noise_cube = np.array([
        np.random.normal(0, mock_noise, (spectral_cube.shape[1], spectral_cube.shape[2]))
        for _ in range(spectral_cube.shape[0])
    ])

    noisy_spectral_cube = spectral_cube + noise_cube
    return noisy_spectral_cube



def compute_flux_residual_rmse_stats(noisy_cube, ground_truth, mask):


    flux_conservation = np.abs((np.sum(mask*noisy_cube) - np.sum(mask*ground_truth))/np.sum(mask*ground_truth))*100
    residual = noisy_cube - ground_truth
    residual_std = np.std(residual) if residual.size > 0 else 0.0
    rmse_masked =  np.sqrt(mean_squared_error((mask*ground_truth).ravel(),( mask*noisy_cube).ravel())) if noisy_cube.size > 0 else 0.0

    return {
        "flux_conservation": flux_conservation,
        "residual_std": residual_std,
        "rmse_masked": rmse_masked
    }

