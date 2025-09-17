import numpy as np
import matplotlib.pyplot as plt
from pycs.misc.cosmostat_init import *
from pycs.misc.stats import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from astrodendro import Dendrogram

def pca_denoising(spectral_cube, mask=None, plot = True, verbose=True):

    """
    Denoises a 3D spectral cube using Principal Component Analysis (PCA).

    The function standardizes the data, applies PCA, and iteratively reconstructs 
    the cube using a subset of principal components. The reconstruction is evaluated 
    using a provided emission mask to determine the total flux in emission regions 
    and the residual noise in non-emission regions. The number of principal components 
    is automatically selected using a plateau criterion based on flux stabilization and 
    minimum noise.

    Parameters
    ----------
    spectral_cube : np.ndarray
        3D array of shape (channels, height, width) representing the noisy spectral cube.
    mask : np.ndarray
        Boolean or 0/1 array of the same shape as `spectral_cube` indicating emission regions.
        Must be provided to compute flux and RMSE for iterative PCA selection.
    plot : bool, optional
        If True, generates diagnostic plots of flux, cumulative variance, and reconstructed cube slices.
        Default is True.
    verbose : bool, optional
        If True, prints progress and intermediate results. Default is True.

    Returns
    -------
    best_model : np.ndarray
        The denoised spectral cube reconstructed using the selected optimal number of principal components.
    n_pc : int
        Number of principal components selected for reconstruction based on plateau criterion.
    flux_history : list of float
        History of total flux in emission regions for each number of principal components used.
    cumulative_explained_variance : np.ndarray
        Cumulative explained variance ratio of the principal components.
    
    Notes
    -----
    - Standardization is applied prior to PCA, and the cube is de-standardized after reconstruction.
    - The plateau criterion considers flux stabilization over consecutive principal components 
      (controlled by `epsilon_flux` and `p`) and selects the configuration with minimal residual noise.
    - If the plateau criterion is not reached, the number of PCs is chosen based on a cumulative explained variance threshold.
    - Diagnostic plots include:
        1. Total flux in emission regions vs number of PCs.
        2. Cumulative explained variance ratio vs number of PCs.
    """
    
    if mask is None:
        raise ValueError("Emission mask must be provided to compute RMSE.")

    final_pca_components = spectral_cube.shape[0]
    spectral_cube_copy = spectral_cube.copy()
    spectral_cube = spectral_cube.T
    # Step 1: Standardize the data (mean 0, variance 1) before applying PCA
    scaler = StandardScaler()
    noisy_centered_data_matrix = spectral_cube.reshape((spectral_cube.shape[0]*spectral_cube.shape[1]), spectral_cube.shape[2])
    noisy_centered_data_matrix_standardized = scaler.fit_transform(noisy_centered_data_matrix)

    # Step 2: Perform PCA on the standardized data
    pca = PCA(random_state=0)
    pca.fit(noisy_centered_data_matrix_standardized)
    components = pca.transform(noisy_centered_data_matrix_standardized)

    if verbose: print('1. Decomposing into principal components')

    # Step 3: Extract eigenvalues (explained variance)
    if verbose: print('2. Calculating the explained variance ratio')
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    init_pca_components = 1
    final_pca_components = spectral_cube.shape[2]

    if verbose: print(f'3. Calculating the total flux of emission regions of denoised cube with ground truth for n_components = 1,2,...{final_pca_components}')

    flux_history = []
    noise_history = []

    epsilon_flux = 1e-2  # Flux tolerance
    p = 3
    plateau_counter = 0
    best_model = None
    n_pc = None
    previous_flux = 1e-33
    min_noise = float('inf')

    
    for i in range(final_pca_components):
        # Reconstruct the data using the selected components
        components_new = components.copy()
        components_new[:, init_pca_components:] = 0
        denoised_data_2d = pca.inverse_transform(components_new)

        # Reshape back to original 3D shape
        reconstructed_data_cube_standardized = denoised_data_2d.reshape(spectral_cube.shape[0], spectral_cube.shape[1], spectral_cube.shape[2])

        # Step 4: De-standardize the data (reversing the scaling)
        reconstructed_data_cube = scaler.inverse_transform(reconstructed_data_cube_standardized.reshape(-1, reconstructed_data_cube_standardized.shape[2])).reshape(reconstructed_data_cube_standardized.shape)
        reconstructed_data_cube = reconstructed_data_cube.T

    
        flux = np.sum(mask * reconstructed_data_cube)
        noise = np.std((1 - mask) * reconstructed_data_cube)
        #noise = np.std(spectral_cube_copy - reconstructed_data_cube)

        flux_history.append(flux)
        noise_history.append(noise)

        if verbose: print(f'PCs {init_pca_components}: flux: {flux:.4e}, noise: {noise:.4e}')

        if abs(flux - previous_flux) / previous_flux <= epsilon_flux:
            plateau_counter += 1
        else:
            plateau_counter = 0

        if noise < min_noise and plateau_counter >= p:
            min_noise = noise
            best_model = reconstructed_data_cube
            n_pc = init_pca_components
            if verbose:
                print('\nflux: ',flux)
                print('noise: {}\n'.format(noise))


        previous_flux = flux
        init_pca_components += 1

    if n_pc is None:
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        threshold = 0.7  # or 0.95
        n_pc = np.argmax(cumulative_explained_variance >= threshold) + 1
        if verbose:
            print(f'\nPlateau not reached. Using {n_pc} PCs to explain {threshold*100:.1f}% of variance.')


        # Reconstruct the data using the selected components
        components[:, n_pc:] = 0
        denoised_data_2d = pca.inverse_transform(components)
        reconstructed_data_cube_standardized = denoised_data_2d.reshape(
            spectral_cube.shape[0], spectral_cube.shape[1], spectral_cube.shape[2])
        reconstructed_data_cube = scaler.inverse_transform(
            reconstructed_data_cube_standardized.reshape(-1, reconstructed_data_cube_standardized.shape[2])
        ).reshape(reconstructed_data_cube_standardized.shape).T
        best_model = reconstructed_data_cube


        
    if verbose: print(f'4. Identifying the number of principal components containing the signal: {n_pc} PCs')

    if plot:
        print('5. Statistical Plots')

        fig1 = plt.figure(figsize=(15, 5))
        fig1.add_subplot(121)
        x = np.linspace(1, final_pca_components, final_pca_components)
        plt.plot(x, flux_history, color='xkcd:blue', marker='.', label='Total flux')
        plt.xlabel('Number of principal components used for reconstruction')
        plt.ylabel('Flux')
        plt.yscale('log')
        plt.title('Comparison of denoised cube with ground truth\nfor different numbers of principal components\n', fontsize=15)
        plt.axvline(x=n_pc, color='xkcd:red', linestyle='--', label=f'Number of PCs = {n_pc}')
        plt.legend(frameon=True)

        fig1.add_subplot(122)
        plt.plot(x, cumulative_explained_variance*100, color='xkcd:blue', marker='.', label='Explained variance ratio for each principal component')
        threshold = cumulative_explained_variance[n_pc-1]*100
        plt.axhline(y=threshold, color='xkcd:red', linestyle='--', label=f'Threshold = {threshold:.3f}')
        #plt.yscale('log')
        plt.xlabel('Number of principal components used for reconstruction')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance of Principal Components\n')
        plt.legend(frameon=True)

        plt.show()

    # Reconstruct the data using the selected components
    components[:, n_pc:] = 0
    reconstructed_data_matrix = pca.inverse_transform(components)

    # Reshape back to original 3D shape
    reconstructed_data_cube = reconstructed_data_matrix.reshape(spectral_cube.shape[0], spectral_cube.shape[1], spectral_cube.shape[2]).T

    return best_model, n_pc, flux_history, cumulative_explained_variance
