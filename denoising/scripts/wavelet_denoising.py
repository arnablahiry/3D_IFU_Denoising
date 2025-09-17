

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pysparse
from pycs.misc.cosmostat_init import *
from pycs.misc.stats import *


class Wavelet2D1DTransform(object):
    """Wavelet decomposition of a 3D data cube."""
    
    # NOT CURRENTLY USED
    # TODO Compute values more accurately by excluding the borders of each sub-band cube
    NOISE_TAB = np.array([[0.9610, 0.9261, 0.9611, 0.9810, 0.9933],
                          [0.2368, 0.2282, 0.2369, 0.2417, 0.2449],
                          [0.1049, 0.1011, 0.1050, 0.1071, 0.1082],
                          [0.0527, 0.0507, 0.0528, 0.0539, 0.0543],
                          [0.0283, 0.0272, 0.0283, 0.0288, 0.0295]])

    def __init__(self, transform_type=2):
        """Wrapper for pysparse's 2D-1D wavelet transform.

        Parameters
        ----------
        transform_type : int
            Type of wavelet transform to perform with `decompose`. See pysap
            documentation for all options. Default is 2, which takes a starlet
            transform in the 2D domain (undecimated) and a 7/9 filter bank
            in the 1D domain (decimated).

        """
        self.transform_type = transform_type

    def decompose(self, cube, num_scales_2d, num_scales_1d):
        """Forward 2D1D wavelet transform.

        Parameters
        ----------
        cube : 3D array
            Data to transform.
        num_scales_2d : int
            Number of wavelet scales in the 2D domain (i.e. the last two axes).
        num_scales_1d : int
            Number of wavelet scales in the 1D domain (i.e. first axis).

        Returns
        -------
        inds : nested list of tuples
            Pairs of index values arranged such that inds[i][j] gives the
            (start, end) indices for the band (2d_scale, 1d_scale) = (i, j).
            See `_extract_index_ranges` docstring.
        coeffs : 1D array
            A flattened array object containing all the wavelet coefficients in
            increasing order of scale. Also included are the sizes of the
            transformed cube bands along each axis as three ints just before
            the coefficients themselves. Unpacking this array to access
            specific coefficients for a given (i, j) band requires `inds`.

        """
        # Compute the transform
        self._mr2d1d = pysparse.MR2D1D(type_of_transform=self.transform_type,
                                       normalize=False,
                                       verbose=False,
                                       NbrScale2d=num_scales_2d,
                                       Nbr_Plan=num_scales_1d)
        coeffs = self._mr2d1d.transform(cube)

        # Determine the starting/ending index values of the 1d array that
        # correspond to the wavelet coefficients of band (scale_2d, scale_1d)
        inds, shapes = self._extract_metadata(coeffs)

        return inds, shapes, coeffs

    def reconstruct(self, coeffs):
        """Inverse 2D1D wavelet transform.

        Parameters
        ----------
        coeffs : 1D array
            Wavelet coefficients plus index markers packaged as a 1D array,
            i.e. the output of decompose().

        Returns
        -------
        cube : 3D array
            Reconstructed 3D data cube.

        """
        #assert hasattr(self, '_mr2d1d'), "Need to call decompose first."

        reconstructed = self._mr2d1d.reconstruct(coeffs)
        
        return reconstructed

    def energy_per_scale(self, num_scales_2d, num_scales_1d):
        return self.NOISE_TAB[:num_scales_2d, :num_scales_1d]

    @property
    def num_precomputed(self):
        return self.NOISE_TAB.shape

    def _extract_metadata(self, coeffs):
        """Get metadata of transformed coefficients for all pairs of scales.

        Parameters
        ----------
        coeffs : 1D array
            Wavelet coefficients plus index markers packaged as a 1D array,
            i.e. the output of decompose().

        Returns
        -------
        inds : nested list of tuples
            Pairs of index values arranged such that inds[i][j] gives the
            (start, end) indices for the band (2d_scale, 1d_scale) = (i, j).
            The actual coefficients of the (i, j) band can therefore be accessed
            as coeffs[start:end].
        shapes : nested list of tuples
            Triplets of sub-band cube shapes arranged such that shapes[i][j]
            gives the (nx, ny, nz) shape of the band (2d_scale, 1d_scale) = (i, j).

        """
        n_scales_2d = int(coeffs[0])
        n_scales_1d = int(coeffs[1])

        inds = [[() for _ in range(n_scales_1d)] for _ in range(n_scales_2d)]
        shapes = [[() for _ in range(n_scales_1d)] for _ in range(n_scales_2d)]

        # Starting index
        start = end = 2

        # Traverse the flattened array to pull out ranges for each index pair
        for ii in range(n_scales_2d):
            for jj in range(n_scales_1d):
                # Starting index for this band
                start = end + 3
                # Extract band sizes
                nx, ny, nz = map(int, coeffs[start-3 : start])
                shapes[ii][jj] = (nx, ny, nz)
                # Total number of coefficients in this band
                ncoeff = nx * ny * nz
                # Ending index for this band
                end = start + ncoeff
                inds[ii][jj] = (start, end)

        return inds, shapes


# ## 3D Wavelet denoising
# 
# The Denoiser2D1D class below is modeled after Aymeric's StarletDenoiser that does 2D denoising on each frequency slice of the input data. The overall structure is kept so that it will be easier to combine the two denoising options (2D or 3D) later for the user's convenience.
# 
# NOTE : the denoise() method here accepts an optional noise cube realisation as input. The noise is transformed and its standard deviation is computed in each wavelet band to establish the noise level there. If not provided, the noise in each band is estimated from the data.

# In[3]:


class Denoiser2D1D(object):
    """Denoise a data cube using a 2D1D wavelet decomposition.

    The model is simply Y = X + N, where X is the noiseless signal and N is
    the noise.

    """
    def __init__(self, threshold_type='soft', verbose=True, plot=False):
        """Initialise the denoiser.

        Parameters
        ----------
        threshold_type : string
            Supported values are 'soft' or 'hard'. Default: 'soft'
        verbose : bool
            If True, prints some more info, useful for debugging. Default: True
            
        """
        self.mr2d1d = Wavelet2D1DTransform()
        self._threshold_type = threshold_type
        self._verbose = verbose
        self._plot = plot

    def __call__(self, *args, **kwargs):
        """Alias for self.denoise()"""
        return self.denoise(*args, **kwargs)

    def denoise(self, x, y, method='simple', threshold_level=3,
                threshold_increment_high_freq=2, num_scales_2d=None, 
                num_scales_1d=None, noise_cube=None, emission_mask = None, **kwargs_method):
        """Denoise a data cube according to the chosen method.

        Parameters
        ----------
        x : array_like (3D)
            Input data cube. The frequency axis is assumed to be first.
        y : array_like (3D)
            Signal (ground truth) data cube. The frequency axis is assumed to be first.
        emission_mask: array_like (3D)
            Mask with only the emission regions visible. The frequency axis is assumed to be first.
        method : string
            Denoising method, either 'simple' or 'iterative'. The iterative method
            should give better results but takes longer to compute.
            Default: 'simple'.
        threshold_level : int
            Threshold level, as a detection signicance, in noise units (generally 
            between 3 and 5 for '3-5 sigmas' detection).
            Default: 3
        threshold_increment_high_freq : int
            Increment of the above threshold_level for the highest frequencies (usually 
            associated with pure noise).
            Default: 2
        num_scales_2d : int
            Number of starlet decomposition scales for the 2D images. Maximal value is 
            int(np.log2(input_image_.shape[-1])). 
            Default: None (max. value).
        num_scales_1d : int
            Number of wavelet scales for the 1D axis. Maximal value is 
            int(np.log2(input_image_.shape[0])). 
            Default: None (max. value).
        noise_cube : array_like, same shape as input `x`
            An estimate of the noise (e.g. by simulation). If not provided, the noise level
            is estimated automatically in each wavelet sub-band.
            Default: None
        
        kwargs_method : dict
            [See docstring of each method]

        Returns
        -------
        array_like
            Denoised array
        
        """        

        # Set the number of 2D decomposition scales
        num_scales_2d_max = int(np.log2(x.shape[1]))
        if num_scales_2d is None or num_scales_2d < 2 or num_scales_2d > num_scales_2d_max:
            # choose the maximum allowed number of scales
            num_scales_2d = num_scales_2d_max
            if self._verbose is True:
                print(f"Number of 2D wavelet scales set to {num_scales_2d} "
                      "(maximum value allowed by input image)")
        
        # Set the number of 1D decomposition scales
        num_scales_1d_max = int(np.log2(x.shape[0]))
        if num_scales_1d is None or num_scales_1d < 2 or num_scales_1d > num_scales_1d_max:
            # choose the maximum allowed number of scales
            num_scales_1d = num_scales_1d_max
            if self._verbose is True:
                print(f"Number of 1D wavelet scales set to {num_scales_1d} "
                      "(maximum value allowed by input image)")
                
        # Check that the pre-computed noise scaling exists for the requested scales
        # if (num_scales_2d - 1 > self.mr2d1d.num_precomputed[0] or 
        #     num_scales_1d - 1 > self.mr2d1d.num_precomputed[1]):
        #     raise NotImplementedError(f"Pre-computed noise in wavelet space has been implemented"
        #                               f" for up to {self.mr2d1d.NOISE_TAB.shape} scales "
        #                               f"[({num_scales_2d}, {num_scales_1d}) required)]")
            
        # Check that the noise realisation has the same shape as the input

        if noise_cube is not None:
            assert x.shape == noise_cube.shape, "Invalid noise estimate shape"

        # Initialise settings for the denoiser
        self._data = x
        self._signal = y
        self._num_bands = self._data.shape[0]
        self._num_pixels = self._data.shape[1] * self._data.shape[2]
        self._num_scales_2d = num_scales_2d
        self._num_scales_1d = num_scales_1d
        self._threshold_level = float(threshold_level)
        self._thresh_increm = float(threshold_increment_high_freq)
        self._noise = noise_cube
        if emission_mask is None:
            emission_mask = np.ones_like(y)
        self._mask = emission_mask

        # Select and run the denoiser
        if method == 'simple':
            if self._verbose:   print('\n--- [ PERFORMING SIMPLE (ONE-STEP) DENOISING ] ---\n')
            result = self._denoise_simple()
        elif method == 'iterative':
            if self._verbose:   print('\n--- [ PERFORMING ITERATIVE DENOISING ] ---\n')
            if self._threshold_type == 'hard':
                result = self._denoise_iterative_hard(**kwargs_method)
            if self._threshold_type == 'soft':
                result = self._denoise_iterative_soft(**kwargs_method)
        else:
            raise ValueError(f"Denoising method '{method}' is not supported")
        
        return result



    def _generate_hard_threshold_mask(self, coeffs, thresh, noise_level):
        """Generate a mask for hard thresholding.
        
        Parameters
        ----------
        coeffs : array_like
            1D array of this sub-band.
        thresh : float
            Threshold value in noise units.
        noise_level : float
            Noise level at the scale represented by `array`.

        Returns
        -------
        array_like
            Binary mask for hard thresholding.
        """
        
        # Calculate the threshold values for this scale
        threshold = thresh * noise_level 
        
        # Generate the mask: 1 for values above threshold, 0 otherwise

        mask_coeff = np.ones_like(coeffs)
        
        mask_coeff[np.abs(coeffs) <= threshold] = 0
        
        
        return mask_coeff





    def _residual_signal_extraction_l0(self, model, mask_coeff, iteration):


        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, max_y, max_x = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        residual = self._data - model
                
        if self._plot:
            fig, axs = plt.subplots(2, 3, figsize=(16, 13), constrained_layout=True)

            im1 = axs[0,0].imshow(model[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,0].set_title('Previously Denoised (Iteration #{})'.format(iteration))

            im2 = axs[0,1].imshow((self._signal - model)[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,1].set_title('SIGNAL Residual')

            im3 = axs[0,2].imshow(self._signal[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,2].set_title('SIGNAL')

            axs[0,0].axis('off')
            axs[0,1].axis('off')
            axs[0,2].axis('off')

            cbar1 = fig.colorbar(im1, ax=axs[0, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar2 = fig.colorbar(im2, ax=axs[0, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar3 = fig.colorbar(im3, ax=axs[0, 2], orientation='horizontal', fraction=0.05, pad=0.02)

            cbar1.ax.tick_params()#labelsize=20)
            cbar2.ax.tick_params()#labelsize=20)
            cbar3.ax.tick_params()#labelsize=20)

            cbar1.set_label('Flux')
            cbar2.set_label('Flux')
            cbar3.set_label('Flux')


        if iteration==1:
            if self._verbose: print('(*) Decomposing residual into wavelet scales')

        inds, shapes, w_residual = self.mr2d1d.decompose(residual,
                                                            self._num_scales_2d,
                                                            self._num_scales_1d)

        # Apply the mask computed from the first iteration


        if iteration==1:
            if self._verbose: print('(*) Applying previously calculated mask on the residual coefficients, and\n considering the unmased coefficients as previously unnoticed signal coefficients')
        
    
        w_residual *= mask_coeff

        
        if iteration==1:
            if self._verbose: print('(*) Reconstructing the new signal coefficients into the real space')
        # Reconstruct the delta from the masked residual
        delta = self.mr2d1d.reconstruct(w_residual)

        if iteration==1:
            if self._verbose: print('(*) Applying the positivity constraint')
        
        delta = np.maximum(0, delta)




        
        if iteration==1:
            if self._verbose: print('(*) Updating the model with the newly detected signal')
        # Update the model
        model = model + delta
        model = np.maximum(0, model)
        

        if self._plot:

            
            im4 = axs[1,0].imshow(residual[iz], cmap = 'RdBu_r')#, vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,0].set_title('Residual')

            im5 = axs[1,1].imshow(delta[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,1].set_title('Residual Information')

            im6 = axs[1,2].imshow(model[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,2].set_title('Updated Model (Iteration #{})'.format(iteration+1))


            cbar4 = fig.colorbar(im4, ax=axs[1, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar5 = fig.colorbar(im5, ax=axs[1, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar6 = fig.colorbar(im6, ax=axs[1, 2], orientation='horizontal', fraction=0.05, pad=0.02)


            cbar4.ax.tick_params()#labelsize=20)
            cbar5.ax.tick_params()#labelsize=20)
            cbar6.ax.tick_params()#labelsize=20)

            cbar4.set_label('Flux')
            cbar5.set_label('Flux')
            cbar6.set_label('Flux')

            # Show the plot

            axs[1,0].axis('off')
            axs[1,1].axis('off')
            axs[1,2].axis('off')

            #plt.tight_layout()

            plt.subplots_adjust(hspace=1)  # Increase vertical gap between rows

            plt.show()


        return model, delta
        

    def _denoise_iterative_hard(self, num_iter=20):
        """Denoise the data using iterative thresholding with a fixed sparsity mask.

        Parameters
        ----------
        num_iter : int
            Number of iterations. Default: 20

        Returns
        -------
        array_like
            Denoised array
        """

        if self._verbose:
            print('L0 regularisation : HARD thresholding')
            print('{} denoising iterations'.format(num_iter))

        # Initialize the model
        model = self._data.copy()

        flux_history = []
        residual_stds = []
        previous_residual_std = 1e-33

        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, max_y, max_x = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        deltas = np.zeros_like(self._data)

        p_init = 4
        epsilon = 1e-3

        
        converged = False

        

        for p in range(p_init, -1, -1):  # Try p from 5 down to 0
            if self._verbose:
                print(f'\n[*] Trying with plateau condition: {p} consecutive stable residuals needed for convergence')

            plateau_counter = 0
            previous_residual_std = 1e-33  # Reset each time p changes
            best_model = None
            best_iteration = 0
            min_residual_std = np.inf

            dists = []
            noise_levels = [] 
            for iteration in range(num_iter):

                if self._verbose: print('\n\n--- [ DE-NOISING ITERATION #{} ] ---\n'.format(iteration + 1))

                if iteration == 0:

                    
                    # First iteration: decompose `self._data`, compute mask, and denoise

                    if self._verbose: print('(*) Decomposing the noisy data into wavelet coefficients')
                    inds, shapes, w_data = self.mr2d1d.decompose(self._data,
                                                                self._num_scales_2d,
                                                                self._num_scales_1d)

                   
                    mask_coeff = np.zeros_like(w_data)
                    mask_coeff_final = np.zeros_like(w_data)

                    if self._verbose: print('(*) Applying hard thresholding in the wavelet space based on the threshold scale (lambda = {} in noise units)'.format(self._threshold_level))

                    for scale2d in range(self._num_scales_2d):
                        for scale1d in range(self._num_scales_1d):
                            
                            start, end = inds[scale2d][scale1d]

                            if scale2d == self._num_scales_2d - 1 and scale1d == self._num_scales_1d - 1:
                                continue

                            c_data = w_data[start:end]

                            noise_level = self._estimate_noise(c_data)

                            
                            thresh = self._threshold_level
                            mask_coeff[start:end] = self._generate_hard_threshold_mask(c_data, thresh, noise_level)

                    if self._verbose: print('(*) Calculating a mask in wavelet space...')
                    if self._verbose: print('(*) Applying the mask to perform denoising for the first iteration')
                    w_data = w_data * mask_coeff#_final #mask_coeff
    

                    if self._verbose: print('(*) Reconstructing the denoised data from wavelet to real space')
                    model = self.mr2d1d.reconstruct(w_data)
                    if self._verbose: print('(*) Applying positivity constraint')
                    model = np.maximum(0, model)
                    model_print = model.copy()

                    

                else:

                    model, delta = self._residual_signal_extraction_l0(model, mask_coeff, iteration)

                    if iteration == 1 and self._verbose:
                        print('Goal: Find previously unnoticed signal in residuals')
                        print('(*) Calculating residual for this iteration')

                    deltas += delta

                aperture_flux = np.sum(model)
                residual_std = np.std(model)

                if self._verbose:
                    print(f"(*) Aperture Flux: {aperture_flux:.3e}, Residual noise std: {residual_std:.3e}")

                flux_history.append(aperture_flux)
                residual_stds.append(residual_std)

                # Track best model so far
                if residual_std < min_residual_std:
                    min_residual_std = residual_std
                    best_model = model.copy()
                    best_iteration = iteration + 1

                # Plateau condition check
                if abs(residual_std - previous_residual_std) / previous_residual_std <= epsilon:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if plateau_counter >= p:
                    if self._verbose:
                        print(f'\nflux: {aperture_flux}')
                        print(f'noise: {residual_std}')
                        print(f'Convergence achieved at iteration #{iteration + 1} with p = {p}')
                    converged = True
                    break
                else:
                    previous_residual_std = residual_std

                if iteration == 1 and self._verbose:
                    print(f'(*) Repeating these steps for subsequent {num_iter - 2} iterations')

            if converged:
                break

        if not converged:
            if self._verbose:
                print('[Warning] Convergence not achieved for any p value from 5 to 0')
                print(f'Using best model at iteration #{best_iteration} with residual std = {min_residual_std:.3e}')

        

        return best_model, deltas, residual_stds, best_iteration, noise_levels






    

#--------
#  SOFT
#--------

       

    def _residual_signal_extraction_l1(self, model, mask_coeff, all_weights, iteration, noise_levels):

        """
        Perform a single iteration of residual signal extraction using L1 soft-thresholding.

        This method decomposes the residual between the current model and the true data
        into wavelet scales, applies weighted soft-thresholding, reconstructs the residual,
        and updates the model with newly detected signal while enforcing positivity.

        Parameters
        ----------
        model : np.ndarray
            Current model of the data to be updated.
        mask_coeff : np.ndarray
            Boolean array indicating coefficients to apply weighted thresholding.
        all_weights : np.ndarray
            Array of weights for adaptive soft-thresholding.
        iteration : int
            Current iteration number.
        noise_levels : list of float
            Noise level estimates for each wavelet sub-band.

        Returns
        -------
        model : np.ndarray
            Updated model after residual signal extraction.
        delta : np.ndarray
            Extracted residual signal added to the model in this iteration.

        Notes
        -----
        - Positivity is enforced on the delta before updating the model.
        - The coarse scale (last wavelet scale) is not thresholded.
        - Optional plotting visualizes the model, residual, and delta at the selected slice.
        """
        
        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, _, _ = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        residual = self._data - model
        thresh = self._threshold_level
        
        if self._plot:
            fig, axs = plt.subplots(2, 3, figsize=(16, 13), constrained_layout=True)

            im1 = axs[0,0].imshow(model[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,0].set_title('Previously Denoised (Iteration #{})'.format(iteration))

            im2 = axs[0,1].imshow((self._signal - model)[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,1].set_title('SIGNAL Residual')

            im3 = axs[0,2].imshow(self._signal[iz], vmin = np.min(self._signal[iz]), vmax = np.max(self._signal[iz]), cmap = 'RdBu_r')
            axs[0,2].set_title('SIGNAL')

            axs[0,0].axis('off')
            axs[0,1].axis('off')
            axs[0,2].axis('off')

            cbar1 = fig.colorbar(im1, ax=axs[0, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar2 = fig.colorbar(im2, ax=axs[0, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar3 = fig.colorbar(im3, ax=axs[0, 2], orientation='horizontal', fraction=0.05, pad=0.02)

            cbar1.ax.tick_params()#labelsize=20)
            cbar2.ax.tick_params()#labelsize=20)
            cbar3.ax.tick_params()#labelsize=20)

            cbar1.set_label('Flux')
            cbar2.set_label('Flux')
            cbar3.set_label('Flux')


        if iteration==1:
            if self._verbose: print('(*) Decomposing residual into wavelet scales')

        inds, shapes, w_residual = self.mr2d1d.decompose(residual,
                                                            self._num_scales_2d,
                                                            self._num_scales_1d)

        # Apply the mask computed from the first iteration
        if iteration==0:
            if self._verbose: print('(*) Performing Weighted de-biasing with previously calculated weights')
        
        i=0 
        for scale2d in range(self._num_scales_2d):
            for scale1d in range(self._num_scales_1d):

                start, end = inds[scale2d][scale1d]

                if scale2d==self._num_scales_2d-1 and scale1d == self._num_scales_1d - 1:
                    w_residual[start:end] = np.zeros_like(w_residual[start:end])
                    continue  # Do not threshold the coarse scale

               
                c_data = w_residual[start:end] - np.median(w_residual[start:end])

                noise_level = noise_levels[i] #self._estimate_noise(c_data)

                # Corrected: Fetch the mask properly
                mask = mask_coeff[start:end].astype(bool)
                weights = all_weights[start:end]

         
                # Apply soft-thresholding with adaptive weights where mask is True
                w_residual[start:end][mask] = np.sign(c_data[mask]) * np.maximum(
                    np.abs(c_data[mask]) - weights[mask] * thresh * noise_level, 0.0
                )

                # Apply uniform soft-thresholding where mask is False
                w_residual[start:end][~mask] = np.sign(c_data[~mask]) * np.maximum(
                    np.abs(c_data[~mask]) - thresh * noise_level, 0.0
                )

                i+=1





        if iteration==0:
            if self._verbose: print('(*) Reconstructing the new signal coefficients into the real space')
        # Reconstruct the delta from the masked residual
        # Remove coarse mean in wavelet residual
        #w_residual = w_residual - np.mean(w_residual)
        #w_residual -= w_residual.mean()
        delta = self.mr2d1d.reconstruct(w_residual)





        if iteration==0:
            if self._verbose: print('(*) Updating the model with the newly detected signal')
         # Ensure positivity and flux conservation
        delta = np.maximum(0, delta)
        residual_flux = np.sum(residual)
        delta_flux = np.sum(delta)
        #if delta_flux != 0:
            #delta *= residual_flux / delta_flux

        delta -= delta.mean()

        # Update model
        model = model + delta
        if iteration==0:
            if self._verbose: print('(*) Applying the positivity constraint')
        #model = np.maximum(0, model)
        

        if self._plot:

            
            im4 = axs[1,0].imshow(residual[iz], cmap = 'RdBu_r')#, vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,0].set_title('Residual')

            im5 = axs[1,1].imshow(delta[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,1].set_title('Residual Information')

            im6 = axs[1,2].imshow(model[iz], cmap = 'RdBu_r', vmin = self._signal[iz].min(), vmax = self._signal[iz].max())
            axs[1,2].set_title('Updated Model (Iteration #{})'.format(iteration+1))


            cbar4 = fig.colorbar(im4, ax=axs[1, 0], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar5 = fig.colorbar(im5, ax=axs[1, 1], orientation='horizontal', fraction=0.05, pad=0.02)
            cbar6 = fig.colorbar(im6, ax=axs[1, 2], orientation='horizontal', fraction=0.05, pad=0.02)


            cbar4.ax.tick_params()#labelsize=20)
            cbar5.ax.tick_params()#labelsize=20)
            cbar6.ax.tick_params()#labelsize=20)

            cbar4.set_label('Flux')
            cbar5.set_label('Flux')
            cbar6.set_label('Flux')

            # Show the plot

            axs[1,0].axis('off')
            axs[1,1].axis('off')
            axs[1,2].axis('off')

            #plt.tight_layout()

            plt.subplots_adjust(hspace=1)  # Increase vertical gap between rows

            plt.show()


        return model, delta
    

    def _denoise_iterative_soft(self, num_iter_reweight=20, num_iter_debias=20, debias = True):

        """
        Perform iterative soft-thresholding denoising on the 3D data cube.

        This method applies multiple re-weighting iterations followed by an optional
        debiasing step to recover the underlying signal from noisy data. It uses a 
        2D-1D multiscale wavelet decomposition for denoising, adaptive thresholding, 
        and plateau-based convergence criteria.

        Parameters
        ----------
        num_iter_reweight : int, optional
            Number of iterations for the re-weighting denoising step (default is 20).
        num_iter_debias : int, optional
            Number of iterations for the debiasing step to extract residual signal (default is 20).
        debias : bool, optional
            Whether to perform the debiasing step (default is True).

        Returns
        -------
        best_model : np.ndarray
            Final denoised model after all iterations.
        model_1_step : np.ndarray
            Model after the first re-weighted denoising iteration.
        model_no_reweight : np.ndarray
            Model obtained without re-weighting in the first iteration.
        deltas : np.ndarray
            Accumulated residual signals extracted during debiasing.
        residual_stds_reweight : list of float
            Standard deviations of residuals during the re-weighting step.
        residual_stds_debias : list of float
            Standard deviations of residuals during the debiasing step.
        best_iteration : int
            Iteration number where the best model (lowest residual std) was achieved.
        dists : list of np.ndarray
            Selected wavelet sub-band distributions for diagnostic purposes.
        noise_levels : list of float
            Estimated noise levels for each wavelet sub-band.

        Notes
        -----
        - Positivity is enforced on the denoised models at each step.
        - Convergence is determined via a plateau condition on the residual standard deviation.
        - Optional plotting shows intermediate and final results for diagnostics.
        """

        if self._verbose:
            
            print('----[ Denosiing with ITERATIVE SOFT THRESHOLDING ]----')



        # Initialize the model
        self.mean, self.std = np.mean(self._data), np.std(self._data)
        model = self._data.copy()
        thresh = self._threshold_level
        
        
        p_init = 1

        max_voxel_index = np.argmax(self._signal)  # Get flattened index
        iz, max_y, max_x = np.unravel_index(max_voxel_index, self._signal.shape)  # Convert to 3D index

        converged = False
        for p in range(p_init, -1, -1):  # Try p from 5 down to 0
            if self._verbose:
                print(f'\n[*] Trying with plateau condition: {p} consecutive stable residuals needed for convergence')

            plateau_counter = 0
            previous_residual_std = 1e-33  # Reset each time p changes
            min_residual_std = np.inf
            dists = []
            noise_levels = []
            epsilon = 1e-3


            flux_history = []
            residual_stds_reweight = []

            inds, shapes, w_data_data = self.mr2d1d.decompose(self._data,
                                            self._num_scales_2d,
                                            self._num_scales_1d)
            
            for scale2d in range(self._num_scales_2d):
                    for scale1d in range(self._num_scales_1d):
                        start, end = inds[scale2d][scale1d]
                        if (scale2d==5) and (scale1d==0):
                            dists.append(w_data_data[start:end])



            
            for iteration in range(num_iter_reweight):


            
                if self._verbose:   print('\n\n--- [ DE-NOISING ITERATION #{} ] ---\n'.format(iteration+1))



                if iteration==0:
                    if self._verbose: print('(*) Decomposing noisy data into wavelet coefficients')
                inds, shapes, w_data_weights = self.mr2d1d.decompose(model,
                                            self._num_scales_2d,
                                            self._num_scales_1d)
                            
        
                
                #model_update = np.zeros_like(model)
                #model_update = data3d.copy()

                mu = 0.5
                gradient = -2*(self._data - model)

                model_update = model - mu * gradient


                inds, shapes, w_data = self.mr2d1d.decompose(model_update,
                                                            self._num_scales_2d,
                                                            self._num_scales_1d)
            
                if iteration == 0:
                    if self._verbose: print('(*) The gradient in the first iteration is 0')


                if self._verbose:
                    if iteration==1:
                        print('(*) Updating model with gradient with respect to data')
                        print('(*) Calculating weights for each iteration (except #1) to account for the soft thresholding bias')
                # Loop through all sub-bands (excluding last/coarse scale) to find the max usable threshold based on noise levels in each

                w_data_copy = w_data.copy()

                mask_coeff = np.zeros_like(w_data, dtype=bool)

                all_weights = np.ones_like(w_data)

                noise_levels = []

                for scale2d in range(self._num_scales_2d):
                    for scale1d in range(self._num_scales_1d):

                        start, end = inds[scale2d][scale1d]

                        # Skip the coarse scale (approximation band) in the 1D spectral transform
                        #if scale2d==self._num_scales_2d-1 and scale1d == self._num_scales_1d - 1: 
                            #continue  # Do not threshold the coarse scale


                        #w_data[start:end]  -= np.median(w_data[start:end])
                        c_data = w_data[start:end] 
                        #c_data -= c_data.median()
                        
                        noise_level = self._estimate_noise(c_data)
                        noise_levels.append(noise_level)

                        noise_level_weight = self._estimate_noise(w_data_weights[start:end])

                    
                        # Compute the mask
                        mask = np.abs(c_data) > thresh * noise_level




                        # Compute weights only where mask is True
                        if iteration == 0:
                            weights = np.ones_like(c_data)
                        else:
                            weights = thresh*noise_level_weight/(np.abs(w_data_weights[start:end]) + 1e-33)

                        all_weights[start:end] = weights
                        mask_coeff[start:end] = mask
                        # Apply soft-thresholding with adaptive weights where mask is True
                        w_data[start:end][mask] = np.sign(c_data[mask]) * np.maximum(
                            np.abs(c_data)[mask] - weights[mask] * thresh * noise_level, 0.0
                        )

                        # Apply uniform soft-thresholding where mask is False
                        w_data[start:end][~mask] = np.sign(c_data[~mask]) * np.maximum(
                            np.abs(c_data)[~mask] - thresh * noise_level, 0.0
                        )

                    

                        #w_data[start:end] -= w_data[start:end].mean()
                        #c_data[~mask] = 0.0
                        #w_data[start:end][~mask] = 0.0
                        



                        if (scale2d==5) and (scale1d==0):
                            dists.append(w_data[start:end])
                            noise_levels.append(noise_level)

                        if self._plot:
                            if (scale2d==5) and (scale1d==0):

                                denosied_dist = w_data[start:end]
                                threshold_noise = thresh * noise_level

                                bins = np.linspace(w_data_copy[start:end].min(), w_data_copy[start:end].max(),100)

                                plt.figure(figsize = (11,7))
                                plt.hist(w_data[start:end], bins = bins, color = 'xkcd:blue', alpha = 0.5, label = 'Denoised')
                                #plt.hist(w_data_weights[start:end], bins = bins, color = 'xkcd:blue', alpha = 0.5, label = 'Denoised')
                                plt.hist(w_data_copy[start:end], bins = bins, histtype='step', color = 'black', alpha = 1, label = 'Original')
                                plt.axvline(thresh * noise_level, color = 'black', linestyle = 'dashed', label = '{:.1f}'.format(self._threshold_level)+r'$\sigma$' + ' Threshold')
                                plt.axvline(-thresh * noise_level, color = 'black', linestyle = 'dashed')
                                #plt.title('Iteration {}\n2D: {}, 1D: {}'.format(iteration+1, scale2d+1, scale1d+1))
                                plt.yscale('log')
                                plt.ylim(0,5e5)
                                #plt.xlim(-0.005, 0.005)
                                plt.ylabel('$N_{C_{ij}}$')
                                plt.xlabel('$C_{ij}$')
                                plt.legend()

                                plt.grid(True)
                                plt.show()


                # Reconstruct the image from the updated coefficients


                if iteration==0:
                        if self._verbose: print('(*) Reconstructing the new signal coefficients into the real space')
        
                model_denoised = self.mr2d1d.reconstruct(np.ascontiguousarray(w_data, dtype=np.float32))
                model_print = self.mr2d1d.reconstruct(np.ascontiguousarray(w_data_copy, dtype=np.float32))


                if iteration==0:
                        if self._verbose: print('(*) Applying the positivity constraint')
                model_denoised = np.maximum(0, model_denoised) #positivity constraint


                
                if self._plot:
                    plt.figure(figsize = (15,12))
                    plt.subplot(221)
                    plt.imshow(model_print[iz], cmap = 'RdBu_r',  vmin=self._data[iz].min(), vmax=self._data[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('Input')

                    plt.subplot(222)
                    plt.imshow(model_denoised[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('Denoised Iteration #{}'.format(iteration+1))

                    plt.figure(figsize = (15,12))
                    plt.subplot(223)
                    plt.imshow(self._signal[iz], cmap = 'RdBu_r',  vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('Signal')

                    plt.subplot(224)
                    plt.imshow((self._signal - model_denoised)[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
                    plt.colorbar()
                    plt.axis('off')
                    plt.title('SIGNAL Residual')

                    
                    plt.show()


            
                

                if iteration==0:
                    if self._verbose: print('(*) Repeating these steps for subsequent iterations')

                model = model_denoised# - model_denosied.mean()

                aperture_flux = np.sum(model)
                residual_std = np.std(self._data - model)

                residual_stds_reweight.append(residual_std)

                if self._verbose:
                    print(f"(*) Aperture Flux: {aperture_flux:.3e}, Clean Flux: {np.sum(self._signal):.3e}, Residual STD: {residual_std:.3e}")


                # Track best model so far
                if residual_std < min_residual_std:
                    min_residual_std = residual_std
                    best_model = model.copy()
                    best_iteration = iteration + 1

                # Plateau condition check
                if abs(residual_std - previous_residual_std) / previous_residual_std <= epsilon:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if plateau_counter >= p:
                    if self._verbose:
                        print(f'\nflux: {aperture_flux}')
                        print(f'noise: {residual_std}')
                        print(f'Re-weight Convergence achieved at iteration #{iteration + 1} with p = {p}')
                    converged = True
                    break
                else:
                    previous_residual_std = residual_std

                if iteration == 1 and self._verbose:
                    print(f'(*) Repeating these steps for subsequent {num_iter_reweight - 2} iterations')

                if iteration==0:
                    model_no_reweight = model_denoised

            if converged:
                break

        if not converged:
            if self._verbose:
                print('[Warning] Re-weight convergence not achieved for any p value from 5 to 0')
                print(f'Using best model at iteration #{best_iteration} with residual std = {min_residual_std:.3e}')

        


        if self._verbose:   '\n----[ DE-BIASING ]----\n'

        if self._verbose:   print('(*) Iteratively extracting remaining signal from residual')


        p_init_debias=1
        epsilon_debias = 5e-4
        model_1_step = best_model.copy()

        for p in range(p_init_debias, -1, -1):  # Try p from 5 down to 0
            if self._verbose:
                print(f'\n[*] Trying with plateau condition: {p} consecutive stable residuals needed for convergence')

            plateau_counter = 0
            previous_residual_std = 1e-33  # Reset each time p changes
            deltas = np.zeros_like(model)
            min_residual_std = np.inf
            residual_stds_debias=[]

            model = model_1_step.copy()

            for iteration in range(num_iter_debias):

                model, delta = self._residual_signal_extraction_l1(model, mask_coeff, all_weights, iteration, noise_levels)
                model = np.maximum(0, model)  # Apply positivity constraint
                deltas+=delta


                aperture_flux = np.sum(model)
                residual_std = np.std(self._data - model)

                if self._verbose:
                    print(f"(*) Aperture Flux: {aperture_flux}, Residual STD: {residual_std:.3e}")


                
                flux_history.append(aperture_flux)
                residual_stds_debias.append(residual_std)

                # Track best model so far
                if residual_std < min_residual_std:
                    min_residual_std = residual_std
                    best_model = model.copy()
                    best_iteration = iteration + 1


                # Plateau condition check
                if abs(residual_std - previous_residual_std) / previous_residual_std <= epsilon_debias:
                    plateau_counter += 1
                else:
                    plateau_counter = 0

                if plateau_counter >= p:
                    if self._verbose:
                        print(f'\nflux: {aperture_flux}')
                        print(f'noise: {residual_std}')
                        print(f'Convergence achieved at iteration #{iteration + 1} with p = {p}')
                    converged = True
                    break
                else:
                    previous_residual_std = residual_std

                if iteration == 1 and self._verbose:
                    print(f'(*) Repeating these steps until convergence')

            if converged:
                break

        if not converged:
            if self._verbose:
                print('[Warning] Convergence not achieved for any p value from 5 to 0')
                print(f'Using best model at iteration #{best_iteration} with residual std = {min_residual_std:.3e}')

        if self._plot:
            plt.figure(figsize=(28, 11))
            plt.subplot(121)
            plt.imshow(self._data[iz], cmap='RdBu_r')
            plt.title('Noisy Data')
            plt.colorbar()

            plt.subplot(122)
            plt.imshow(self._signal[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('Clean Signal')
            plt.colorbar()
            plt.show()

            plt.figure(figsize=(28, 9))
            plt.subplot(131)
            plt.imshow(model_1_step[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('One-Step Denoising')
            plt.colorbar()
            plt.axis('off')

            plt.subplot(132)
            plt.imshow(np.maximum(deltas, 0)[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('Residual Signal')
            plt.colorbar()
            plt.axis('off')

            plt.subplot(133)
            plt.imshow(best_model[iz], cmap='RdBu_r', vmin=self._signal[iz].min(), vmax=self._signal[iz].max())
            plt.title('Final Denoised')
            plt.colorbar()
            plt.axis('off')
            plt.show()

                
        return best_model, model_1_step, model_no_reweight, deltas, residual_stds_reweight, residual_stds_debias, best_iteration, dists, noise_levels #, denosied_dist, threshold_noise






    # RMSE computation function
    def _compute_emission_rmse(self, model):
        rmse = np.sqrt(np.mean((self._mask * self._signal - self._mask * model) ** 2))
        return rmse

    @staticmethod
    def _prox_positivity_constraint(array):
        """Proximal operator of the positivity constraint
        
        Parameters
        ----------
        array : array_like
            Any array that supports index slicing
        
        Returns
        -------
        array_like
            Array with all negative entries set to zero
        
        """
        return np.maximum(0, array)


    def _estimate_noise(self, array):
        """Estimate noise standard deviation from the median absolute deviation (MAD)

        Parameters
        ----------
        array : array_like
            Values on which the noise is estimated

        Returns
        -------
        float
            Noise standard deviation
        
        """
        mad = np.median(np.abs(array - np.median(array)))
        return 1.48 * mad

def mock_noise_value(mock_cube, peak_snr):

    print('max snr', peak_snr)
    mock_cube_noise = np.max(mock_cube)/peak_snr
    print('mock noise', np.max(mock_cube_noise))

    return mock_cube_noise