import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import random
import cv2
from PIL import Image
from io import BytesIO
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

# --- Dependency Imports for Original Functions (Keep if needed) ---
try:
    from wand.image import Image as WandImage
    from wand.api import library as wandlibrary
    import ctypes
    # Tell Python about the C method
    wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                                  ctypes.c_double,  # radius
                                                  ctypes.c_double,  # sigma
                                                  ctypes.c_double)  # angle

    # Extend wand.image.Image class to include method signature
    class MotionImage(WandImage):
        def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
            wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
    _wand_available = True
except ImportError:
    _wand_available = False
    warnings.warn("Wand library not found. MotionBlur augmentation will not be available.")

warnings.simplefilter("ignore", UserWarning)

# Helper function to convert tensor to PIL Image
def tensor_to_pil(img_tensor):
    """Converts a [C, H, W] tensor (0-1 range) to a PIL Image."""
    # Ensure tensor is on CPU and detach from graph
    img_tensor = img_tensor.cpu().detach()
    # Handle potential normalization if needed (assuming input is 0-1)
    # img_tensor = img_tensor * 0.5 + 0.5 # Example if input was [-1, 1]
    img_tensor = torch.clamp(img_tensor, 0, 1)
    # Convert to [H, W, C] and then to PIL Image
    if img_tensor.shape[0] == 1: # Grayscale
        img_tensor = img_tensor.squeeze(0)
        pil_img = TF.to_pil_image(img_tensor, mode='L')
    else: # RGB
        pil_img = TF.to_pil_image(img_tensor, mode='RGB')
    return pil_img

# Helper function to convert PIL Image back to Tensor
def pil_to_tensor(pil_img):
    """Converts a PIL Image to a [C, H, W] tensor (0-1 range)."""
    return TF.to_tensor(pil_img) # Outputs tensor in [0.0, 1.0] range

# Helper function to convert tensor to NumPy array (H, W, C) [0-255]
def tensor_to_numpy_uint8(img_tensor):
    """Converts a [C, H, W] tensor (0-1 range) to a NumPy uint8 array [H, W, C] (0-255 range)."""
    img_tensor = img_tensor.cpu().detach()
    img_tensor = torch.clamp(img_tensor * 255, 0, 255).byte()
    # Permute from [C, H, W] to [H, W, C]
    np_img = img_tensor.permute(1, 2, 0).numpy()
    # If grayscale, ensure it's (H, W) or (H, W, 1) as needed by some functions
    # if np_img.shape[2] == 1:
    #     np_img = np_img.squeeze(2)
    return np_img

# Helper function to convert NumPy array (H, W, C) [0-255] back to Tensor
def numpy_uint8_to_tensor(np_img):
    """Converts a NumPy uint8 array [H, W, C] or [H, W] (0-255 range) to a [C, H, W] tensor (0-1 range)."""
    if np_img.ndim == 2: # Handle grayscale that might be (H, W)
        np_img = np_img[:, :, np.newaxis]
    # Ensure it's uint8
    if np_img.dtype != np.uint8:
         np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    # Permute from [H, W, C] to [C, H, W]
    tensor = torch.from_numpy(np_img.transpose((2, 0, 1)))
    # Convert to float and scale to [0.0, 1.0]
    return tensor.float().div(255.0)

# --- Original Helper Functions (Needed by some augmentations) ---
# (Copied from the provided script, potentially modified for direct use)

def disk(radius, alias_blur=0.1, dtype=np.float32):
    """Creates a 2D disk kernel."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    # Check if the sum is zero before dividing
    disk_sum = np.sum(aliased_disk)
    if disk_sum > 0:
        aliased_disk /= disk_sum
    else:
        # Handle the case where the disk is empty (e.g., radius=0)
        # Maybe return a delta function or handle appropriately
        aliased_disk = np.zeros_like(aliased_disk)
        center = tuple(s // 2 for s in aliased_disk.shape)
        if all(c >= 0 and c < s for c, s in zip(center, aliased_disk.shape)):
             aliased_disk[center] = 1.0


    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def plasma_fractal(mapsize=32, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm."""
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    # Avoid division by zero if maparray is constant
    map_max = maparray.max()
    if map_max > 0:
        return maparray / map_max
    else:
        return maparray # Return the constant array (likely all zeros)

def clipped_zoom(img, zoom_factor):
    """Zooms into the center of an image and crops back to original size."""
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    if ch > h: # Handle cases where zoom_factor < 1, prevent negative slicing
        ch = h
        top = 0
    else:
        top = (h - ch) // 2

    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    # Ensure trimming indices are valid
    if trim_top < 0:
        # This can happen if the zoomed image is smaller than h due to rounding/zoom_factor < 1
        # Pad the image instead of trimming
        pad_before = abs(trim_top)
        pad_after = h - img.shape[0] - pad_before
        img = np.pad(img, ((pad_before, pad_after), (pad_before, pad_after), (0, 0)), mode='constant')
        trim_top = 0 # Reset trim_top as padding handled it

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# --- Augmentation Layers ---

class GaussianNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        self.c = [0.04, 0.06, .08, .09, .10][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        noise = torch.randn_like(x) * self.c
        x_noisy = x + noise
        return torch.clamp(x_noisy, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'

class ShotNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # Original parameter 'c' represents the scaling factor applied *before* Poisson.
        # In torch.poisson, the input rate is lambda. rate = input * c.
        # So, the output is Poisson(input * c). We then divide by c.
        self.c = [500, 250, 100, 75, 50][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        # Calculate lambda = x * c. Ensure lambda >= 0.
        rate = torch.clamp(x * self.c, min=0)
        # Apply Poisson noise. The output of torch.poisson is float.
        x_noisy = torch.poisson(rate) / self.c
        return torch.clamp(x_noisy, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'

class ImpulseNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        self.c = [.01, .02, .03, .05, .07][self.severity - 1] # proportion of pixels to replace

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        B, C, H, W = x.shape
        # Create masks for salt and pepper noise
        # Total pixels to change per image: num_pixels * c
        num_pixels = C * H * W
        num_sp_pixels = int(num_pixels * self.c)
        if num_sp_pixels == 0: return x # No noise if c is too small

        # Generate random indices for noise for the whole batch
        # We need B * num_sp_pixels indices in total
        # Flatten spatial and channel dimensions for easier indexing
        x_flat = x.view(B, -1) # Shape [B, C*H*W]

        # Generate unique random indices per image in the batch
        noisy_batch = []
        for i in range(B):
            img_flat = x_flat[i]
            indices = torch.randperm(num_pixels, device=x.device)[:num_sp_pixels]

            # Decide which indices get salt (1.0) and which get pepper (0.0)
            num_salt = random.randint(0, num_sp_pixels)
            salt_indices = indices[:num_salt]
            pepper_indices = indices[num_salt:]

            # Apply noise
            img_noisy_flat = img_flat.clone()
            img_noisy_flat[salt_indices] = 1.0
            img_noisy_flat[pepper_indices] = 0.0
            noisy_batch.append(img_noisy_flat.view(C, H, W))

        x_noisy = torch.stack(noisy_batch)
        return torch.clamp(x_noisy, 0.0, 1.0) # Clamp shouldn't be necessary here

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'


class SpeckleNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        self.c = [.06, .1, .12, .16, .2][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        noise = torch.randn_like(x) * self.c
        x_noisy = x + x * noise # Multiplicative noise
        return torch.clamp(x_noisy, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'

class GaussianBlur(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # Map severity to sigma. Kernel size is determined automatically in TF.gaussian_blur
        self.sigma = [0.4, 0.6, 0.7, 0.8, 1.0][self.severity - 1]
        # Kernel size needs to be odd
        # A common heuristic: kernel_size ≈ 6*sigma + 1
        k = int(round(6 * self.sigma + 1))
        self.kernel_size = k if k % 2 == 1 else k + 1


    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        # TF.gaussian_blur expects a list of sigmas or a float sigma
        # It applies the same blur to all images in the batch
        return TF.gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, sigma={self.sigma}, kernel_size={self.kernel_size})'


class GlassBlur(nn.Module):
    """
    Wraps the original glass_blur function.
    Requires scikit-image. Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # sigma, max_delta, iterations
        self.params = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]

    def _glass_blur_single(self, np_img):
        # np_img is uint8 [H, W, C]
        sigma, max_delta, iterations = self.params

        # Apply initial Gaussian blur
        img_blurred = gaussian(np_img / 255., sigma=sigma, channel_axis=-1) * 255
        img_blurred = np.uint8(img_blurred) # Convert back to uint8 for shuffling

        # Locally shuffle pixels
        h, w = img_blurred.shape[:2]
        img_shuffled = img_blurred.copy() # Work on a copy

        for _ in range(iterations):
            for y in range(h - max_delta, max_delta -1, -1): # Iterate downwards to avoid using already swapped pixels
                 for x in range(w - max_delta, max_delta -1, -1):
                    # Generate random displacement within [-max_delta, max_delta]
                    dx, dy = np.random.randint(-max_delta, max_delta + 1, size=(2,))
                    # Calculate neighbor coordinates, ensuring they are within bounds
                    y_prime = np.clip(y + dy, 0, h - 1)
                    x_prime = np.clip(x + dx, 0, w - 1)

                    # Swap pixels between (y, x) and (y_prime, x_prime)
                    # Use a temporary variable for swapping
                    temp = img_shuffled[y, x].copy()
                    img_shuffled[y, x] = img_shuffled[y_prime, x_prime]
                    img_shuffled[y_prime, x_prime] = temp

        # Apply final Gaussian blur
        img_final_blur = gaussian(img_shuffled / 255., sigma=sigma, channel_axis=-1)
        return np.clip(img_final_blur * 255, 0, 255).astype(np.uint8)


    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
            # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._glass_blur_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class DefocusBlur(nn.Module):
    """
    Applies defocus blur using a disk kernel.
    Requires OpenCV (cv2). Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # radius, alias_blur
        self.params = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    def _defocus_blur_single(self, np_img):
        # np_img is uint8 [H, W, C]
        radius, alias_blur = self.params
        kernel = disk(radius=radius, alias_blur=alias_blur)

        # Apply filter2D for each channel
        channels = []
        img_float = np_img.astype(np.float32) / 255.0 # Work with float for filtering
        for d in range(img_float.shape[2]): # Iterate through channels
            channels.append(cv2.filter2D(img_float[:, :, d], -1, kernel))

        # Stack channels back and handle potential dimension issues if grayscale
        if len(channels) > 1:
            blurred_img = np.array(channels).transpose((1, 2, 0)) # C x H x W -> H x W x C
        elif len(channels) == 1:
             blurred_img = channels[0][:, :, np.newaxis] # Add channel dim back
        else: # Should not happen if input has channels
             blurred_img = img_float

        return np.clip(blurred_img * 255, 0, 255).astype(np.uint8)

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
            # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._defocus_blur_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class MotionBlur(nn.Module):
    """
    Applies motion blur using the Wand library.
    Requires Wand and ImageMagick. Operates on CPU via PIL conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        if not _wand_available:
            raise ImportError("Wand library is required for MotionBlur but not found.")
        self.severity = severity
        # radius, sigma
        self.params = [(6,1), (6,1.5), (6,2), (8,2), (9,2.5)][severity - 1]

    def _motion_blur_single(self, pil_img):
        # pil_img is a PIL Image object
        radius, sigma = self.params
        angle = np.random.uniform(-45, 45) # Random angle per image

        try:
            # Convert PIL image to WandImage
            output = BytesIO()
            # Ensure saving in a format Wand understands well, like PNG
            pil_img.save(output, format='PNG')
            output.seek(0) # Reset buffer position
            wand_img = MotionImage(blob=output.getvalue())

            # Apply motion blur
            wand_img.motion_blur(radius=radius, sigma=sigma, angle=angle)

            # Convert back to NumPy array (Wand uses BGR by default)
            blob = wand_img.make_blob(format='BGR') # Get raw bytes in BGR
            # Decode using OpenCV
            # Using frombuffer instead of deprecated fromstring
            np_img_bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

            # Convert BGR to RGB
            np_img_rgb = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2RGB)

            # Handle potential grayscale conversion if original was grayscale
            # Note: Wand might force RGB, check output shape
            # if pil_img.mode == 'L' and np_img_rgb.shape[2] == 3:
                 # Convert back to grayscale if needed, e.g., taking the mean
                 # np_img_rgb = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2GRAY)
                 # np_img_rgb = np_img_rgb[:, :, np.newaxis] # Add channel dim back

            return np.clip(np_img_rgb, 0, 255).astype(np.uint8)

        except Exception as e:
             print(f"Error during motion blur: {e}")
             # Return original image data as numpy array in case of error
             return np.array(pil_img)


    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_pil = tensor_to_pil(x[i])
            corrupted_np = self._motion_blur_single(img_pil)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class ZoomBlur(nn.Module):
    """
    Applies zoom blur by averaging multiple zoomed versions.
    Requires SciPy (for zoom). Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # zoom factors
        self.zoom_factors = [np.arange(1, 1.06, 0.01),
                             np.arange(1, 1.11, 0.01),
                             np.arange(1, 1.16, 0.01),
                             np.arange(1, 1.21, 0.01),
                             np.arange(1, 1.26, 0.01)][severity - 1]

    def _zoom_blur_single(self, np_img):
        # np_img is uint8 [H, W, C]
        img_float = np_img.astype(np.float32) / 255.0
        out = np.zeros_like(img_float)

        for zoom_factor in self.zoom_factors:
            zoomed = clipped_zoom(img_float, zoom_factor)
            # Ensure shapes match before adding
            if zoomed.shape == out.shape:
                 out += zoomed
            else:
                 # Handle potential shape mismatches (e.g., due to rounding)
                 # Option 1: Resize zoomed to match out (might introduce blur)
                 # zoomed_resized = cv2.resize(zoomed, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
                 # if zoomed_resized.ndim == 2 and out.ndim == 3: # Add channel dim if needed
                 #    zoomed_resized = zoomed_resized[:, :, np.newaxis]
                 # out += zoomed_resized
                 # Option 2: Skip this zoom factor if shape doesn't match (safer)
                 warnings.warn(f"Shape mismatch in ZoomBlur (original: {out.shape}, zoomed: {zoomed.shape}). Skipping zoom factor {zoom_factor}.")
                 pass


        # Average the original image and the zoomed versions
        final_img = (img_float + out) / (len(self.zoom_factors) + 1)
        return np.clip(final_img * 255, 0, 255).astype(np.uint8)

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
             # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._zoom_blur_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity})'


class Fog(nn.Module):
    """
    Adds fog effect using plasma fractal noise.
    Requires NumPy. Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # fog_amount, wibbledecay
        self.params = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]

    def _fog_single(self, np_img):
        # np_img is uint8 [H, W, C]
        fog_amount, wibbledecay = self.params
        img_float = np_img.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Determine appropriate mapsize (power of 2 >= max(h, w))
        mapsize = 2**int(np.ceil(np.log2(max(h, w))))

        # Generate plasma fractal noise
        plasma = plasma_fractal(mapsize=mapsize, wibbledecay=wibbledecay)
        # Crop plasma to image size and add channel dimension
        plasma_cropped = plasma[:h, :w][..., np.newaxis]

        # Add fog
        foggy_img = img_float + fog_amount * plasma_cropped

        # Rescale to keep max value consistent (as in original code)
        max_val = img_float.max() # Max value of original image
        # Avoid division by zero if max_val is 0 or very close
        if max_val + fog_amount > 1e-6:
             foggy_img = foggy_img * max_val / (max_val + fog_amount)
        # If max_val is near zero, adding fog dominates, clamp later

        return np.clip(foggy_img * 255, 0, 255).astype(np.uint8)


    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
             # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._fog_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class Frost(nn.Module):
    """
    Adds frost effect by blending with a frost image.
    Requires OpenCV (cv2) and frost image files. Operates on CPU.
    NOTE: Assumes frost images are in a 'frosts/' subdirectory.
    """
    def __init__(self, severity=1, frost_dir='frosts'):
        super().__init__()
        self.severity = severity
        self.frost_dir = frost_dir
        # image_weight, frost_weight
        self.params = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
        self.frost_filenames = [
            'frost1.png', 'frost2.png', 'frost3.png',
            'frost4.jpg', 'frost5.jpg', 'frost6.jpg'
        ]
        # Pre-load or check frost files existence? For now, load on demand.

    def _load_random_frost_img(self, target_h, target_w):
        import os # Import here if not globally imported
        try:
            idx = np.random.randint(len(self.frost_filenames))
            frost_path = os.path.join(self.frost_dir, self.frost_filenames[idx])
            frost = cv2.imread(frost_path)
            if frost is None:
                raise FileNotFoundError(f"Could not load frost image: {frost_path}")

            # Original code resizes frost based on a fixed factor (0.2)
            # It might be better to resize based on target image size?
            # Let's stick to original for now, but this might need adjustment.
            frost_resized = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)

            # Randomly crop the resized frost to match target size
            fh, fw = frost_resized.shape[:2]
            if fh < target_h or fw < target_w:
                 # If resized frost is smaller than target, resize it up
                 # This differs from original, which would fail here.
                 frost_resized = cv2.resize(frost_resized, (target_w, target_h))
                 x_start, y_start = 0, 0
            else:
                 x_start = np.random.randint(0, fw - target_w + 1)
                 y_start = np.random.randint(0, fh - target_h + 1)

            frost_cropped = frost_resized[y_start:y_start + target_h, x_start:x_start + target_w]

            # Convert to RGB (OpenCV loads as BGR)
            return cv2.cvtColor(frost_cropped, cv2.COLOR_BGR2RGB)

        except Exception as e:
            warnings.warn(f"Could not load or process frost image: {e}. Returning None.")
            return None


    def _frost_single(self, np_img):
        # np_img is uint8 [H, W, C]
        img_weight, frost_weight = self.params
        h, w = np_img.shape[:2]

        frost_overlay = self._load_random_frost_img(h, w)

        if frost_overlay is not None:
            # Blend original image with frost overlay
            frosted_img = img_weight * np_img + frost_weight * frost_overlay
            return np.clip(frosted_img, 0, 255).astype(np.uint8)
        else:
            # Return original image if frost loading failed
            return np_img

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
             # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._frost_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class Snow(nn.Module):
    """
    Adds snow effect using noise, zoom, and motion blur.
    Requires SciPy, PIL, OpenCV, Wand. Operates on CPU.
    """
    def __init__(self, severity=1):
        super().__init__()
        if not _wand_available:
            raise ImportError("Wand library is required for Snow augmentation but not found.")
        self.severity = severity
        # loc, scale, zoom, threshold, mb_radius, mb_sigma, blend_factor
        self.params = [(0.1,0.2,1,0.6,8,3,0.95),
                       (0.1,0.2,1,0.5,10,4,0.9),
                       (0.15,0.3,1.75,0.55,10,4,0.9),
                       (0.25,0.3,2.25,0.6,12,6,0.85),
                       (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]

    def _snow_single(self, np_img):
         # np_img is uint8 [H, W, C]
        loc, scale, zoom, threshold, mb_radius, mb_sigma, blend_factor = self.params
        img_float = np_img.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Generate monochrome noise layer
        snow_layer = np.random.normal(size=(h, w), loc=loc, scale=scale)

        # Zoom noise layer
        # Use clipped_zoom which expects [H, W, C], add channel dim
        snow_layer_zoomed = clipped_zoom(snow_layer[..., np.newaxis], zoom)

        # Threshold the zoomed noise
        snow_layer_zoomed[snow_layer_zoomed < threshold] = 0

        # Convert to PIL Image for motion blur via Wand
        # Squeeze channel dim, scale to 0-255, convert to uint8
        snow_pil = Image.fromarray(
            (np.clip(snow_layer_zoomed.squeeze(), 0, 1) * 255).astype(np.uint8),
            mode='L' # Grayscale
        )

        # Apply motion blur using Wand
        try:
            output = BytesIO()
            snow_pil.save(output, format='PNG')
            output.seek(0)
            snow_wand = MotionImage(blob=output.getvalue())
            # Random angle for snow fall direction
            angle = np.random.uniform(-135, -45)
            snow_wand.motion_blur(radius=mb_radius, sigma=mb_sigma, angle=angle)

            # Decode back to NumPy array (Wand blob -> OpenCV decode)
            blob = snow_wand.make_blob(format='GRAY') # Get grayscale blob
            snow_blurred_np = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)

            # Scale back to 0-1 and add channel dimension
            snow_final = (snow_blurred_np / 255.0)[..., np.newaxis]

        except Exception as e:
            warnings.warn(f"Wand motion blur failed in Snow: {e}. Using unblurred snow layer.")
            # Fallback: use the zoomed, thresholded layer without blur
            snow_final = np.clip(snow_layer_zoomed, 0, 1)


        # Blend snow with original image
        # Original code has a complex blend involving grayscale version
        img_gray_blend = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5
        img_blend_base = np.maximum(img_float, img_gray_blend)
        img_blended = blend_factor * img_float + (1 - blend_factor) * img_blend_base

        # Add snow layer (and rotated version)
        snowy_img = img_blended + snow_final + np.rot90(snow_final, k=2)

        return np.clip(snowy_img * 255, 0, 255).astype(np.uint8)


    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
             # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._snow_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class Spatter(nn.Module):
    """
    Adds spatter effect (water or mud).
    Requires Scikit-Image, OpenCV. Operates on CPU.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # loc, scale, sigma, threshold, intensity_multiplier, mud_flag(0=water, 1=mud)
        self.params = [(0.62,0.1,0.7,0.7,0.5,0),
                       (0.65,0.1,0.8,0.7,0.5,0),
                       (0.65,0.3,1,0.69,0.5,0),
                       (0.65,0.1,0.7,0.69,0.6,1),
                       (0.65,0.1,0.5,0.68,0.6,1)][severity - 1]

    def _spatter_single(self, np_img):
        # np_img is uint8 [H, W, C] (expects color)
        loc, scale, sigma, threshold, intensity, mud_flag = self.params
        img_float = np_img.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Generate liquid layer noise
        liquid_layer = np.random.normal(size=(h, w), loc=loc, scale=scale)
        liquid_layer = gaussian(liquid_layer, sigma=sigma) # Apply Gaussian blur
        liquid_layer[liquid_layer < threshold] = 0 # Threshold

        if mud_flag == 0: # Water spatter
            liquid_layer_uint8 = (liquid_layer * 255).astype(np.uint8)
            # Canny edge detection
            dist = 255 - cv2.Canny(liquid_layer_uint8, 50, 150)
            # Distance transform
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC) # Threshold distance
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8) # Blur distance map
            dist = cv2.equalizeHist(dist) # Equalize histogram

            # Emboss kernel (approximated from original)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32) / 255. # Blur again, scale to 0-1

            # Create RGBA mask from liquid layer and distance transform
            # Use liquid_layer (0-1) directly instead of multiplying uint8 versions
            mask_intensity = liquid_layer * dist
            # Convert to BGRA (OpenCV convention for color operations often assumes BGR)
            m = cv2.cvtColor(mask_intensity, cv2.COLOR_GRAY2BGRA)
            # Normalize mask intensity
            m_max = np.max(m, axis=(0, 1))
            m_max[m_max == 0] = 1.0 # Avoid division by zero
            m /= m_max
            m *= intensity # Apply overall intensity multiplier

            # Define water color (pale turquoise - BGR format for OpenCV)
            water_color = np.array([238, 238, 175, 255], dtype=np.float32) / 255.0 # BGRA

            # Convert input image to BGRA float
            img_bgra = cv2.cvtColor(img_float, cv2.COLOR_RGB2BGRA)

            # Blend image with spatter mask and color
            spattered_img_bgra = img_bgra + m * water_color
            spattered_img_bgr = cv2.cvtColor(np.clip(spattered_img_bgra, 0, 1), cv2.COLOR_BGRA2BGR)

        else: # Mud spatter
            # Create binary mask based on threshold
            m = np.where(liquid_layer > threshold, 1, 0).astype(np.float32)
            # Apply Gaussian blur to the mask
            m = gaussian(m, sigma=intensity) # 'intensity' used as sigma here in original
            m[m < 0.8] = 0 # Threshold blurred mask

            # Define mud color (dark brown - RGB format)
            mud_color = np.array([63, 42, 20], dtype=np.float32) / 255.0

            # Add channel dimension to mask for broadcasting
            m_rgb = m[..., np.newaxis]

            # Blend image with mud color based on mask
            spattered_img_rgb = img_float * (1 - m_rgb) + mud_color * m_rgb
            spattered_img_bgr = cv2.cvtColor(np.clip(spattered_img_rgb, 0, 1), cv2.COLOR_RGB2BGR) # Convert back to BGR if needed later


        # Convert final BGR float image back to RGB uint8
        final_rgb = cv2.cvtColor(spattered_img_bgr, cv2.COLOR_BGR2RGB)
        return (final_rgb * 255).astype(np.uint8)


    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
             # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._spatter_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class Contrast(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # Contrast factor
        self.c = [.75, .5, .4, .3, 0.15][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        # TF.adjust_contrast expects factor: 0 gives gray image, 1 gives original, >1 enhances.
        # The original formula (x - means) * c + means corresponds to TF contrast factor c.
        return TF.adjust_contrast(x, contrast_factor=self.c)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, contrast_factor={self.c})'

class Brightness(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # Brightness adjustment value (added in HSV's V channel)
        self.c = [.05, .1, .15, .2, .3][self.severity - 1]
        # TF.adjust_brightness uses a factor: 0 gives black image, 1 original, >1 brighter.
        # Need to map the additive factor 'c' to a multiplicative factor.
        # If new_V = old_V + c, factor = new_V / old_V = 1 + c / old_V.
        # This mapping isn't straightforward as it depends on old_V.
        # The original skimage implementation adds 'c' directly to V channel.
        # We'll use TF.adjust_brightness, but the effect might differ slightly.
        # A factor of 1 + c might approximate the effect for mid-range V values.
        self.brightness_factor = 1.0 + self.c

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        # Using TF.adjust_brightness with the calculated factor
        # Note: This might not perfectly match the original HSV manipulation.
        return TF.adjust_brightness(x, brightness_factor=self.brightness_factor)
        # --- Alternative: Mimic original HSV manipulation (CPU-bound) ---
        # device = x.device
        # dtype = x.dtype
        # processed_batch = []
        # for i in range(x.shape[0]):
        #     img_np_uint8 = tensor_to_numpy_uint8(x[i])
        #     img_np_float = img_np_uint8.astype(np.float32) / 255.0
        #     img_hsv = sk.color.rgb2hsv(img_np_float)
        #     img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] + self.c, 0, 1)
        #     img_rgb_back = sk.color.hsv2rgb(img_hsv)
        #     corrupted_np = np.clip(img_rgb_back * 255, 0, 255).astype(np.uint8)
        #     processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
        # return torch.stack(processed_batch).to(device=device, dtype=dtype)
        # --- End Alternative ---


    def __repr__(self):
        # Show both c and the derived factor if using TF.adjust_brightness
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c}, brightness_factor={self.brightness_factor})'


class Saturate(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # Saturation multiplication factor, additive factor
        self.params = [(0.3, 0), (0.1, 0), (1.5, 0), (2, 0.1), (2.5, 0.2)][self.severity - 1]
        # TF.adjust_saturation uses a factor: 0 grayscale, 1 original, >1 enhances.
        # The original formula S' = clip(S * factor + add, 0, 1) is complex to map directly.
        # We will use TF.adjust_saturation with the multiplicative factor.
        # The additive factor is ignored here, which will cause differences.
        self.saturation_factor = self.params[0]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        # Using TF.adjust_saturation with the multiplicative factor.
        # Note: Ignores the additive component from the original code.
        return TF.adjust_saturation(x, saturation_factor=self.saturation_factor)
        # --- Alternative: Mimic original HSV manipulation (CPU-bound) ---
        # device = x.device
        # dtype = x.dtype
        # factor, add = self.params
        # processed_batch = []
        # for i in range(x.shape[0]):
        #     img_np_uint8 = tensor_to_numpy_uint8(x[i])
        #     img_np_float = img_np_uint8.astype(np.float32) / 255.0
        #     img_hsv = sk.color.rgb2hsv(img_np_float)
        #     img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * factor + add, 0, 1)
        #     img_rgb_back = sk.color.hsv2rgb(img_hsv)
        #     corrupted_np = np.clip(img_rgb_back * 255, 0, 255).astype(np.uint8)
        #     processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
        # return torch.stack(processed_batch).to(device=device, dtype=dtype)
        # --- End Alternative ---

    def __repr__(self):
        # Show original params and the factor used if using TF.adjust_saturation
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params}, saturation_factor={self.saturation_factor})'


class JpegCompression(nn.Module):
    """
    Applies JPEG compression artifact.
    Requires PIL/Pillow. Operates on CPU via PIL conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # JPEG quality factor
        self.quality = [80, 65, 58, 50, 40][self.severity - 1]

    def _jpeg_single(self, pil_img):
        # pil_img is a PIL Image object
        output = BytesIO()
        pil_img.save(output, 'JPEG', quality=self.quality)
        output.seek(0)
        return Image.open(output)

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_pil = tensor_to_pil(x[i])
            corrupted_pil = self._jpeg_single(img_pil)
            # Need to ensure the reloaded image has the same mode (RGB/L)
            # If original was L, JPEG might force RGB. Convert back if needed.
            if img_pil.mode == 'L' and corrupted_pil.mode == 'RGB':
                 corrupted_pil = corrupted_pil.convert('L')
            processed_batch.append(pil_to_tensor(corrupted_pil))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, quality={self.quality})'


class Pixelate(nn.Module):
    """
    Applies pixelation effect by downsampling and upsampling.
    Uses torch.nn.functional.interpolate.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # Resizing factor (inverse of pixelation amount)
        self.c = [0.95, 0.9, 0.85, 0.75, 0.65][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        B, C, H, W = x.shape
        # Calculate target size for downsampling
        target_h, target_w = int(H * self.c), int(W * self.c)

        # Ensure target size is at least 1
        target_h = max(1, target_h)
        target_w = max(1, target_w)

        # Downsample using nearest neighbor interpolation (like Image.BOX)
        x_down = F.interpolate(x, size=(target_h, target_w), mode='nearest')

        # Upsample back to original size using nearest neighbor
        x_pixelated = F.interpolate(x_down, size=(H, W), mode='nearest')

        return x_pixelated

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, factor={self.c})'


class ElasticTransform(nn.Module):
    """
    Applies elastic transformation.
    Requires OpenCV, SciPy, Scikit-Image. Operates on CPU.
    """
    def __init__(self, severity=1):
        super().__init__()
        self.severity = severity
        # alpha_scale, sigma, affine_scale_multiplier (derived from original IMSIZE-based params)
        # Assuming IMSIZE=32 for CIFAR-like images as a reference point
        # If your images are different size, these params might need scaling.
        ref_imsize = 32.0
        self.params = [(ref_imsize*0, ref_imsize*0, ref_imsize*0.08),
                       (ref_imsize*0.05, ref_imsize*0.2, ref_imsize*0.07),
                       (ref_imsize*0.08, ref_imsize*0.06, ref_imsize*0.06),
                       (ref_imsize*0.1, ref_imsize*0.04, ref_imsize*0.05),
                       (ref_imsize*0.1, ref_imsize*0.03, ref_imsize*0.03)][severity - 1]

    def _elastic_transform_single(self, np_img):
        # np_img is uint8 [H, W, C]
        alpha_scale, sigma, affine_scale = self.params
        img_float = np_img.astype(np.float32) / 255.0
        shape = img_float.shape
        shape_size = shape[:2] # (H, W)

        # --- Random Affine Transform ---
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # Define 3 points for affine transform
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        # Add random displacement
        # Scale affine displacement by image size relative to reference 32
        # This makes the visual effect more consistent across sizes
        current_imsize = max(shape_size)
        scaled_affine_disp = affine_scale * (current_imsize / 32.0)
        pts2 = pts1 + np.random.uniform(-scaled_affine_disp, scaled_affine_disp, size=pts1.shape).astype(np.float32)

        # Get affine matrix and warp image
        M = cv2.getAffineTransform(pts1, pts2)
        # Use BORDER_REFLECT_101 for pixel extrapolation
        img_affine = cv2.warpAffine(img_float, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)


        # --- Elastic Deformation ---
        # Scale alpha and sigma based on image size relative to reference 32
        scaled_alpha = alpha_scale * (current_imsize / 32.0)
        scaled_sigma = sigma * (current_imsize / 32.0)

        # Generate random displacement fields (dx, dy)
        # Use gaussian filter from skimage.filters
        dx = gaussian(np.random.uniform(-1, 1, size=shape_size),
                      scaled_sigma, mode='reflect', truncate=3) * scaled_alpha
        dy = gaussian(np.random.uniform(-1, 1, size=shape_size),
                      scaled_sigma, mode='reflect', truncate=3) * scaled_alpha

        # Add channel dimension for mapping coordinates
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        # Create coordinate grid
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

        # Calculate new coordinates using displacement fields
        # indices shape: (3, H*W*C)
        indices = (np.reshape(y + dy, (-1, 1)),
                   np.reshape(x + dx, (-1, 1)),
                   np.reshape(z, (-1, 1)))

        # Map coordinates from the affinely transformed image
        # Use map_coordinates from scipy.ndimage.interpolation
        img_elastic = map_coordinates(img_affine, indices, order=1, mode='reflect').reshape(shape)

        return np.clip(img_elastic * 255, 0, 255).astype(np.uint8)


    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        device = x.device
        dtype = x.dtype
        processed_batch = []
        for i in range(x.shape[0]):
            img_np = tensor_to_numpy_uint8(x[i])
             # Ensure 3 channels if grayscale
            if img_np.ndim == 2:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 1:
                 img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            corrupted_np = self._elastic_transform_single(img_np)
            processed_batch.append(numpy_uint8_to_tensor(corrupted_np))

        return torch.stack(processed_batch).to(device=device, dtype=dtype)

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


# --- Dictionary of Augmentations ---
# (Excluding original geometric transforms like rot90, flip, etc.,
#  as these are standard in torchvision.transforms)

corruption_layers = {
    'gaussian_noise': GaussianNoise,
    'shot_noise': ShotNoise,
    'impulse_noise': ImpulseNoise,
    'speckle_noise': SpeckleNoise,
    'gaussian_blur': GaussianBlur,
    'glass_blur': GlassBlur,
    'defocus_blur': DefocusBlur,
    'motion_blur': MotionBlur,
    'zoom_blur': ZoomBlur,
    'fog': Fog,
    'frost': Frost,
    'snow': Snow,
    'spatter': Spatter,
    'contrast': Contrast,
    'brightness': Brightness,
    'saturate': Saturate,
    'jpeg_compression': JpegCompression,
    'pixelate': Pixelate,
    'elastic_transform': ElasticTransform,
}

# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy batch of images (e.g., 4 images, 3 channels, 32x32)
    dummy_images = torch.rand(4, 3, 32, 32) # Range [0.0, 1.0]

    # --- Test a PyTorch-native layer ---
    print("--- Testing GaussianNoise ---")
    gauss_noise_layer = GaussianNoise(severity=3)
    print(gauss_noise_layer)
    try:
        noisy_images = gauss_noise_layer(dummy_images)
        print("Output shape:", noisy_images.shape)
        print("Output range:", noisy_images.min().item(), "-", noisy_images.max().item())
        # Check if output is different from input
        print("Noise applied:", not torch.equal(dummy_images, noisy_images))
    except Exception as e:
        print(f"Error applying GaussianNoise: {e}")

    # --- Test a layer wrapping an external library (CPU-bound) ---
    print("\n--- Testing MotionBlur ---")
    try:
        # Ensure Wand is installed and working for this test
        motion_blur_layer = MotionBlur(severity=2)
        print(motion_blur_layer)
        blurred_images = motion_blur_layer(dummy_images)
        print("Output shape:", blurred_images.shape)
        print("Output range:", blurred_images.min().item(), "-", blurred_images.max().item())
        print("Blur applied:", not torch.equal(dummy_images, blurred_images))
    except ImportError as e:
        print(f"Skipping MotionBlur test: {e}")
    except Exception as e:
        print(f"Error applying MotionBlur: {e}")

    # --- Test another layer (Pixelate) ---
    print("\n--- Testing Pixelate ---")
    pixelate_layer = Pixelate(severity=4)
    print(pixelate_layer)
    try:
        pixelated_images = pixelate_layer(dummy_images)
        print("Output shape:", pixelated_images.shape)
        print("Output range:", pixelated_images.min().item(), "-", pixelated_images.max().item())
        print("Pixelation applied:", not torch.equal(dummy_images, pixelated_images))
    except Exception as e:
        print(f"Error applying Pixelate: {e}")

    # --- Test Frost (requires frost images) ---
    print("\n--- Testing Frost ---")
    # Create dummy frost dir/files if they don't exist for testing
    import os
    if not os.path.exists('frosts'):
        os.makedirs('frosts')
    if not os.path.exists('frosts/frost1.png'):
         # Create a dummy white image
         dummy_frost = Image.new('RGB', (100, 100), color = 'white')
         dummy_frost.save('frosts/frost1.png')
         print("Created dummy frost image for testing.")

    try:
        frost_layer = Frost(severity=3, frost_dir='frosts')
        print(frost_layer)
        frosted_images = frost_layer(dummy_images)
        print("Output shape:", frosted_images.shape)
        print("Output range:", frosted_images.min().item(), "-", frosted_images.max().item())
        print("Frost applied:", not torch.equal(dummy_images, frosted_images))
    except FileNotFoundError as e:
         print(f"Skipping Frost test: {e}")
    except Exception as e:
         print(f"Error applying Frost: {e}")


    # --- Using the dictionary ---
    print("\n--- Accessing layer from dictionary ---")
    corruption_name = 'defocus_blur'
    severity_level = 2
    if corruption_name in corruption_layers:
        LayerClass = corruption_layers[corruption_name]
        layer_instance = LayerClass(severity=severity_level)
        print(f"Created layer: {layer_instance}")
        try:
            output_images = layer_instance(dummy_images)
            print("Output shape:", output_images.shape)
        except Exception as e:
             print(f"Error applying {corruption_name}: {e}")
    else:
        print(f"Corruption '{corruption_name}' not found in layers.")
