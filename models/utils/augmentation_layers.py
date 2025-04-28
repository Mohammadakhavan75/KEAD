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

    # --- Assertions ---
    assert isinstance(img_tensor, torch.Tensor), f"Input must be a torch.Tensor, got {type(img_tensor)}"
    assert img_tensor.ndim == 3, f"Input tensor must have 3 dimensions (C, H, W), got {img_tensor.ndim}"
    # Optional: Check value range if strictly required, can add overhead
    # assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0, "Input tensor values must be in [0, 1]"
    try:
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
    except Exception as e:
        print(f"Error converting tensor to PIL Image: {e}")
        # Depending on desired behavior, you might raise the exception,
        # return None, or a default image. Raising is often preferred.
        raise e


# Helper function to convert PIL Image back to Tensor
def pil_to_tensor(pil_img):
    """Converts a PIL Image to a [C, H, W] tensor (0-1 range)."""
     # --- Assertions ---
    assert isinstance(pil_img, Image.Image), f"Input must be a PIL.Image.Image, got {type(pil_img)}"

    try:
        return TF.to_tensor(pil_img) # Outputs tensor in [0.0, 1.0] range
    except Exception as e:
        print(f"Error converting PIL Image to tensor: {e}")
        raise e


# Helper function to convert tensor to NumPy array (H, W, C) [0-255]
def tensor_to_numpy_uint8(img_tensor):
    """Converts a [C, H, W] tensor (0-1 range) to a NumPy uint8 array [H, W, C] (0-255 range)."""
    # --- Assertions ---
    assert isinstance(img_tensor, torch.Tensor), f"Input must be a torch.Tensor, got {type(img_tensor)}"
    assert img_tensor.ndim == 3, f"Input tensor must have 3 dimensions (C, H, W), got {img_tensor.ndim}"
    # Optional: Check value range
    # assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0, "Input tensor values must be in [0, 1]"

    try:
        img_tensor = img_tensor.cpu().detach()
        img_tensor = torch.clamp(img_tensor * 255, 0, 255).byte()
        # Permute from [C, H, W] to [H, W, C]
        np_img = img_tensor.permute(1, 2, 0).numpy()
        # If grayscale, ensure it's (H, W) or (H, W, 1) as needed by some functions
        # if np_img.shape[2] == 1:
        #     np_img = np_img.squeeze(2)
        return np_img
    except Exception as e:
        print(f"Error converting tensor to NumPy uint8: {e}")
        raise e


# Helper function to convert NumPy array (H, W, C) [0-255] back to Tensor
def numpy_uint8_to_tensor(np_img):
    """Converts a NumPy uint8 array [H, W, C] or [H, W] (0-255 range) to a [C, H, W] tensor (0-1 range)."""
     # --- Assertions ---
    assert isinstance(np_img, np.ndarray), f"Input must be a numpy.ndarray, got {type(np_img)}"
    assert np_img.ndim in [2, 3], f"Input array must have 2 or 3 dimensions, got {np_img.ndim}"

    try:
        if np_img.ndim == 2: # Handle grayscale that might be (H, W)
            np_img = np_img[:, :, np.newaxis]
        # Ensure it's uint8
        if np_img.dtype != np.uint8:
            # Check if values are roughly in the 0-255 range before clipping
            if np_img.min() < 0 or np_img.max() > 255:
                 warnings.warn(f"Input NumPy array has dtype {np_img.dtype} and values outside [0, 255]. Clipping to [0, 255] and converting to uint8.", UserWarning)
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)

        # Permute from [H, W, C] to [C, H, W]
        tensor = torch.from_numpy(np_img.transpose((2, 0, 1)))
        # Convert to float and scale to [0.0, 1.0]
        return tensor.float().div(255.0)
    
    except Exception as e:
        print(f"Error converting NumPy uint8 to tensor: {e}")
        raise e

# --- Original Helper Functions (Needed by some augmentations) ---
# (Copied from the provided script, potentially modified for direct use)

def disk(radius, alias_blur=0.1, dtype=np.float32):
    """Creates a 2D disk kernel."""
     # --- Assertions ---
    assert isinstance(radius, (int, float)) and radius >= 0, f"Radius must be a non-negative number, got {radius}"
    assert isinstance(alias_blur, (int, float)) and alias_blur >= 0, f"alias_blur must be a non-negative number, got {alias_blur}"
    try:
        # Check if dtype is valid
        _ = np.dtype(dtype)
    except TypeError:
        raise TypeError(f"Invalid dtype provided: {dtype}")

    try:
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
            warnings.warn(f"Disk with radius {radius} resulted in a sum of 0. Returning a delta kernel.", UserWarning)
            aliased_disk = np.zeros_like(aliased_disk)
            center = tuple(s // 2 for s in aliased_disk.shape)
            if all(c >= 0 and c < s for c, s in zip(center, aliased_disk.shape)):
                aliased_disk[center] = 1.0
            else:
                 # Fallback if center calculation fails (e.g., 0-size dimension)
                 if aliased_disk.size > 0:
                     aliased_disk.flat[0] = 1.0 # Put 1 at the first element

        # supersample disk to antialias
        # Ensure sigmaX is positive for GaussianBlur
        sigmaX_val = max(1e-6, alias_blur) # Prevent zero or negative sigma
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    except Exception as e:
        print(f"Error creating disk kernel: {e}")
        raise e

def plasma_fractal(mapsize=32, wibbledecay=3):
    """Generate a heightmap using diamond-square algorithm."""
    # Input validation
    if not isinstance(mapsize, int):
        raise TypeError(f"mapsize must be an integer, got {type(mapsize)}")
    if not isinstance(wibbledecay, (int, float)):
        raise TypeError(f"wibbledecay must be a number, got {type(wibbledecay)}")
    if mapsize <= 0:
        raise ValueError(f"mapsize must be positive, got {mapsize}")
    if wibbledecay <= 0:
        raise ValueError(f"wibbledecay must be positive, got {wibbledecay}")
    
    # Check if mapsize is a power of 2
    if not (mapsize & (mapsize - 1) == 0):
        raise ValueError(f"mapsize must be a power of 2, got {mapsize}")

    try:
        maparray = np.empty((mapsize, mapsize), dtype=np.float32)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            try:
                return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)
            except Exception as e:
                raise RuntimeError(f"Error in wibbledmean calculation: {str(e)}")

        def fillsquares():
            try:
                cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
                squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
                squareaccum += np.roll(squareaccum, shift=-1, axis=1)
                maparray[stepsize // 2:mapsize:stepsize,
                stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)
            except Exception as e:
                raise RuntimeError(f"Error in fillsquares: {str(e)}")

        def filldiamonds():
            try:
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
            except Exception as e:
                raise RuntimeError(f"Error in filldiamonds: {str(e)}")

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        map_max = maparray.max()
        
        if map_max > 0:
            return maparray / map_max
        else:
            return maparray # Return the constant array (likely all zeros)

    except Exception as e:
        raise RuntimeError(f"Error generating plasma fractal: {str(e)}")

def clipped_zoom(img, zoom_factor):
    """Zooms into the center of an image and crops back to original size."""
    # Input validation
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a numpy array, got {type(img)}")
    if not isinstance(zoom_factor, (int, float)):
        raise TypeError(f"zoom_factor must be a number, got {type(zoom_factor)}")
    if zoom_factor <= 0:
        raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")
    if img.ndim != 3:
        raise ValueError(f"img must have 3 dimensions, got shape {img.shape}")

    try:
        h = img.shape[0]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / zoom_factor))

        if ch > h: # Handle cases where zoom_factor < 1, prevent negative slicing
            ch = h
            top = 0
        else:
            top = (h - ch) // 2

        # Validate slicing indices
        if top < 0 or top + ch > h:
            raise ValueError(f"Invalid crop coordinates: top={top}, ch={ch}, h={h}")

        try:
            img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
        except Exception as e:
            raise RuntimeError(f"Error during scipy zoom operation: {str(e)}")

        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2

        # Ensure trimming indices are valid
        if trim_top < 0:
            # This can happen if the zoomed image is smaller than h due to rounding/zoom_factor < 1
            # Pad the image instead of trimming
            pad_before = abs(trim_top)
            pad_after = h - img.shape[0] - pad_before
            try:
                img = np.pad(img, ((pad_before, pad_after), (pad_before, pad_after), (0, 0)), mode='constant')
                trim_top = 0 # Reset trim_top as padding handled it
            except Exception as e:
                raise RuntimeError(f"Error during image padding: {str(e)}")

        # Validate final slicing indices
        if trim_top < 0 or trim_top + h > img.shape[0]:
            raise ValueError(f"Invalid trimming coordinates: trim_top={trim_top}, h={h}, shape={img.shape}")

        return img[trim_top:trim_top + h, trim_top:trim_top + h]

    except Exception as e:
        raise RuntimeError(f"Error in clipped_zoom: {str(e)}")
# --- Augmentation Layers ---



# --- Geometric Augmentations as Layers ---

class Rotate90(nn.Module):
    """Applies a 90-degree clockwise rotation (equivalent to 270 degrees counter-clockwise)."""
    def __init__(self):
        super().__init__()
        # torchvision.transforms.functional.rotate expects counter-clockwise degrees
        self.angle = -90 # Corresponds to 270 degrees counter-clockwise

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        # Value range check might be less critical for geometric transforms but good practice
        # if x.min() < 0.0 or x.max() > 1.0:
        #     warnings.warn(f"Input tensor values outside range [0.0, 1.0], got [{x.min():.3f}, {x.max():.3f}]")

        try:
            # TF.rotate works on batches directly
            return TF.rotate(x, self.angle)
        except Exception as e:
            raise RuntimeError(f"Error applying Rotate90: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(angle={self.angle})'


class Rotate270(nn.Module):
    """Applies a 270-degree clockwise rotation (equivalent to 90 degrees counter-clockwise)."""
    def __init__(self):
        super().__init__()
        # torchvision.transforms.functional.rotate expects counter-clockwise degrees
        self.angle = 90 # Corresponds to 90 degrees counter-clockwise

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")

        try:
            # TF.rotate works on batches directly
            return TF.rotate(x, self.angle)
        except Exception as e:
            raise RuntimeError(f"Error applying Rotate270: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(angle={self.angle})'


class HorizontalFlip(nn.Module):
    """Applies a horizontal flip with probability 1.0."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")

        try:
            # TF.hflip works on batches directly
            return TF.hflip(x)
        except Exception as e:
            raise RuntimeError(f"Error applying HorizontalFlip: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomCropResize(nn.Module):
    """Applies a random crop (scaling factor 0.75) and resizes back to original size."""
    def __init__(self, scale_factor=0.75):
        super().__init__()
        if not isinstance(scale_factor, (float, int)) or not 0 < scale_factor <= 1:
             raise ValueError(f"scale_factor must be a number between 0 and 1, got {scale_factor}")
        self.scale_factor = scale_factor

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")

        try:
            b, c, h, w = x.shape
            original_size = (h, w)
            crop_h = int(h * self.scale_factor)
            crop_w = int(w * self.scale_factor)
            if crop_h == 0 or crop_w == 0:
                warnings.warn(f"Calculated crop size is zero ({crop_h}x{crop_w}) for input size {h}x{w} and scale {self.scale_factor}. Skipping crop.", UserWarning)
                return x

            # Get random crop parameters for the batch
            # Note: This applies the *same* crop to all images in the batch.
            # If you need a *different* random crop per image, you'll need to loop.
            # top = torch.randint(0, h - crop_h + 1, size=(1,), device=x.device).item()
            # left = torch.randint(0, w - crop_w + 1, size=(1,), device=x.device).item()
            # cropped = TF.crop(x, top, left, crop_h, crop_w)

            # Alternative: Apply different crop per image using a loop (slower)
            cropped_resized_batch = []
            for img in x: # Iterate through batch dimension
                 top = torch.randint(0, h - crop_h + 1, size=(1,)).item()
                 left = torch.randint(0, w - crop_w + 1, size=(1,)).item()
                 cropped_img = TF.crop(img, top, left, crop_h, crop_w)
                 resized_img = TF.resize(cropped_img, original_size, interpolation=TF.InterpolationMode.BILINEAR)
                 cropped_resized_batch.append(resized_img)
            
            return torch.stack(cropped_resized_batch)

            # Resize back to original size
            # resized = TF.resize(cropped, original_size, interpolation=TF.InterpolationMode.BILINEAR)
            # return resized

        except Exception as e:
            raise RuntimeError(f"Error applying RandomCropResize: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


class ColorJitterLayer(nn.Module):
    """Applies ColorJitter transformation."""
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        super().__init__()
        # Store parameters for repr
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        # Initialize the transform
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
             warnings.warn(f"Input tensor values outside range [0.0, 1.0], got [{x.min():.3f}, {x.max():.3f}]. ColorJitter might behave unexpectedly.", UserWarning)

        try:
            # Apply jitter. T.ColorJitter handles batches.
            return self.jitter(x)
        except Exception as e:
            raise RuntimeError(f"Error applying ColorJitterLayer: {str(e)}")

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"brightness={self.brightness}, contrast={self.contrast}, "
                f"saturation={self.saturation}, hue={self.hue})")


class GaussianNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        self.c = [0.04, 0.06, .08, .09, .10][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            # Generate and apply noise
            noise = torch.randn_like(x) * self.c
            x_noisy = x + noise
            return torch.clamp(x_noisy, 0.0, 1.0)
            
        except RuntimeError as e:
            raise RuntimeError(f"Error applying Gaussian noise: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in GaussianNoise forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'


class ShotNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # Original parameter 'c' represents the scaling factor applied *before* Poisson.
        # In torch.poisson, the input rate is lambda. rate = input * c.
        # So, the output is Poisson(input * c). We then divide by c.
        self.c = [500, 250, 100, 75, 50][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            # Calculate lambda = x * c. Ensure lambda >= 0.
            rate = torch.clamp(x * self.c, min=0)
            # Apply Poisson noise. The output of torch.poisson is float.
            x_noisy = torch.poisson(rate) / self.c
            return torch.clamp(x_noisy, 0.0, 1.0)
            
        except RuntimeError as e:
            raise RuntimeError(f"Error applying shot noise: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in ShotNoise forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'


class ImpulseNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        self.c = [.01, .02, .03, .05, .07][self.severity - 1] # proportion of pixels to replace

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
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
                try:
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
                    
                except RuntimeError as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")
                except Exception as e:
                    raise Exception(f"Unexpected error processing image {i} in batch: {str(e)}")

            x_noisy = torch.stack(noisy_batch)
            return torch.clamp(x_noisy, 0.0, 1.0) # Clamp shouldn't be necessary here
            
        except RuntimeError as e:
            raise RuntimeError(f"Error applying impulse noise: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in ImpulseNoise forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'


class SpeckleNoise(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        self.c = [.06, .1, .12, .16, .2][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            # Generate and apply multiplicative noise
            noise = torch.randn_like(x) * self.c
            x_noisy = x + x * noise
            return torch.clamp(x_noisy, 0.0, 1.0)
            
        except RuntimeError as e:
            raise RuntimeError(f"Error applying speckle noise: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in SpeckleNoise forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, c={self.c})'


class GaussianBlur(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity input
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # Map severity to sigma. Kernel size is determined automatically in TF.gaussian_blur
        self.sigma = [0.4, 0.6, 0.7, 0.8, 1.0][self.severity - 1]
        # Kernel size needs to be odd
        # A common heuristic: kernel_size ≈ 6*sigma + 1
        try:
            k = int(round(6 * self.sigma + 1))
            if k < 1:
                raise ValueError(f"Calculated kernel size {k} is invalid (must be >= 1)")
            self.kernel_size = k if k % 2 == 1 else k + 1
        except Exception as e:
            raise RuntimeError(f"Error calculating kernel size: {str(e)}")

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            # TF.gaussian_blur expects a list of sigmas or a float sigma
            # It applies the same blur to all images in the batch
            return TF.gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
        except RuntimeError as e:
            raise RuntimeError(f"Error applying Gaussian blur: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in GaussianBlur forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, sigma={self.sigma}, kernel_size={self.kernel_size})'


class GlassBlur(nn.Module):
    """
    Wraps the original glass_blur function.
    Requires scikit-image. Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # sigma, max_delta, iterations
        self.params = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]

    def _glass_blur_single(self, np_img):
        # Input validation
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise ValueError(f"Input array must be uint8, got {np_img.dtype}")
        if np_img.ndim not in [2, 3]:
            raise ValueError(f"Input array must have 2 or 3 dimensions, got {np_img.ndim}")

        try:
            sigma, max_delta, iterations = self.params

            # Apply initial Gaussian blur
            img_blurred = gaussian(np_img / 255., sigma=sigma, channel_axis=-1) * 255
            img_blurred = np.uint8(img_blurred) # Convert back to uint8 for shuffling

            # Locally shuffle pixels
            h, w = img_blurred.shape[:2]
            if h <= 2*max_delta or w <= 2*max_delta:
                raise ValueError(f"Image dimensions ({h},{w}) too small for max_delta={max_delta}")
                
            img_shuffled = img_blurred.copy() # Work on a copy

            try:
                for _ in range(iterations):
                    for y in range(h - max_delta, max_delta -1, -1):
                        for x in range(w - max_delta, max_delta -1, -1):
                            # Generate random displacement
                            dx, dy = np.random.randint(-max_delta, max_delta + 1, size=(2,))
                            # Calculate neighbor coordinates
                            y_prime = np.clip(y + dy, 0, h - 1)
                            x_prime = np.clip(x + dx, 0, w - 1)

                            # Swap pixels
                            temp = img_shuffled[y, x].copy()
                            img_shuffled[y, x] = img_shuffled[y_prime, x_prime]
                            img_shuffled[y_prime, x_prime] = temp
            except Exception as e:
                raise RuntimeError(f"Error during pixel shuffling: {str(e)}")

            # Apply final Gaussian blur
            img_final_blur = gaussian(img_shuffled / 255., sigma=sigma, channel_axis=-1)
            return np.clip(img_final_blur * 255, 0, 255).astype(np.uint8)

        except Exception as e:
            raise RuntimeError(f"Error in glass blur processing: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")

        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    # Ensure 3 channels if grayscale
                    if img_np.ndim == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    elif img_np.shape[2] == 1:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                    corrupted_np = self._glass_blur_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error in GlassBlur forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class DefocusBlur(nn.Module):
    """
    Applies defocus blur using a disk kernel.
    Requires OpenCV (cv2). Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # radius, alias_blur
        self.params = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1)][severity - 1]

    def _defocus_blur_single(self, np_img):
        # Input validation
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise ValueError(f"Input array must be uint8, got {np_img.dtype}")
        if np_img.ndim not in [2, 3]:
            raise ValueError(f"Input array must have 2 or 3 dimensions, got {np_img.ndim}")
            
        try:
            radius, alias_blur = self.params
            kernel = disk(radius=radius, alias_blur=alias_blur)

            # Apply filter2D for each channel
            channels = []
            img_float = np_img.astype(np.float32) / 255.0 # Work with float for filtering
            
            # Handle potential dimension issues
            if img_float.ndim == 2:
                img_float = img_float[..., np.newaxis]
                
            for d in range(img_float.shape[2]): # Iterate through channels
                try:
                    filtered = cv2.filter2D(img_float[:, :, d], -1, kernel)
                    channels.append(filtered)
                except cv2.error as e:
                    raise RuntimeError(f"OpenCV filter2D failed on channel {d}: {str(e)}")

            # Stack channels back and handle potential dimension issues if grayscale
            if len(channels) > 1:
                blurred_img = np.array(channels).transpose((1, 2, 0)) # C x H x W -> H x W x C
            elif len(channels) == 1:
                blurred_img = channels[0][:, :, np.newaxis] # Add channel dim back
            else: # Should not happen if input has channels
                raise ValueError("No channels processed successfully")

            return np.clip(blurred_img * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            raise RuntimeError(f"Error in defocus blur processing: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    # Ensure 3 channels if grayscale
                    if img_np.ndim == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    elif img_np.shape[2] == 1:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                    corrupted_np = self._defocus_blur_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error in DefocusBlur forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class MotionBlur(nn.Module):
    """
    Applies motion blur using the Wand library.
    Requires Wand and ImageMagick. Operates on CPU via PIL conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Check if Wand library is available
        if not _wand_available:
            raise ImportError("Wand library is required for MotionBlur but not found.")
            
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # radius, sigma
        self.params = [(6,1), (6,1.5), (6,2), (8,2), (9,2.5)][severity - 1]

    def _motion_blur_single(self, pil_img):
        """Apply motion blur to a single PIL image."""
        # Input validation
        if not isinstance(pil_img, Image.Image):
            raise TypeError(f"Input must be a PIL.Image.Image, got {type(pil_img)}")
            
        radius, sigma = self.params
        angle = np.random.uniform(-45, 45) # Random angle per image

        try:
            # Convert PIL image to WandImage
            output = BytesIO()
            # Ensure saving in a format Wand understands well, like PNG
            pil_img.save(output, format='PNG')
            output.seek(0) # Reset buffer position
            
            try:
                wand_img = MotionImage(blob=output.getvalue())
            except Exception as e:
                raise RuntimeError(f"Failed to create WandImage: {str(e)}")

            # Apply motion blur
            try:
                wand_img.motion_blur(radius=radius, sigma=sigma, angle=angle)
            except Exception as e:
                raise RuntimeError(f"Failed to apply motion blur: {str(e)}")

            # Convert back to NumPy array (Wand uses BGR by default)
            try:
                blob = wand_img.make_blob(format='BGR') # Get raw bytes in BGR
                np_img_bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if np_img_bgr is None:
                    raise ValueError("Failed to decode image blob")
            except Exception as e:
                raise RuntimeError(f"Failed to convert WandImage to NumPy array: {str(e)}")

            # Convert BGR to RGB
            try:
                np_img_rgb = cv2.cvtColor(np_img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise RuntimeError(f"Failed to convert BGR to RGB: {str(e)}")

            return np.clip(np_img_rgb, 0, 255).astype(np.uint8)

        except Exception as e:
            warnings.warn(f"Error during motion blur, returning original image: {str(e)}")
            # Return original image data as numpy array in case of error
            return np.array(pil_img)

    def forward(self, x):
        """
        Apply motion blur to a batch of images.
        Args:
            x: Input tensor of shape [B, C, H, W] in range [0.0, 1.0]
        Returns:
            Blurred tensor of same shape and range
        """
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")

        device = x.device
        dtype = x.dtype
        processed_batch = []
        
        try:
            for i in range(x.shape[0]):
                try:
                    img_pil = tensor_to_pil(x[i])
                    corrupted_np = self._motion_blur_single(img_pil)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                except Exception as e:
                    warnings.warn(f"Error processing image {i}, using original: {str(e)}")
                    processed_batch.append(x[i])

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Failed to process batch: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class ZoomBlur(nn.Module):
    """
    Applies zoom blur by averaging multiple zoomed versions.
    Requires SciPy (for zoom). Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # zoom factors
        self.zoom_factors = [np.arange(1, 1.06, 0.01),
                             np.arange(1, 1.11, 0.01),
                             np.arange(1, 1.16, 0.01),
                             np.arange(1, 1.21, 0.01),
                             np.arange(1, 1.26, 0.01)][severity - 1]

    def _zoom_blur_single(self, np_img):
        # Input validation
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise ValueError(f"Input array must be uint8, got {np_img.dtype}")
        if np_img.ndim != 3:
            raise ValueError(f"Input array must have 3 dimensions, got shape {np_img.shape}")
            
        try:
            img_float = np_img.astype(np.float32) / 255.0
            out = np.zeros_like(img_float)
            zoom_count = 0  # Track successful zooms

            for zoom_factor in self.zoom_factors:
                try:
                    zoomed = clipped_zoom(img_float, zoom_factor)
                    # Ensure shapes match before adding
                    if zoomed.shape == out.shape:
                        out += zoomed
                        zoom_count += 1
                    else:
                        warnings.warn(f"Shape mismatch in ZoomBlur (original: {out.shape}, zoomed: {zoomed.shape}). Skipping zoom factor {zoom_factor}.")
                except Exception as e:
                    warnings.warn(f"Error applying zoom factor {zoom_factor}: {str(e)}")
                    continue

            # Only average if we have at least one successful zoom
            if zoom_count > 0:
                final_img = (img_float + out) / (zoom_count + 1)
            else:
                warnings.warn("No successful zooms applied, returning original image")
                final_img = img_float

            return np.clip(final_img * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            raise RuntimeError(f"Error in zoom blur processing: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    # Ensure 3 channels if grayscale
                    if img_np.ndim == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    elif img_np.shape[2] == 1:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                    corrupted_np = self._zoom_blur_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error in ZoomBlur forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity})'


class Fog(nn.Module):
    """
    Adds fog effect using plasma fractal noise.
    Requires NumPy. Operates on CPU via NumPy conversion.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # fog_amount, wibbledecay
        self.params = [(.2,3), (.5,3), (0.75,2.5), (1,2), (1.5,1.75)][severity - 1]

    def _fog_single(self, np_img):
        # Input validation
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise ValueError(f"Input array must be uint8, got {np_img.dtype}")
        if np_img.ndim not in [2, 3]:
            raise ValueError(f"Input array must have 2 or 3 dimensions, got {np_img.ndim}")

        try:
            fog_amount, wibbledecay = self.params
            img_float = np_img.astype(np.float32) / 255.0
            h, w = img_float.shape[:2]

            # Determine appropriate mapsize (power of 2 >= max(h, w))
            mapsize = 2**int(np.ceil(np.log2(max(h, w))))

            # Generate plasma fractal noise
            try:
                plasma = plasma_fractal(mapsize=mapsize, wibbledecay=wibbledecay)
            except Exception as e:
                raise RuntimeError(f"Error generating plasma fractal: {str(e)}")

            # Crop plasma to image size and add channel dimension
            try:
                plasma_cropped = plasma[:h, :w][..., np.newaxis]
            except Exception as e:
                raise RuntimeError(f"Error cropping plasma: {str(e)}")

            # Add fog
            foggy_img = img_float + fog_amount * plasma_cropped

            # Rescale to keep max value consistent
            max_val = img_float.max()
            if max_val + fog_amount > 1e-6:
                foggy_img = foggy_img * max_val / (max_val + fog_amount)

            return np.clip(foggy_img * 255, 0, 255).astype(np.uint8)

        except Exception as e:
            raise RuntimeError(f"Error applying fog effect: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")

        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    
                    # Ensure 3 channels if grayscale
                    try:
                        if img_np.ndim == 2:
                            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                        elif img_np.shape[2] == 1:
                            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    except Exception as e:
                        raise RuntimeError(f"Error converting to RGB: {str(e)}")

                    corrupted_np = self._fog_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error in fog forward pass: {str(e)}")

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
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        # Validate frost_dir
        if not isinstance(frost_dir, str):
            raise TypeError(f"frost_dir must be a string, got {type(frost_dir)}")
            
        self.severity = severity
        self.frost_dir = frost_dir
        # image_weight, frost_weight
        self.params = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.75, 0.45)][severity - 1]
        self.frost_filenames = [
            'frost1.png', 'frost2.png', 'frost3.png',
            'frost4.jpg', 'frost5.jpg', 'frost6.jpg'
        ]

    def _load_random_frost_img(self, target_h, target_w):
        """Load and process a random frost image to match target dimensions."""
        # Validate input dimensions
        if not isinstance(target_h, int) or not isinstance(target_w, int):
            raise TypeError(f"Target dimensions must be integers, got h:{type(target_h)}, w:{type(target_w)}")
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Target dimensions must be positive, got h:{target_h}, w:{target_w}")
            
        import os
        try:
            # Validate frost files list
            if not self.frost_filenames:
                raise ValueError("No frost filenames provided")
                
            idx = np.random.randint(len(self.frost_filenames))
            frost_path = os.path.join(self.frost_dir, self.frost_filenames[idx])
            
            # Check if frost directory exists
            if not os.path.exists(self.frost_dir):
                raise FileNotFoundError(f"Frost directory not found: {self.frost_dir}")
                
            # Load frost image
            frost = cv2.imread(frost_path)
            if frost is None:
                raise FileNotFoundError(f"Could not load frost image: {frost_path}")

            # Resize frost image
            try:
                frost_resized = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
            except Exception as e:
                raise RuntimeError(f"Failed to resize frost image: {str(e)}")

            # Handle cropping
            fh, fw = frost_resized.shape[:2]
            if fh < target_h or fw < target_w:
                try:
                    frost_resized = cv2.resize(frost_resized, (target_w, target_h))
                    x_start, y_start = 0, 0
                except Exception as e:
                    raise RuntimeError(f"Failed to resize small frost image: {str(e)}")
            else:
                try:
                    x_start = np.random.randint(0, fw - target_w + 1)
                    y_start = np.random.randint(0, fh - target_h + 1)
                except Exception as e:
                    raise RuntimeError(f"Failed to calculate crop coordinates: {str(e)}")

            try:
                frost_cropped = frost_resized[y_start:y_start + target_h, x_start:x_start + target_w]
                return cv2.cvtColor(frost_cropped, cv2.COLOR_BGR2RGB)
            except Exception as e:
                raise RuntimeError(f"Failed to crop or convert frost image: {str(e)}")

        except Exception as e:
            warnings.warn(f"Could not load or process frost image: {e}. Returning None.")
            return None

    def _frost_single(self, np_img):
        """Apply frost effect to a single image."""
        # Validate input image
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise TypeError(f"Input must be uint8, got {np_img.dtype}")
        if np_img.ndim != 3:
            raise ValueError(f"Input must have 3 dimensions, got shape {np_img.shape}")
            
        img_weight, frost_weight = self.params
        h, w = np_img.shape[:2]

        try:
            frost_overlay = self._load_random_frost_img(h, w)

            if frost_overlay is not None:
                # Validate frost overlay dimensions
                if frost_overlay.shape != np_img.shape:
                    raise ValueError(f"Frost overlay shape {frost_overlay.shape} does not match input shape {np_img.shape}")
                    
                # Blend images
                try:
                    frosted_img = img_weight * np_img + frost_weight * frost_overlay
                    return np.clip(frosted_img, 0, 255).astype(np.uint8)
                except Exception as e:
                    raise RuntimeError(f"Failed to blend images: {str(e)}")
            else:
                return np_img

        except Exception as e:
            warnings.warn(f"Error in frost effect application: {e}. Returning original image.")
            return np_img

    def forward(self, x):
        """Apply frost effect to a batch of images."""
        # Validate input tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        device = x.device
        dtype = x.dtype
        processed_batch = []
        
        try:
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    
                    # Handle grayscale conversion
                    if img_np.ndim == 2 or img_np.shape[2] == 1:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                    corrupted_np = self._frost_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                    
                except Exception as e:
                    warnings.warn(f"Error processing image {i} in batch: {e}. Skipping.")
                    processed_batch.append(x[i])  # Use original image on failure

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Failed to process batch: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class Snow(nn.Module):
    """
    Adds snow effect using noise, zoom, and motion blur.
    Requires SciPy, PIL, OpenCV, Wand. Operates on CPU.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
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
        # Input validation
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise ValueError(f"Input array must be uint8, got {np_img.dtype}")
        if np_img.ndim != 3:
            raise ValueError(f"Input array must have 3 dimensions, got shape {np_img.shape}")
            
        try:
            loc, scale, zoom, threshold, mb_radius, mb_sigma, blend_factor = self.params
            img_float = np_img.astype(np.float32) / 255.0
            h, w = img_float.shape[:2]

            # Generate monochrome noise layer
            snow_layer = np.random.normal(size=(h, w), loc=loc, scale=scale)

            # Zoom noise layer
            try:
                snow_layer_zoomed = clipped_zoom(snow_layer[..., np.newaxis], zoom)
            except Exception as e:
                raise RuntimeError(f"Error in clipped_zoom: {str(e)}")

            # Threshold the zoomed noise
            snow_layer_zoomed[snow_layer_zoomed < threshold] = 0

            # Convert to PIL Image
            try:
                snow_pil = Image.fromarray(
                    (np.clip(snow_layer_zoomed.squeeze(), 0, 1) * 255).astype(np.uint8),
                    mode='L'
                )
            except Exception as e:
                raise RuntimeError(f"Error converting to PIL Image: {str(e)}")

            # Apply motion blur using Wand
            try:
                output = BytesIO()
                snow_pil.save(output, format='PNG')
                output.seek(0)
                snow_wand = MotionImage(blob=output.getvalue())
                angle = np.random.uniform(-135, -45)
                snow_wand.motion_blur(radius=mb_radius, sigma=mb_sigma, angle=angle)

                blob = snow_wand.make_blob(format='GRAY')
                snow_blurred_np = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
                if snow_blurred_np is None:
                    raise RuntimeError("Failed to decode image from Wand blob")

                snow_final = (snow_blurred_np / 255.0)[..., np.newaxis]

            except Exception as e:
                warnings.warn(f"Wand motion blur failed in Snow: {e}. Using unblurred snow layer.")
                snow_final = np.clip(snow_layer_zoomed, 0, 1)

            # Blend snow with original image
            try:
                img_gray_blend = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5
                img_blend_base = np.maximum(img_float, img_gray_blend)
                img_blended = blend_factor * img_float + (1 - blend_factor) * img_blend_base

                snowy_img = img_blended + snow_final + np.rot90(snow_final, k=2)
                return np.clip(snowy_img * 255, 0, 255).astype(np.uint8)
                
            except Exception as e:
                raise RuntimeError(f"Error in blending operation: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Error in snow effect generation: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    
                    # Ensure 3 channels if grayscale
                    if img_np.ndim == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    elif img_np.shape[2] == 1:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                    corrupted_np = self._snow_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error in Snow forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class Spatter(nn.Module):
    """
    Adds spatter effect (water or mud).
    Requires Scikit-Image, OpenCV. Operates on CPU.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # loc, scale, sigma, threshold, intensity_multiplier, mud_flag(0=water, 1=mud)
        self.params = [(0.62,0.1,0.7,0.7,0.5,0),
                       (0.65,0.1,0.8,0.7,0.5,0),
                       (0.65,0.3,1,0.69,0.5,0),
                       (0.65,0.1,0.7,0.69,0.6,1),
                       (0.65,0.1,0.5,0.68,0.6,1)][severity - 1]

    def _spatter_single(self, np_img):
        # Input validation
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise ValueError(f"Input array must be uint8, got {np_img.dtype}")
        if np_img.ndim != 3:
            raise ValueError(f"Input array must have 3 dimensions, got shape {np_img.shape}")
            
        try:
            loc, scale, sigma, threshold, intensity, mud_flag = self.params
            img_float = np_img.astype(np.float32) / 255.0
            h, w = img_float.shape[:2]

            # Generate liquid layer noise
            liquid_layer = np.random.normal(size=(h, w), loc=loc, scale=scale)
            liquid_layer = gaussian(liquid_layer, sigma=sigma)
            liquid_layer[liquid_layer < threshold] = 0

            if mud_flag == 0: # Water spatter
                try:
                    liquid_layer_uint8 = (liquid_layer * 255).astype(np.uint8)
                    dist = 255 - cv2.Canny(liquid_layer_uint8, 50, 150)
                    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
                    _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
                    dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
                    dist = cv2.equalizeHist(dist)

                    ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
                    dist = cv2.filter2D(dist, cv2.CV_8U, ker)
                    dist = cv2.blur(dist, (3, 3)).astype(np.float32) / 255.

                    mask_intensity = liquid_layer * dist
                    m = cv2.cvtColor(mask_intensity, cv2.COLOR_GRAY2BGRA)
                    m_max = np.max(m, axis=(0, 1))
                    m_max[m_max == 0] = 1.0
                    m /= m_max
                    m *= intensity

                    water_color = np.array([238, 238, 175, 255], dtype=np.float32) / 255.0
                    img_bgra = cv2.cvtColor(img_float, cv2.COLOR_RGB2BGRA)
                    spattered_img_bgra = img_bgra + m * water_color
                    spattered_img_bgr = cv2.cvtColor(np.clip(spattered_img_bgra, 0, 1), cv2.COLOR_BGRA2BGR)
                    
                except cv2.error as e:
                    raise RuntimeError(f"OpenCV operation failed in water spatter: {str(e)}")
                except Exception as e:
                    raise RuntimeError(f"Error applying water spatter: {str(e)}")

            else: # Mud spatter
                try:
                    m = np.where(liquid_layer > threshold, 1, 0).astype(np.float32)
                    m = gaussian(m, sigma=intensity)
                    m[m < 0.8] = 0

                    mud_color = np.array([63, 42, 20], dtype=np.float32) / 255.0
                    m_rgb = m[..., np.newaxis]
                    spattered_img_rgb = img_float * (1 - m_rgb) + mud_color * m_rgb
                    spattered_img_bgr = cv2.cvtColor(np.clip(spattered_img_rgb, 0, 1), cv2.COLOR_RGB2BGR)
                    
                except cv2.error as e:
                    raise RuntimeError(f"OpenCV operation failed in mud spatter: {str(e)}")
                except Exception as e:
                    raise RuntimeError(f"Error applying mud spatter: {str(e)}")

            final_rgb = cv2.cvtColor(spattered_img_bgr, cv2.COLOR_BGR2RGB)
            return (final_rgb * 255).astype(np.uint8)
            
        except Exception as e:
            raise RuntimeError(f"Error in _spatter_single: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    # Ensure 3 channels if grayscale
                    if img_np.ndim == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    elif img_np.shape[2] == 1:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                    corrupted_np = self._spatter_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'


class Contrast(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # Contrast factor
        self.c = [.75, .5, .4, .3, 0.15][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        # TF.adjust_contrast expects factor: 0 gives gray image, 1 gives original, >1 enhances.
        # The original formula (x - means) * c + means corresponds to TF contrast factor c.
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            return TF.adjust_contrast(x, contrast_factor=self.c)
        except RuntimeError as e:
            raise RuntimeError(f"Error adjusting contrast: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in contrast adjustment: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, contrast_factor={self.c})'


class Brightness(nn.Module):
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
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
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            # Using TF.adjust_brightness with the calculated factor
            # Note: This might not perfectly match the original HSV manipulation.
            return TF.adjust_brightness(x, brightness_factor=self.brightness_factor)
            
        except RuntimeError as e:
            raise RuntimeError(f"Error applying brightness adjustment: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in Brightness forward pass: {str(e)}")
        
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
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
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
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            # Using TF.adjust_saturation with the multiplicative factor.
            # Note: Ignores the additive component from the original code.
            return TF.adjust_saturation(x, saturation_factor=self.saturation_factor)
            
        except RuntimeError as e:
            raise RuntimeError(f"Error applying saturation adjustment: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error in Saturate forward pass: {str(e)}")
        
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
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # JPEG quality factor
        self.quality = [80, 65, 58, 50, 40][self.severity - 1]

    def _jpeg_single(self, pil_img):
        # Input validation
        if not isinstance(pil_img, Image.Image):
            raise TypeError(f"Input must be a PIL.Image.Image, got {type(pil_img)}")
            
        try:
            output = BytesIO()
            pil_img.save(output, 'JPEG', quality=self.quality)
            output.seek(0)
            return Image.open(output)
        except Exception as e:
            raise RuntimeError(f"Error applying JPEG compression: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    # Convert tensor to PIL
                    img_pil = tensor_to_pil(x[i])
                    
                    # Apply JPEG compression
                    corrupted_pil = self._jpeg_single(img_pil)
                    
                    # Handle mode conversion if needed
                    if img_pil.mode == 'L' and corrupted_pil.mode == 'RGB':
                        try:
                            corrupted_pil = corrupted_pil.convert('L')
                        except Exception as e:
                            raise RuntimeError(f"Error converting image mode: {str(e)}")
                    
                    # Convert back to tensor
                    processed_batch.append(pil_to_tensor(corrupted_pil))
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")
            
            # Stack batch and restore device/dtype
            try:
                return torch.stack(processed_batch).to(device=device, dtype=dtype)
            except Exception as e:
                raise RuntimeError(f"Error stacking processed batch: {str(e)}")
                
        except Exception as e:
            raise RuntimeError(f"Error in JPEG compression forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, quality={self.quality})'


class Pixelate(nn.Module):
    """
    Applies pixelation effect by downsampling and upsampling.
    Uses torch.nn.functional.interpolate.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity level
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
        self.severity = severity
        # Resizing factor (inverse of pixelation amount)
        self.c = [0.95, 0.9, 0.85, 0.75, 0.65][self.severity - 1]

    def forward(self, x):
        # x is assumed to be a batch tensor [B, C, H, W] in range [0.0, 1.0]
        
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            B, C, H, W = x.shape
            
            # Validate input dimensions
            if H < 1 or W < 1:
                raise ValueError(f"Input height and width must be positive, got H={H}, W={W}")
            
            # Calculate target size for downsampling
            target_h, target_w = int(H * self.c), int(W * self.c)

            # Ensure target size is at least 1
            target_h = max(1, target_h)
            target_w = max(1, target_w)

            try:
                # Downsample using nearest neighbor interpolation (like Image.BOX)
                x_down = F.interpolate(x, size=(target_h, target_w), mode='nearest')

                # Upsample back to original size using nearest neighbor
                x_pixelated = F.interpolate(x_down, size=(H, W), mode='nearest')

                return x_pixelated
                
            except RuntimeError as e:
                raise RuntimeError(f"Error during interpolation: {str(e)}")
                
        except Exception as e:
            raise Exception(f"Unexpected error in Pixelate forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, factor={self.c})'


class ElasticTransform(nn.Module):
    """
    Applies elastic transformation.
    Requires OpenCV, SciPy, Scikit-Image. Operates on CPU.
    """
    def __init__(self, severity=1):
        super().__init__()
        # Validate severity
        if not isinstance(severity, int):
            raise TypeError(f"Severity must be an integer, got {type(severity)}")
        if severity < 1 or severity > 5:
            raise ValueError(f"Severity must be between 1 and 5, got {severity}")
            
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
        # Input validation
        if not isinstance(np_img, np.ndarray):
            raise TypeError(f"Input must be a numpy array, got {type(np_img)}")
        if np_img.dtype != np.uint8:
            raise ValueError(f"Input array must be uint8, got {np_img.dtype}")
        if np_img.ndim != 3:
            raise ValueError(f"Input array must have 3 dimensions, got shape {np_img.shape}")
            
        try:
            alpha_scale, sigma, affine_scale = self.params
            img_float = np_img.astype(np.float32) / 255.0
            shape = img_float.shape
            shape_size = shape[:2] # (H, W)

            # --- Random Affine Transform ---
            try:
                center_square = np.float32(shape_size) // 2
                square_size = min(shape_size) // 3
                # Define 3 points for affine transform
                pts1 = np.float32([center_square + square_size,
                                [center_square[0] + square_size, center_square[1] - square_size],
                                center_square - square_size])
                
                current_imsize = max(shape_size)
                scaled_affine_disp = affine_scale * (current_imsize / 32.0)
                pts2 = pts1 + np.random.uniform(-scaled_affine_disp, scaled_affine_disp, size=pts1.shape).astype(np.float32)

                M = cv2.getAffineTransform(pts1, pts2)
                if M is None:
                    raise ValueError("Failed to compute affine transform matrix")
                    
                img_affine = cv2.warpAffine(img_float, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
                
            except Exception as e:
                raise RuntimeError(f"Error in affine transformation: {str(e)}")

            # --- Elastic Deformation ---
            try:
                scaled_alpha = alpha_scale * (current_imsize / 32.0)
                scaled_sigma = sigma * (current_imsize / 32.0)

                dx = gaussian(np.random.uniform(-1, 1, size=shape_size),
                            scaled_sigma, mode='reflect', truncate=3) * scaled_alpha
                dy = gaussian(np.random.uniform(-1, 1, size=shape_size),
                            scaled_sigma, mode='reflect', truncate=3) * scaled_alpha

                dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

                x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

                indices = (np.reshape(y + dy, (-1, 1)),
                        np.reshape(x + dx, (-1, 1)),
                        np.reshape(z, (-1, 1)))

                img_elastic = map_coordinates(img_affine, indices, order=1, mode='reflect').reshape(shape)
                
            except Exception as e:
                raise RuntimeError(f"Error in elastic deformation: {str(e)}")

            return np.clip(img_elastic * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            raise RuntimeError(f"Error in elastic transform: {str(e)}")

    def forward(self, x):
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"Input tensor must have 4 dimensions (B,C,H,W), got {x.ndim}")
        if x.min() < 0.0 or x.max() > 1.0:
            raise ValueError(f"Input tensor values must be in range [0.0, 1.0], got range [{x.min():.3f}, {x.max():.3f}]")
            
        try:
            device = x.device
            dtype = x.dtype
            processed_batch = []
            
            for i in range(x.shape[0]):
                try:
                    img_np = tensor_to_numpy_uint8(x[i])
                    # Ensure 3 channels if grayscale
                    if img_np.ndim == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    elif img_np.shape[2] == 1:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                    corrupted_np = self._elastic_transform_single(img_np)
                    processed_batch.append(numpy_uint8_to_tensor(corrupted_np))
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image {i} in batch: {str(e)}")

            return torch.stack(processed_batch).to(device=device, dtype=dtype)
            
        except Exception as e:
            raise RuntimeError(f"Error in forward pass: {str(e)}")

    def __repr__(self):
        return self.__class__.__name__ + f'(severity={self.severity}, params={self.params})'

