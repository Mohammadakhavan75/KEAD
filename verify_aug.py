import torch
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import cv2 # Needed for some conversions/checks

# --- Add project paths if necessary ---
# Ensure the directories containing the modules are in the Python path
# Adjust these paths if your project structure is different
project_root = '/Users/mohammad/Documents/projects/KEAD'
sys.path.append(project_root)
# Add specific paths if imports fail
# sys.path.append(os.path.join(project_root, 'models/utils'))
# sys.path.append(os.path.join(project_root, 'preprocessing'))

# --- Import Augmentation Implementations ---
try:
    from models.utils.augmentation_layers import (
        GaussianNoise as LayerGaussianNoise,
        ShotNoise as LayerShotNoise,
        ImpulseNoise as LayerImpulseNoise,
        SpeckleNoise as LayerSpeckleNoise,
        GaussianBlur as LayerGaussianBlur,
        GlassBlur as LayerGlassBlur,
        DefocusBlur as LayerDefocusBlur,
        MotionBlur as LayerMotionBlur,
        ZoomBlur as LayerZoomBlur,
        Fog as LayerFog,
        Frost as LayerFrost,
        Snow as LayerSnow,
        Spatter as LayerSpatter,
        Contrast as LayerContrast,
        Brightness as LayerBrightness,
        Saturate as LayerSaturate,
        JpegCompression as LayerJPEGCompression,
        Pixelate as LayerPixelate,
        ElasticTransform as LayerElasticTransform,
        tensor_to_pil, pil_to_tensor, tensor_to_numpy_uint8, numpy_uint8_to_tensor,
        _wand_available # Check if Wand is available for layers
    )
    layers_imported = True
except ImportError as e:
    print(f"Error importing from augmentation_layers.py: {e}")
    layers_imported = False

try:
    # Import functions directly, renaming to avoid conflicts if necessary
    from preprocessing.create_augmentation import (
        gaussian_noise as func_gaussian_noise,
        shot_noise as func_shot_noise,
        impulse_noise as func_impulse_noise,
        speckle_noise as func_speckle_noise,
        gaussian_blur as func_gaussian_blur,
        glass_blur as func_glass_blur,
        defocus_blur as func_defocus_blur,
        motion_blur as func_motion_blur,
        zoom_blur as func_zoom_blur,
        fog as func_fog,
        frost as func_frost,
        snow as func_snow,
        spatter as func_spatter,
        contrast as func_contrast,
        brightness as func_brightness,
        saturate as func_saturate,
        jpeg_compression as func_jpeg_compression,
        pixelate as func_pixelate,
        elastic_transform as func_elastic_transform
    )
    # Check if Wand is available for functions (implicitly checked by MotionBlur/Snow import)
    try:
        from wand.image import Image as WandImage
        func_wand_available = True
    except ImportError:
        func_wand_available = False
    functions_imported = True
except ImportError as e:
    print(f"Error importing from create_augmentation.py: {e}")
    functions_imported = False

# --- Configuration ---
SEED = 123
OUTPUT_DIR = 'augmentation_verification_output'
TEST_IMAGE_SIZE = 64 # Smaller size for faster testing
PASS_THRESHOLD_MEAN = 15.0 # Max average pixel difference (0-255 scale)
PASS_THRESHOLD_MAX = 60.0 # Max single pixel difference (0-255 scale)

# --- Setup ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def generate_test_image(size=TEST_IMAGE_SIZE, batch_size=1):
    """Generate a random test image tensor [B, C, H, W] in range [0.0, 1.0]"""
    tensor = torch.rand(batch_size, 3, size, size)
    return tensor

def visualize_comparison(original_pil, output1_pil, output2_pil, title, filename):
    """Visualize original image and outputs from both implementations"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_pil)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(output1_pil)
    axes[1].set_title('Layer Output')
    axes[1].axis('off')

    axes[2].imshow(output2_pil)
    axes[2].set_title('Function Output')
    axes[2].axis('off')

    plt.suptitle(title, fontsize=10) # Smaller font size for long titles
    fig.subplots_adjust(top=0.88) # Adjust top margin
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

def to_comparable_numpy(data):
    """Converts input (Tensor, PIL, NumPy) to NumPy uint8 [H, W, C]"""
    if isinstance(data, torch.Tensor):
        # Ensure tensor is on CPU and has batch dim removed
        if data.dim() == 4 and data.shape[0] == 1:
            data = data.squeeze(0)
        elif data.dim() != 3:
             raise ValueError(f"Unexpected tensor shape: {data.shape}")
        np_array = tensor_to_numpy_uint8(data)
    elif isinstance(data, Image.Image):
        # Convert PIL to NumPy
        np_array = np.array(data)
        # Ensure 3 channels if grayscale
        if np_array.ndim == 2:
            np_array = cv2.cvtColor(np_array, cv2.COLOR_GRAY2RGB)
        elif np_array.shape[2] == 1:
             np_array = cv2.cvtColor(np_array, cv2.COLOR_GRAY2RGB)
    elif isinstance(data, np.ndarray):
        np_array = data
    else:
        raise TypeError(f"Unsupported data type for comparison: {type(data)}")

    # Ensure uint8 and 0-255 range
    if np_array.dtype != np.uint8:
        if np.issubdtype(np_array.dtype, np.floating):
            np_array = np.clip(np_array * 255, 0, 255).astype(np.uint8)
        else:
            np_array = np.clip(np_array, 0, 255).astype(np.uint8)

    # Ensure HWC format
    if np_array.ndim == 3 and np_array.shape[0] in [1, 3]: # Check if it might be CHW
        # Simple heuristic: if first dim is 1 or 3, assume CHW and transpose
        np_array = np_array.transpose((1, 2, 0))

    # Ensure 3 channels if grayscale numpy was passed
    if np_array.ndim == 2:
        np_array = cv2.cvtColor(np_array, cv2.COLOR_GRAY2RGB)
    elif np_array.ndim == 3 and np_array.shape[2] == 1:
        np_array = cv2.cvtColor(np_array, cv2.COLOR_GRAY2RGB)

    # Final shape check
    if np_array.ndim != 3 or np_array.shape[2] != 3:
         raise ValueError(f"Could not convert data to HWC uint8 format. Final shape: {np_array.shape}, dtype: {np_array.dtype}")

    return np_array


def compute_difference(output1, output2):
    """Compute numerical difference between outputs after converting them"""
    try:
        output1_np = to_comparable_numpy(output1)
        output2_np = to_comparable_numpy(output2)
    except (TypeError, ValueError) as e:
        return {"error": f"Conversion failed: {e}"}

    # Ensure same shape (resize if necessary - might indicate an issue)
    if output1_np.shape != output2_np.shape:
        h1, w1, _ = output1_np.shape
        h2, w2, _ = output2_np.shape
        print(f"Warning: Output shapes differ - Layer: {output1_np.shape}, Function: {output2_np.shape}. Resizing Function output to match Layer output.")
        # Resize output2 to match output1 using OpenCV
        output2_np = cv2.resize(output2_np, (w1, h1), interpolation=cv2.INTER_LINEAR)
        # Ensure it's still 3 channels if resize somehow changed it
        if output2_np.ndim == 2:
            output2_np = cv2.cvtColor(output2_np, cv2.COLOR_GRAY2RGB)


    # Compute difference metrics
    abs_diff = np.abs(output1_np.astype(float) - output2_np.astype(float))
    mean_diff = np.mean(abs_diff)
    max_diff = np.max(abs_diff)

    return {
        'mean_diff': mean_diff,
        'max_diff': max_diff,
        'abs_diff_map': abs_diff # Optional: keep the difference map
    }

# Function to test a pair of augmentations
def test_augmentation_pair(layer_class, func, severity, test_image_tensor, name):
    """Test a pair of augmentations (layer and function) and compare results"""
    print(f"--- Testing {name} (Severity {severity}) ---")

    # Reset seeds for each specific test for better isolation
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # --- Apply Layer ---
    try:
        layer = layer_class(severity=severity)
        # Ensure input tensor is on the correct device if using GPU
        # device = next(layer.parameters()).device if list(layer.parameters()) else torch.device('cpu')
        # layer_output_tensor = layer(test_image_tensor.to(device))
        # Forcing CPU for simplicity as many ops require it anyway
        layer_output_tensor = layer(test_image_tensor.cpu())
        layer_output_pil = tensor_to_pil(layer_output_tensor.squeeze(0)) # For visualization
    except Exception as e:
        print(f"  ERROR applying Layer: {e}")
        return {"error": f"Layer execution failed: {e}"}

    # --- Apply Function ---
    try:
        # Function usually expects PIL Image
        test_image_pil = tensor_to_pil(test_image_tensor.squeeze(0))
        func_output_raw = func(test_image_pil, severity=severity)

        # Function output might be PIL or NumPy, convert PIL for visualization
        if isinstance(func_output_raw, Image.Image):
            func_output_pil = func_output_raw
        elif isinstance(func_output_raw, np.ndarray):
             # Convert NumPy (0-255) to PIL
             func_output_pil = Image.fromarray(np.clip(func_output_raw, 0, 255).astype(np.uint8))
        else:
             raise TypeError(f"Unexpected function output type: {type(func_output_raw)}")

    except Exception as e:
        print(f"  ERROR applying Function: {e}")
        return {"error": f"Function execution failed: {e}"}

    # --- Compare Outputs ---
    diff = compute_difference(layer_output_tensor, func_output_raw)

    if "error" in diff:
        print(f"  ERROR comparing outputs: {diff['error']}")
        # Still try to visualize if possible
        try:
            visualize_comparison(
                tensor_to_pil(test_image_tensor.squeeze(0)),
                layer_output_pil,
                func_output_pil, # Use the PIL version for visualization
                f"ERROR: {name} (Severity {severity}) - Comparison Failed",
                f"ERROR_{name.lower().replace(' ', '_')}_severity_{severity}.png"
            )
        except Exception as viz_e:
            print(f"    Visualization also failed: {viz_e}")
        return diff # Return the comparison error

    # --- Visualize ---
    mean_diff_str = f"{diff['mean_diff']:.2f}"
    max_diff_str = f"{diff['max_diff']:.2f}"
    viz_title = f"{name} (Sev {severity}) | Mean Diff: {mean_diff_str}, Max Diff: {max_diff_str}"
    viz_filename = f"{name.lower().replace(' ', '_')}_severity_{severity}.png"

    try:
        visualize_comparison(
            tensor_to_pil(test_image_tensor.squeeze(0)),
            layer_output_pil,
            func_output_pil,
            viz_title,
            viz_filename
        )
    except Exception as e:
        print(f"  ERROR during visualization: {e}")
        # Continue with reporting results

    # --- Report Pass/Fail ---
    passed = diff['mean_diff'] < PASS_THRESHOLD_MEAN and diff['max_diff'] < PASS_THRESHOLD_MAX
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  Mean Diff: {mean_diff_str}, Max Diff: {max_diff_str} -> {status}")

    return diff

# Main function to run all tests
def run_verification():
    """Run verification tests for all augmentation pairs"""
    # if not layers_imported or not functions_imported:
    #     print("Cannot run verification due to import errors.")
    #     return

    # Define augmentation pairs to test
    # Format: (LayerClass, Function, Name, requires_wand)
    augmentation_pairs = [
        (LayerGaussianNoise, func_gaussian_noise, "Gaussian Noise", False),
        (LayerShotNoise, func_shot_noise, "Shot Noise", False),
        (LayerImpulseNoise, func_impulse_noise, "Impulse Noise", False),
        (LayerSpeckleNoise, func_speckle_noise, "Speckle Noise", False),
        (LayerGaussianBlur, func_gaussian_blur, "Gaussian Blur", False),
        (LayerGlassBlur, func_glass_blur, "Glass Blur", False), # Requires scikit-image
        (LayerDefocusBlur, func_defocus_blur, "Defocus Blur", False), # Requires opencv-python
        (LayerMotionBlur, func_motion_blur, "Motion Blur", True), # Requires Wand
        (LayerZoomBlur, func_zoom_blur, "Zoom Blur", False), # Requires scipy
        (LayerFog, func_fog, "Fog", False),
        (LayerFrost, func_frost, "Frost", False), # Requires opencv-python, needs frost images
        (LayerSnow, func_snow, "Snow", True), # Requires Wand, scipy, opencv-python, PIL
        (LayerSpatter, func_spatter, "Spatter", False), # Requires scikit-image, opencv-python
        (LayerContrast, func_contrast, "Contrast", False),
        (LayerBrightness, func_brightness, "Brightness", False), # Requires scikit-image
        (LayerSaturate, func_saturate, "Saturate", False), # Requires scikit-image
        (LayerJPEGCompression, func_jpeg_compression, "JPEG Compression", False),
        (LayerPixelate, func_pixelate, "Pixelate", False),
        (LayerElasticTransform, func_elastic_transform, "Elastic Transform", False), # Requires scipy, opencv-python
    ]

    # Generate a single test image
    test_image = generate_test_image(size=TEST_IMAGE_SIZE)
    print(f"Generated test image with shape: {test_image.shape}")

    # Test each pair with different severities
    results = {}
    print("\n=== Starting Augmentation Verification ===")
    for layer_class, func, name, requires_wand in augmentation_pairs:
        if requires_wand and (not _wand_available or not func_wand_available):
            print(f"\n--- Skipping {name} (Requires Wand library, not available) ---")
            results[name] = {"skipped": "Wand library not available"}
            continue

        results[name] = {}
        for severity in range(1, 6): # Test all 5 severity levels
            diff = test_augmentation_pair(layer_class, func, severity, test_image.clone(), name) # Use clone
            results[name][severity] = diff

    # Print summary
    print("\n=== Verification Summary ===")
    all_passed = True
    for name, severities in results.items():
        print(f"\n{name}:")
        if "skipped" in severities:
            print(f"  SKIPPED: {severities['skipped']}")
            continue

        passed_severities = 0
        for severity, diff in severities.items():
            if "error" in diff:
                print(f"  Severity {severity}: ❌ ERROR - {diff['error']}")
                all_passed = False
            else:
                passed = diff['mean_diff'] < PASS_THRESHOLD_MEAN and diff['max_diff'] < PASS_THRESHOLD_MAX
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  Severity {severity}: Mean Diff = {diff['mean_diff']:.2f}, Max Diff = {diff['max_diff']:.2f} -> {status}")
                if passed:
                    passed_severities += 1
                else:
                    all_passed = False
        if passed_severities == 5:
             print(f"  Overall: ✅ PASSED ALL SEVERITIES")
        else:
             print(f"  Overall: ❌ FAILED {5-passed_severities}/5 SEVERITIES")


    print("\n--- Verification Complete ---")
    if all_passed:
        print("✅ All tested augmentations passed verification thresholds.")
    else:
        print("❌ Some augmentations failed verification. Check logs and images in:", OUTPUT_DIR)

if __name__ == "__main__":
    # Check if imports were successful before running
    # if layers_imported and functions_imported:
    run_verification()
    # else:
        # print("\nVerification aborted due to import errors.")
        # print("Please ensure both 'models/utils/augmentation_layers.py' and 'preprocessing/create_augmentation.py' are accessible and correct.")