import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.patches as patches
# -----------------------------
# 1. Load Pre-trained ResNet-18
# -----------------------------
model = models.resnet18(pretrained=True)
model.eval()  # set model to evaluation mode

# -----------------------------------
# 2. Set up hooks to capture features
# -----------------------------------
# This dictionary will store the outputs of the layers we hook into.
feature_maps = {}

def hook_fn(module, input, output):
    """A hook function that stores the output of a module."""
    feature_maps[module] = output

# Register hooks on every convolutional layer
# (Alternatively, you can choose specific layers by name.)
conv_layers = []  # to keep track of the layer names
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(hook_fn)
        conv_layers.append(name)

print("Registered hooks on the following conv layers:")
print(conv_layers)

# --------------------------------------
# 3. Prepare an input image for the model
# --------------------------------------
# Replace 'path_to_your_image.jpg' with the path to your image.
img_path = "cat.jpg"
img = Image.open(img_path).convert('RGB')

# Define the transformation (ResNet expects 224x224, normalized images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0)  # add batch dimension

# ------------------------------------
# 4. Forward pass through the network
# ------------------------------------
with torch.no_grad():
    _ = model(img_tensor)

# --------------------------------------------------
# 5. Helper functions to visualize feature maps
# --------------------------------------------------
def plot_feature_maps(feature, title, names, num_cols=3):
    """
    Plot up to 9 feature maps in a 3x3 grid.
    
    Parameters:
        feature (torch.Tensor): Feature map of shape [1, C, H, W].
        title (str): Title for the plot.
        num_cols (int): Number of columns in the grid.
    """
    num_maps = feature.shape[1]
    num_plots = min(num_maps, 9)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    for i in range(9):
        ax = axes[i // num_cols, i % num_cols]
        if i < num_plots:
            fmap = (feature[0, i].cpu().numpy() * 255.).astype(np.uint8)
            # fmap = cv2.cvtColor(fmap, cv2.COLOR_GRAY2RGB)
            # cv2.rectangle(fmap, (0, 0), (3, 3), (255, 0, 0), 1)
            rect = patches.Rectangle((-0.4, -0.4), 3, 3, linewidth=4, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f"Map {i}")
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"./feature_map/{names[0]}_{names[1]}.png", dpi=1000)
    # plt.show()

def plot_feature_patch(feature, channel_idx, title):
    """
    Plot a 3x3 patch from the center of a feature map channel.
    
    Parameters:
        feature (torch.Tensor): Feature map of shape [1, C, H, W].
        channel_idx (int): Index of the channel to visualize.
        title (str): Title for the plot.
    """
    fmap = feature[0, channel_idx].cpu().numpy()
    h, w = fmap.shape
    # Compute indices for a 3x3 patch at the center
    center_h, center_w = h // 2, w // 2
    patch = fmap[center_h - 1:center_h + 2, center_w - 1:center_w + 2]
    plt.figure(figsize=(4, 4))
    plt.imshow(patch, cmap='viridis')
    plt.title(title, fontsize=14)
    plt.colorbar()
    plt.axis('off')
    plt.show()



# --------------------------------------------------
# 6. Visualize feature maps at different depths
# --------------------------------------------------

if model.layer1[0].conv1 in feature_maps:
    plot_feature_maps(feature_maps[model.layer1[0].conv1], "Feature Maps from conv1", ("layer1", "conv1"))

if model.layer2[0].conv1 in feature_maps:
    plot_feature_maps(feature_maps[model.layer2[0].conv1], "Feature Maps from conv1", ("layer2", "conv1"))

if model.layer3[0].conv1 in feature_maps:
    plot_feature_maps(feature_maps[model.layer3[0].conv1], "Feature Maps from conv1", ("layer3", "conv1"))

if model.layer4[0].conv1 in feature_maps:
    plot_feature_maps(feature_maps[model.layer4[0].conv1], "Feature Maps from conv1", ("layer4", "conv1"))




# --------------------------------------------------
# 7. Visualize original image with 3*3 filter
# --------------------------------------------------

img = img_tensor.squeeze(0).permute(1,2,0).detach().numpy()
rect = patches.Rectangle((-0.4, -0.4), 3, 3, linewidth=4, edgecolor='r', facecolor='none')
fig, ax = plt.subplots()
ax.add_patch(rect)
ax.axis('off')
ax.imshow(img)
plt.savefig(f"./feature_map/original_img.png", dpi=1000)