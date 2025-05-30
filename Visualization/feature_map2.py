import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def get_feature_maps(model, image):
    feature_maps = []
    hooks = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                hooks.append(sub_layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        model(image)
    
    for hook in hooks:
        hook.remove()
    
    return feature_maps

def plot_feature_maps(feature_maps, num_layers=[1, 5, 10]):
    plt.figure(figsize=(15, 10))
    for i, layer_index in enumerate(num_layers):
        fmap = feature_maps[layer_index][0]  # First image in batch
        fmap = fmap.cpu().numpy()
        num_channels = min(16, fmap.shape[0])  # Display up to 16 feature maps
        
        for j in range(num_channels):
            plt.subplot(len(num_layers), num_channels, i * num_channels + j + 1)
            plt.imshow(fmap[j], cmap='viridis')
            plt.axis("off")
            if j == 0:
                plt.ylabel(f'Layer {layer_index}', fontsize=12, color='red')
        
        plt.savefig(f"./feature_map/{i}_{layer_index}.png")


def plot_convolution_on_feature_map(feature_map):
    fmap = feature_map[0].cpu().numpy()
    fig, ax = plt.subplots()
    ax.imshow(fmap[0], cmap='viridis')
    
    kernel_size = 3
    stride = 10  # Adjust for visibility
    
    for i in range(0, fmap.shape[1] - kernel_size, stride):
        for j in range(0, fmap.shape[2] - kernel_size, stride):
            rect = plt.Rectangle((j, i), kernel_size, kernel_size, edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(rect)
    
    plt.show()

if __name__ == "__main__":
    image_path = "cat.jpg"  # Change this to your image path
    model = models.resnet18(pretrained=True).eval()
    image = load_image(image_path)
    
    feature_maps = get_feature_maps(model, image)
    plot_feature_maps(feature_maps)
    
    # Plot convolutional filter on a deeper layer
    plot_convolution_on_feature_map(feature_maps[10])