# CIFAR-10 Compare Augmentation Effect

Overview
This repository explores the impact of various augmentations on the accuracy of a ResNet34 model trained on the CIFAR-10 dataset. The goal is to assess how different augmentations influence the model's performance in detecting and classifying objects in the CIFAR-10 images.

Training Without Augmentation
To establish a baseline, we first train a ResNet34 model without using any augmentation. The following command can be used for training:
```
python train.py --learning_rate=0.1 --lr_update_rate=2 --lr_gamma=0.5 --optimizer sgd --epochs 10
```
This command trains the model with specified hyperparameters for 10 epochs.

Evaluating Augmentation Effects
After training the model without augmentation, we employ an evaluation code to load augmented data and split it into each class. This allows us to assess the effect of each augmentation on the accuracy of detecting individual classes. The evaluation code examines the performance of the model on augmented data and provides insights into how each augmentation technique influences the model's ability to correctly identify objects in the CIFAR-10 dataset.

Usage
To replicate the experiments and evaluate the augmentation effects, follow these steps:

Train the model without augmentation using the provided training command.

Run the evaluation code to assess the impact of augmentations on class-specific accuracy.

```
./run.sh
```

This will generate results and visualizations to help understand the comparative effect of different augmentations.

Augmentations Explored
The repository explores various augmentations, and additional augmentations can be easily added to the evaluation script. Common augmentations include:


brightness
color_jitter
contrast
defocus_blur
elastic_transform
flip
fog
gaussian_blur
gaussian_noise
glass_blur
impulse_noise
jpeg_compression
motion_blur
pixelate
random_crop
rot270
rot90
saturate
shot_noise
snow
spatter
speckle_noise
zoom_blur		




Experiment with different combinations and parameters to observe how each affects the model's accuracy.

Results
The results and visualizations obtained from the evaluation can be found in the results directory. Analyzing these results will provide valuable insights into the impact of augmentations on the model's performance.

Feel free to customize the code and experiment with additional augmentations to further enhance the understanding of how data augmentation influences the accuracy of the CIFAR-10 classification model.