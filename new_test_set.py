# Create new Dataset
import os
import PIL
import cv2
import argparse
import collections
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision import transforms
from skimage.filters import gaussian
from wand.image import Image as WandImage
from scipy.ndimage import zoom as scizoom
from wand.api import library as wandlibrary
from dataset_loader import load_cifar10, load_svhn, load_cifar100
from torchvision import transforms as T


IMG_SIZE=32


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


def rot90(img):
    rotater = transforms.RandomRotation(degrees=(270, 270))
    return rotater(img)


def flip(img):
    hflipper = transforms.RandomHorizontalFlip(p=1.0)
    return hflipper(img)


def random_crop(img):
    resize_cropper = transforms.RandomCrop(size=(int(IMG_SIZE * 0.86), int(IMG_SIZE * 0.86))) # (32, 32)
    resized = transforms.Resize(size=IMG_SIZE)(resize_cropper(img))
    return resized


def gaussian_noise(x, severity=5):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def glass_blur(x, severity=5):
    # sigma, max_delta, iterations
    c = [(0.05,1,1), (0.25,1,1), (0.4,1,1), (0.25,1,2), (0.4,1,2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(IMG_SIZE - c[1], c[1], -1):
            for w in range(IMG_SIZE - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def jpeg_compression(x, severity=5):
    c = [80, 65, 58, 50, 40][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)

    return x


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def snow(x, severity=5):
    c = [(0.1,0.2,1,0.6,8,3,0.95),
         (0.1,0.2,1,0.5,10,4,0.9),
         (0.15,0.3,1.75,0.55,10,4,0.9),
         (0.25,0.3,2.25,0.6,12,6,0.85),
         (0.3,0.3,1.25,0.65,14,12,0.8)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(IMG_SIZE, IMG_SIZE, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def gaussian_blur(x, severity=5):
    c = [.4, .6, 0.7, .8, 1][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def color_jitter(x):
    jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    return jitter(x)


def rot270(x):
    rotater = T.RandomRotation(degrees=(90, 90))
    return rotater(x)


noises_list = ['rot90', 'flip', 'random_crop', 'gaussian_noise', 'glass_blur', 'jpeg_compression', 'snow', 'rot270', 'color_jitter', 'gaussian_blur']

noise_dict = collections.OrderedDict()
noise_dict['rot90'] = rot90
noise_dict['flip'] = flip
noise_dict['random_crop'] = random_crop
noise_dict['gaussian_noise'] = gaussian_noise
noise_dict['glass_blur'] = glass_blur
noise_dict['jpeg_compression'] = jpeg_compression
noise_dict['snow'] = snow
noise_dict['rot270'] = rot270
noise_dict['color_jitter'] = color_jitter
noise_dict['gaussian_blur'] = gaussian_blur


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', default=1, type=str,
                        help='Number of epochs to train.')

    args = parser.parse_args()
    return args

args = parsing()

if args.dataset == 'cifar10':
    cifar10_path = '/finals/datasets/data/'
    train_loader, test_loader, shuffle_loader = load_cifar10(cifar10_path, 
                                                             batch_size=1,
                                                             one_class_idx=None, 
                                                             tail_normal=None)
if args.dataset == 'cifar100':
    cifar10_path = '/finals/datasets/data/'
    train_loader, test_loader, shuffle_loader = load_cifar100(cifar10_path, 
                                                             batch_size=1,
                                                             one_class_idx=None)
elif args.dataset == 'svhn':
    svhn_path = '/finals/datasets/data/'
    train_loader, test_loader, shuffle_loader = load_svhn(svhn_path, 
                                                          batch_size=1,
                                                          one_class_idx=None)

os.makedirs(f'new_test_set/{args.dataset}/', exist_ok=True)

np.random.seed(args.seed)
method_names = list(noise_dict.keys())
cifar_c, labels_c, aug = [], [], []
for img, label in test_loader:
    try:
        method_name = np.random.choice(method_names)
        labels_c.append(label.detach().cpu().numpy())
        aug.append(method_name)
        cifar_c.append(np.uint8(noise_dict[method_name](PIL.Image.fromarray((img[0].permute(1,2,0).detach().cpu().numpy() * 255.).astype(np.uint8)))))
    except:
        print(f"Error occured in: {method_name}")

np.save(f'new_test_set/{args.dataset}/test_s6.npy', np.array(cifar_c).astype(np.uint8))
np.save(f'new_test_set/{args.dataset}/test_labels_s6.npy', np.array(labels_c).astype(np.uint8))
np.save(f'new_test_set/{args.dataset}/test_augs_s6.npy', np.array(aug))

if args.dataset == 'shvn':
    anomaly = {
        0: ["gaussian_noise", "glass_blur"],
        1: ["gaussian_noise", "glass_blur"],
        2: ["gaussian_noise", "glass_blur", "flip", "rot90", "rot270"],
        3: ["gaussian_noise", "glass_blur", "flip", "rot90", "rot270"],
        4: ["gaussian_noise", "color_jitter", "flip", "rot90", "rot270"], # random_crop,
        5: ["gaussian_noise", "glass_blur", "flip", "rot90", "rot270"], # random_crop,
        6: ["gaussian_noise", "glass_blur", "flip"] ,
        7: ["gaussian_noise", "glass_blur", "flip", "rot90", "rot270"] ,
        8: ["gaussian_noise", "glass_blur", "rot90", "rot270"] ,
        9: ["gaussian_noise", "glass_blur", "flip", "rot90", "rot270"] ,
    }
elif args.dataset == 'cifar10':
    anomaly = {
        0: ['gaussian_noise', 'glass_blur', 'rot90', 'rot270'], # airplain
        1: ['gaussian_noise', 'glass_blur', 'rot90', 'rot270'], # car
        2: ['gaussian_noise', 'glass_blur'], # bird
        3: ['gaussian_noise', 'glass_blur'], # cat
        4: ['gaussian_noise', 'glass_blur', 'rot90', 'rot270'], # deer
        5: ['gaussian_noise', 'glass_blur'], # dog
        6: ['gaussian_noise', 'glass_blur'], # frog
        7: ['gaussian_noise', 'glass_blur', 'rot90', 'rot270'], # horse
        8: ['gaussian_noise', 'glass_blur', 'rot90', 'rot270'], # ship
        9: ['gaussian_noise', 'glass_blur', 'rot90', 'rot270'], # truck
    }
elif args.dataset == 'cifar100':
    anomaly = {
        0: ["gaussian_noise", "glass_blur"],
        1: ["gaussian_noise", "glass_blur"],
        2: ["gaussian_noise", "glass_blur"],
        3: ["gaussian_noise", "glass_blur"],
        4: ["gaussian_noise", "color_jitter"], # random_crop
        5: ["gaussian_noise", "glass_blur"], # random_crop
        6: ["gaussian_noise", "glass_blur"],
        7: ["gaussian_noise", "glass_blur"],
        8: ["gaussian_noise", "glass_blur"],
        9: ["gaussian_noise", "glass_blur", "rot90", "rot270"],
        10: ["gaussian_noise", "glass_blur", "color_jitter"],
        11: ["gaussian_noise", "rot90", "rot270"],
        12: ["gaussian_noise", "glass_blur"],  # random_crop,
        13: ["gaussian_noise", "glass_blur"],  # random_crop,
        14: ["gaussian_noise", "glass_blur", "color_jitter"],  # random_crop,
        15: ["gaussian_noise", "glass_blur"],
        16: ["gaussian_noise", "glass_blur"],
        17: ["gaussian_noise", "glass_blur", "color_jitter"],
        18: ["gaussian_noise", "glass_blur", "rot90", "rot270"],
        19: ["gaussian_noise", "glass_blur", "rot90", "rot270"],
    }

anomaly_label = []
for i in range(len(aug)):
    if aug[i] in anomaly[labels_c[i][0]]:
        anomaly_label.append(1)
    else:
        anomaly_label.append(0)

np.save(f'./new_test_set/{args.dataset}/test_anomaly_labels_s6.npy', anomaly_label)