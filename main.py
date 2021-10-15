import argparse
import torch
from glob import glob
from PIL import Image, ImageDraw
import os
from torchvision import transforms as transforms
from entropy import Entropy

parser = argparse.ArgumentParser('calculate entropy of the batch of images or image patches')

parser.add_argument('--data_path', type=str, help='path to the images')
parser.add_argument('--patch_size', type=int, default=64, help='size of the each patch which entropy are going to be '
                                                               'calculated. 0 for passing full image')
parser.add_argument('--image_size', type=int, default=512, help='size of the images in the batch')
parser.add_argument('--result_path', type=str, default='./results', help='folder to store the images with the outlined '
                                                                         'patches with the highest entropy')


args = parser.parse_args()

if args.image_size % args.patch_size != 0:
    print('image size must be dividable to the patch size')
    exit()

paths = glob(os.path.join(args.data_path, "*"))

list_to_tensor = transforms.ToTensor()
images = []
for path in paths:
    img = Image.open(path).convert('RGB')
    img = img.resize((args.image_size, args.image_size), Image.BICUBIC)
    images.append(list_to_tensor(img))


entropy = Entropy(patch_size=args.patch_size, image_width=args.image_size, image_height=args.image_size)

# tensor with the shape (len(paths) x 3 x h x w)
images = torch.stack(images)
entropy_values = entropy(images)

# outline 10 patches per image with the highest entropy
if not os.path.exists(args.result_path):
    os.mkdir(args.result_path)

sorted_entropy_indices = torch.argsort(entropy_values, dim=1, descending=True)
num_of_max_patches = 10
max_entropy_patches = sorted_entropy_indices[:, 0:num_of_max_patches]

num_patches_in_row = args.image_size / args.patch_size
max_entropy_top = (max_entropy_patches / num_patches_in_row).to(dtype=torch.int) * args.patch_size
max_entropy_left = (max_entropy_patches % num_patches_in_row) * args.patch_size

for k, path in enumerate(paths):
    image = Image.open(path).convert("RGB")
    image = image.resize((args.image_size, args.image_size), Image.BICUBIC)

    for i in range(num_of_max_patches):
        shape = (max_entropy_left[k][i], max_entropy_top[k][i],
                 max_entropy_left[k][i] + args.patch_size, max_entropy_top[k][i] + args.patch_size)
        draw = ImageDraw.Draw(image)
        draw.rectangle(shape, outline="red", width=5)

    image.save(f"results/{path.split('/')[-1]}")
