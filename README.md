# image_entropy_GPU
Script for calculating the entropy of the batch of images on GPU with Pytorch.
Two options are available
1. Calculate entropy of the whole image - pass `--patch_size 0`
2. Calculate entropy of patches in the image - pass `--patch_size 'size of the patch'` (default is 64)

## Requirements
for installing required packages run
` pip install -r requirements.txt`

## Usage
To test the code run

`python main.py --data_path 'path to images' `


## Acknowledgement
Calculation of the pdf function is modified from [kornia](https://github.com/kornia/kornia). 



