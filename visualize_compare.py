import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageChops

def compare_coarse_with_controlnet(dataset_name='helmet2',test_or_train='test'):
    """ compare output images of coarse and controlnet for a given dataset"""
    output_dir_path = f'output/gs_init/{dataset_name}'
    coarse_output_dir_path = f'{output_dir_path}/{test_or_train}/ours_10000/renders/'
    controlnet_output_dir_path = f'{output_dir_path}/{test_or_train}/ours_None/renders/'
    comparison_output_dir = f'{output_dir_path}/{test_or_train}/comparison'
    # makedir if needed
    os.makedirs(comparison_output_dir, exist_ok=True)
    # read images_names
    coarse_img_names = os.listdir(coarse_output_dir_path)
    controlnet_img_names = os.listdir(controlnet_output_dir_path)
    assert len(coarse_img_names) == len(controlnet_img_names)
    # display every images side by side 
    for i in range(len(coarse_img_names)):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        coarse_img = plt.imread(f'{coarse_output_dir_path}/{coarse_img_names[i]}')
        controlnet_img = plt.imread(f'{controlnet_output_dir_path}/{controlnet_img_names[i]}')
        # display images side by side
        axs[0].imshow(coarse_img)
        axs[0].axis('off')
        axs[1].imshow(controlnet_img)
        axs[1].axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle(f'Comapring (GO) Coarse and Controlnet for helmet')
        plt.savefig(f'{comparison_output_dir}/comparison_{i}.jpg', bbox_inches='tight')
    print(f'saved {comparison_output_dir}')    

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    

def compare_coarse_with_controlnet_PIL(dataset_name='helmet2',test_or_train='test'):
    """ compare output images of coarse and controlnet for a given dataset"""
    output_dir_path = f'output/gs_init/{dataset_name}'
    coarse_output_dir_path = f'{output_dir_path}/{test_or_train}/ours_10000/renders/'
    controlnet_output_dir_path = f'{output_dir_path}/{test_or_train}/ours_None/renders/'
    comparison_output_dir = f'{output_dir_path}/{test_or_train}/comparison'
    # makedir if needed
    os.makedirs(comparison_output_dir, exist_ok=True)
    # read images_names
    coarse_img_names = os.listdir(coarse_output_dir_path)
    controlnet_img_names = os.listdir(controlnet_output_dir_path)
    assert len(coarse_img_names) == len(controlnet_img_names)
    # display every images side by side 
    for i in range(len(coarse_img_names)):
        coarse_img = Image.open(f'{coarse_output_dir_path}/{coarse_img_names[i]}')
        coarse_img = trim(coarse_img)
        controlnet_img = Image.open(f'{controlnet_output_dir_path}/{controlnet_img_names[i]}')
        controlnet_img = trim(controlnet_img)
        # display images using PIL
        gap_size = 30
        total_width = coarse_img.width + controlnet_img.width + gap_size
        max_height = max(coarse_img.height, controlnet_img.height)
        new_image = Image.new('RGB', (total_width, max_height), 'white')
        new_image.paste(coarse_img, (0, 0))
        new_image.paste(controlnet_img, (coarse_img.width + gap_size, 0))
        new_image.save(f'{comparison_output_dir}/comparison_{i}.png')
    print(f'saved {comparison_output_dir}')   
    
compare_coarse_with_controlnet_PIL(dataset_name='helmet2', test_or_train='test')