#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:07:19 2020

@author: prakash
"""

import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2,os
import glob



#image_path = "/home/prakash/Documents/test/medical white coat in lab/2f93bb11476d44aca98854349393d10e.jpg"

#image_string=tf.read_file(image_path)

#image=tf.image.decode_jpeg(image_string,channels=3)
# image=tf.image.convert_image_dtype(image,dtype=tf.float32)

# img_dir = '/home/prakash/Documents/test/data/valid/white frock'

def img_augmentation(img_dir,num_images):

    try:
        if os.path.isdir(img_dir) and os.path.exists(img_dir):
            img_dir_path = img_dir
    except Exception as e:
        print('No valid directory ', e)

    images = np.array([cv2.resize(cv2.imread(img),(224,224)) for img in glob.glob(img_dir_path + '/*.jpg') for _ in range(num_images)])

    seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8))
        ], random_order=True) # apply augmenters in random order

    images_aug = seq(images=images)

    return(images_aug)


def write_img(images , root_dir):

    for idx,im in enumerate(images):

        file_path = os.path.join(root_dir, 'mod_med_coat_%s.jpg' % idx)

        cv2.imwrite(file_path, im)

    return (file_path)

def run(save_directory, num_images=10):
    logger.info("Augmentation images")
    images = img_augmentation(img_dir,num_images)
    logger.info("Transforming the images")
    write_img(images, save_directory)
    logger.info("Finished")

def main():
    parser = argparse.ArgumentParser(description='Image Augmentation')
    parser.add_argument('-i','--scrap_img_dir',type=str,help='Scraped Images directory')
    parser.add_argument('-n', '--num_images', default=10, type=int, help='num images to save')
    parser.add_argument('-d', '--directory', default='/home/prakash/Documents/test/data/test/white frock/', type=str, help='save directory')
    args = parser.parse_args()
    run(args.scrap_img_dir, args.directory, args.num_images)

if __name__=='__main__':
    main()
# write_img(images_aug,'/home/prakash/Documents/test/data/valid/white frock')
