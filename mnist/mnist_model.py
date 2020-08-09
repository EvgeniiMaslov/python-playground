#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 12:38:31 2020

@author: evgenii
"""
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import numpy as np


new_model = tf.keras.models.load_model('mnist.h5')
new_model._make_predict_function()
def image_preprocess(filename):
    basewidth = 28
    img = Image.open(filename).convert('L')
    img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
    img = img.filter(ImageFilter.CONTOUR) 
    img = ImageOps.invert(img)
    img.save('test.png')
    img = np.array(img)
    img = img[..., np.newaxis]/255.0
    img
    img = img.reshape(1, 28, 28, 1)
    return img


def apply_mnist(filename):
    img = image_preprocess(filename)
    prediction = new_model.predict(img)
    return prediction.argmax(), prediction.max()
    
print(apply_mnist('user_photo.png'))