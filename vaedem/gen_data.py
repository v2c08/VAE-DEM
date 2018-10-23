import os
import numpy as np
import gym
import matplotlib.pyplot as plt
from scipy.misc import imresize
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import six
import time

def get_data(batches, nT, gen_mode, out_dir, target_size, n_channels):
    
    datagen = ImageDataGenerator(rescale = 1./255)
    
    image_generator = datagen.flow_from_directory(os.path.join(out_dir,gen_mode),
                                            target_size=target_size,
                                            shuffle=False,
                                            batch_size=nT,
                                            color_mode='rgb',
                                            class_mode=None)
    
    cam_train = np.genfromtxt(r"{}/{}/{}\\cam_anim.txt".format(out_dir, gen_mode, image_generator.filenames[0][:2]),delimiter=",", autostrip=True)
    camxloc = []
    camyloc = []
    for x in cam_train:
        camxloc.append(x[0])
        camyloc.append(x[1])
    camxloc_y = np.roll(camxloc,1)
    camyloc_y = np.roll(camyloc,1)
    
    obj_train = np.genfromtxt(r"{}/{}/{}\\circle_anim.txt".format(out_dir, gen_mode, image_generator.filenames[0][:2]),delimiter=",", autostrip=True)
    objxrot = []
    objyrot = []
    for x in obj_train:
        objxrot.append(x[0])
        objyrot.append(x[1])
    
    objxrot_y = np.roll(objxrot,1)
    objyrot_y = np.roll(objyrot,1)
    
    
    def prediction_gen(nT, target_size, n_channels, generator, gen_mode):
    
        while True:
            
            if gen_mode != 'test':
                x = next(generator)
                y = np.roll(x,1,axis=1)
                print(x.shape)
                x = np.reshape(x, (1,nT,target_size[0], target_size[1], n_channels))
                y = np.reshape(y, (1,nT,target_size[0], target_size[1], n_channels))
                # yield [x, {'prediction_output':y}]
                yield [x, y]
            else:
                x = next(generator)
                x = np.reshape(x, (1,nT,target_size[0], target_size[1], n_channels))
                yield x
    
    
    gen = prediction_gen(nT, target_size, n_channels, image_generator, gen_mode)
    
    
    return gen, np.array(camxloc), np.array(camyloc), np.array(objxrot), np.array(objyrot)
