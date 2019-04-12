from __future__ import division

import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np

sys.path.append("/home/nowgeun1/Desktop/FCN/")
sys.path.append("/home/nowgeun1/Desktop/FCN/models/slim/")

fcn_16s_checkpoint_path = \
 '/home/nowgeun1/Desktop/FCN/tf_image_segmentation/saver/model_fcn8s_final.ckpt'

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

number_of_classes = 21


image_data_folder_path = "/home/nowgeun1/Desktop/FCN/data/"
image_files = [x for x in os.listdir(image_data_folder_path) if x.endswith((".jpg",".png"))]


slim = tf.contrib.slim
	
image_filename_placeholder = tf.placeholder(tf.string)

image_tensor = tf.read_file(image_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_8s = adapt_network_for_any_size_input(FCN_8s, 32)


pred, fcn_16s_variables_mapping = FCN_8s(image_batch_tensor=image_batch_tensor,
	                                  number_of_classes=number_of_classes,
	                                  is_training=False)

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(initializer)

    saver.restore(sess,
     "/home/nowgeun1/Desktop/FCN/tf_image_segmentation/saver/model_fcn8s_final.ckpt")
    
    
    for image in image_files:
        image_filename =  image_data_folder_path + image
    
        feed_dict_to_use = {image_filename_placeholder: image_filename}

    
    
        image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)

        io.imsave("/home/nowgeun1/Desktop/FCN/tf_image_segmentation/generated/pred_{}".format(image),pred_np.squeeze())

