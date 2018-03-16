import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage, misc
import re


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

with tf.Session() as sess:
    print("starting session")
    saver = tf.train.import_meta_graph("{0}.meta".format("./digits_modelV3.ckpt"))
    saver.restore(sess, "./digits_modelV3.ckpt")
    
    x = sess.graph.get_tensor_by_name("input:0")
    y_conv = sess.graph.get_tensor_by_name("output:0")

    images = []
    for root, dirnames, filenames in os.walk("./digits"):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff|gif)$", filename):
                filepath = os.path.join(root, filename)
                image = ndimage.imread(filepath, mode="L")
                image_resized = np.reshape(np.pad(misc.imresize(image, (16, 16)),6,'constant'),(28,28))
                plt.imshow(image_resized, cmap='Greys')
                image_b = image_resized.reshape((1, 784))
                result = sess.run(y_conv, feed_dict={x:image_b})
                print()
                print(filename)
                print(sess.run(tf.argmax(result, 1)))
                images.append(image_resized)
                plt.pause(0.05)