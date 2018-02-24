import sys
import tensorflow as tf
import os
from PIL import Image, ImageFilter


def predictint(imvalue):
  """
  This function returns the predicted integer.
  The imput is the pixel values from the imageprepare() function.
  """

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Define the model (same as when creating the model file)
  x = tf.placeholder(tf.float32, [None, 784])

  W_fc1 = weight_variable([784, 100])
  b_fc1 = bias_variable([100])

  h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

  W_fc2 = weight_variable([100, 30])
  b_fc2 = bias_variable([30])

  h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

  W_fc3 = weight_variable([30, 10])
  b_fc3 = bias_variable([10])

  h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3

  init_op = tf.initialize_all_variables()

  # saver = tf.train.Saver()

  # First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('/Users/bar/Downloads/tfMnist/my_test_model-1000.meta')
  # saver.restore(sess,tf.train.latest_checkpoint('./'))

  """
  Load the model2.ckpt file
  file is stored in the same directory as this python script is started
  Use the model to predict the integer. Integer is returend as list.
  Based on the documentatoin at
  https://www.tensorflow.org/versions/master/how_tos/variables/index.html
  """
  with tf.Session() as sess:
    sess.run(init_op)
    # saver.restore(sess, "model2.ckpt")
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # print ("Model restored.")

    prediction = tf.argmax(h_fc3, 1)
    return prediction.eval(feed_dict={x: [imvalue]}, session=sess)


def imageprepare(argv):
  """
  This function returns the pixel values.
  The imput is a png file location.
  """
  im = Image.open(argv).convert('L')
  width = float(im.size[0])
  height = float(im.size[1])
  newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

  if width > height:  # check which dimension is bigger
    # Width is bigger. Width becomes 20 pixels.
    nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
    if (nheigth == 0):  # rare case but minimum is 1 pixel
      nheigth = 1
      # resize and sharpen
    img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
    newImage.paste(img, (4, wtop))  # paste resized image on white canvas
  else:
    # Height is bigger. Heigth becomes 20 pixels.
    nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
    if (nwidth == 0):  # rare case but minimum is 1 pixel
      nwidth = 1
    # resize and sharpen
    img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
    newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

  # newImage.save("sample.png")

  tv = list(newImage.getdata())  # get pixel values

  # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
  tva = [(255 - x) * 1.0 / 255.0 for x in tv]
  return tva
  # print(tva)


def main(argv):
  """
  Main function.
  """
  for filename in os.listdir(argv):
    imvalue = imageprepare(filename)
    predint = predictint(imvalue)
    print ('the ' + str(filename), predint[0])  # first value in list


if __name__ == "__main__":
  main(sys.argv[1])