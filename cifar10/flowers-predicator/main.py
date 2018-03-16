from keras.models import load_model
import os
from scipy import ndimage, misc
import re
from numpy.random import shuffle
from numpy import array
from matplotlib import pyplot as plt

images = []
for root, dirnames, filenames in os.walk("./flowers_to_predict"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff|gif)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            images.append(misc.imresize(image, (32, 32)))

shuffle(images)
images = array(images)

print ('loading model')
model = load_model('fine_tune-ep249-loss0.275-BEST_MODEL.h5')
print ('finished loading')

predictions = model.predict(images)

for i in range(len(images)):
    plt.imshow(images[i])
    plt.pause(0.05)
    print('{0} with prediction of {1}'.format(('Flower' if predictions[i] > 0.8 else 'Not flower'), predictions[i]))