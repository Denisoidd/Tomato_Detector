import pathlib
import numpy as np
import cv2
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model, Model
from utils import load_image

# load saved model
test_model = load_model(str(pathlib.Path(__file__).parent.absolute()) + "/saved_model_30ep_bin_corlabels", compile=False)

# create the model which will extract last convolution layer, global average pooling and dense connections
extractor = Model(inputs=test_model.inputs,
                  outputs=[test_model.get_layer('global_average_pooling2d').output,
                           test_model.get_layer('conv2d_4').output,
                           test_model.get_layer('dense').output])


# load image
im = load_image(str(pathlib.Path(__file__).parent.absolute()) + '/test_data/4.jpeg', resize=True)

# print extracted features
features = extractor(im)
# print(features)

# print dense weights
weights = np.squeeze(extractor.get_layer('dense').get_weights()[0])
print("Shape of weights is {}".format(weights.shape))

# multiply conv layers by weights
mult = np.sum(features[1] * weights, axis=-1)

# normalize data


print("final matrix")
print(mult)
plt.matshow(np.squeeze(mult), cmap='viridis')
print('////')
print(np.min(mult), np.max(mult))
# plt.clim(np.min(mult), np.max(mult))
plt.colorbar()
plt.savefig('activation.png')

cv2.imwrite('activation.jpg', np.squeeze(mult) * 255)

# print(features[1] * weights)


# get result
print(test_model(im))
# print(test_model.predict(im))
if test_model(im) > 0.15:
    print("There are some tomatoes in the food")
else:
    print("No tomatoes in the food")

