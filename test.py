import pathlib
import numpy as np
import sys
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model, Model
from utils import load_image


def test(name):
    # load saved model
    test_model = load_model(str(pathlib.Path(__file__).parent.absolute()) + "/saved_model_final", compile=False)

    # create the model which will extract last convolution layer, global average pooling and dense connections
    extractor = Model(inputs=test_model.inputs,
                      outputs=[test_model.get_layer('global_average_pooling2d').output,
                               test_model.get_layer('conv2d_4').output,
                               test_model.get_layer('dense').output])

    # load image
    im = load_image(str(pathlib.Path(__file__).parent.absolute()) + '/test_data/' + name, resize=True)

    # extract features from the image
    features = extractor(im)

    # print dense weights
    weights = np.squeeze(extractor.get_layer('dense').get_weights()[0])

    # multiply conv layers by weights
    mult = np.sum(features[1] * weights, axis=-1)

    # save activation
    plt.matshow(np.squeeze(mult), cmap='viridis')
    plt.colorbar()
    plt.savefig('activation.png')

    # get result
    print()
    if test_model(im) > 0.1:
        print("There are some tomatoes in the food")
    else:
        print("No tomatoes in the food")


if __name__ == "__main__":
    test(sys.argv[1])
