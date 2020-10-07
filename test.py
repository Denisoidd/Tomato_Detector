import pathlib
import numpy as np

from tensorflow.keras.models import load_model
from utils import load_image, precision_m, recall_m

# load saved model
test_model = load_model(str(pathlib.Path(__file__).parent.absolute()) + "/saved_model_15ep_bin", compile=True,
                        custom_objects={'precision_m': precision_m, 'recall_m': recall_m})

# load image
im = load_image(str(pathlib.Path(__file__).parent.absolute()) + '/test_data/3.jpeg')

# get result
print(test_model(im))
print(test_model.predict(im))
if test_model(im) > 0.5:
    print("No tomatoes in the food")
else:
    print("There are some tomatoes in the food")