import json
import pandas as pd
import os
import pathlib
import tensorflow as tf
import shutil

from utils import load_config
from model import get_model

# get the directory path
folder_path = str(pathlib.Path(__file__).parent.absolute())

# load config
config = load_config(str(pathlib.Path(__file__).parent.absolute()) + "/config.yaml")
if config:
    print("Config loaded correctly")

# get values from config
val_spl = config["train"]["val_split"]
im_h, im_w = config["train"]["im_h"], config["train"]["im_w"]
b_s = config["train"]["b_s"]
n_cl = config["train"]["n_classes"]
ep = config["train"]["ep"]

# load image annotations
with open(folder_path + '/annotations/img_annotations.json') as f:
    image_anns = json.load(f)

# load label annotations
label_df = pd.read_csv(folder_path + '/annotations/label_mapping.csv')

# find strings that contains tomato or Tomato words
correct_labels = label_df[label_df['labelling_name_en'].str.contains('Tom|tom')]
correct_labels = correct_labels['labelling_id'].values.tolist()

# create lists with tomato photos and without
tomato_images = []
no_tomato_images = []
for img in image_anns:
    image_has_tomate = False
    for item in image_anns[img]:
        for label in correct_labels:
            # print(image_anns[img])
            if label in item['id']:
                image_has_tomate = True
    if image_has_tomate:
        tomato_images.append(img)
    else:
        no_tomato_images.append(img)
print("Total number of images: {}".format(len(os.listdir('assignment_imgs'))))
print("We have {} tomato images".format(len(tomato_images)))
print("We have {} images without tomatoes".format(len(no_tomato_images)))

# create data directory
if not os.path.exists(folder_path + '/data'):
    os.makedirs(folder_path + '/data')

# create with_tomatoes and without_tomatoes directories
if not os.path.exists(folder_path + '/data/without_tomatoes'):
    os.makedirs(folder_path + '/data/without_tomatoes')

# create with_tomatoes and without_tomatoes directories
if not os.path.exists(folder_path + '/data/with_tomatoes'):
    os.makedirs(folder_path + '/data/with_tomatoes')

print(os.listdir(folder_path + '/data/with_tomatoes/'))
if tomato_images[0] in os.listdir(folder_path + '/data/with_tomatoes/'):
    print("I'm inside")

# copy all tomato pictures to another dir
for img_name in tomato_images:
    # check if image has not already copied
    if img_name not in os.listdir(folder_path + '/data/with_tomatoes'):
        shutil.copy(folder_path + '/assignment_imgs/' + img_name, folder_path + '/data/with_tomatoes')

# copy all pictures without tomato to another dir
for img_name in no_tomato_images:
    # check if image has not already copied
    if img_name not in os.listdir(folder_path + '/data/without_tomatoes'):
        shutil.copy(folder_path + '/assignment_imgs/' + img_name, folder_path + '/data/without_tomatoes')

# divide data on train and validation
# load train part
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    folder_path + '/data',
    validation_split=val_spl,
    subset="training",
    seed=123,
    image_size=(im_h, im_w),
    batch_size=b_s)

if train_ds:
    print("Train dataset prepared successfully")

# load validation part
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    folder_path + '/data',
    validation_split=val_spl,
    subset="validation",
    seed=123,
    image_size=(im_h, im_w),
    batch_size=b_s)

if val_ds:
    print("Val dataset prepared successfully")

# in the following lines we will make sure that we use our memory properly while reading
# the images for trainig. Cache keeps images in memory after they were loaded. Prefetch
# overlaps data preprocessing and model execution while training.

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# get the model
model = get_model(im_h, im_w, n_cl)
model.summary()

# save model callback
list_of_callbacks = []
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(pathlib.Path(__file__).parent.absolute()) + config["train"]["save_path"],
    save_freq=1)
list_of_callbacks.append(model_checkpoint_callback)


# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['precision', 'recall'])

# train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=list_of_callbacks,
    epochs=ep
)
