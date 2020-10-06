import json
import pandas as pd
import os
import pathlib

# get the directory path
folder_path = str(pathlib.Path(__file__).parent.absolute())

# load image annotations
with open(folder_path + '/annotations/img_annotations.json') as f:
    image_anns = json.load(f)

# load label annotations
label_df = pd.read_csv(folder_path + '/annotations/label_mapping.csv')

# find strings that contains tomato or Tomato words
correct_labels = label_df[label_df['labelling_name_en'].str.contains('Tom|tom')]
correct_labels = correct_labels['labelling_id'].values.tolist()

# create lists with tomato photos and without
has_tomato_images = []
without_tomato_images = []
for img in image_anns:
    image_has_tomate = False
    for item in image_anns[img]:
        for label in correct_labels:
            # print(image_anns[img])
            if label in item['id']:
                image_has_tomate = True
    if image_has_tomate:
        has_tomato_images.append(img)
    else:
        without_tomato_images.append(img)
print("Total number of images: {}".format(len(os.listdir('assignment_imgs'))))
print("We have {} correct images".format(len(has_tomato_images)))
print("We have {} uncorrect images".format(len(without_tomato_images)))
