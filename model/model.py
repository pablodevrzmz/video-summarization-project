from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from sklearn_extra.cluster import KMedoids
import numpy as np
import os

def get_frames_len(dir_path):
    _, _, files = next(os.walk(dir_path))
    return len(files)

#https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0
# image shape=[samples, rows, cols, channels]
def extract_features(image_dir_path):
    vgg = VGG16(weights='imagenet', include_top=False)
    model = Flatten(name="flatten") (vgg.output)
    model = Model(inputs=vgg.input, outputs=model)
    #frame_count= get_frames_len(image_dir_path)
    model.summary()
    X = list()
    for i in range(0, 1000):
        #print(i)
        img_path=image_dir_path+str(i)+'.jpg'
        img = image.load_img(img_path)
        loaded_img = image.img_to_array(img)
        img_dims = np.expand_dims(loaded_img, axis=0)
        X.append(img_dims)
    X = np.array(X).squeeze() #Removes the 'samples' part of the shape from each image, which is of shape=1, and the new amount of sample will be the first array, containing the amount of images.
    #print(X.shape)
    #print(model.output)
    X = preprocess_input(X)

    block4_pool_features = model.predict(X)
    kmedoids = KMedoids(n_clusters=2, random_state=0).fit(block4_pool_features)
    print(kmedoids.labels_)
    #print(block4_pool_features.shape)
    #print(block4_pool_features)