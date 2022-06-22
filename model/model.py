from statistics import mode
import tensorflow.keras.applications.vgg16 as vgg16
import tensorflow.keras.applications.inception_v3 as inception_v3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
import numpy as np
import os

def __print_cuda_summary():
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    print("Cuda Availability: ", tf.test.is_built_with_cuda())

def __chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0
# image shape=[samples, rows, cols, channels]

def extract_features(image_dir_path, frames_chunks=25, features_chunks = 5, selection_factor = 5, arq = "InceptionV3"):

    __print_cuda_summary()
    
    files = os.listdir(image_dir_path)
    files = [ f for f in files if f.endswith(".jpg") ]
    files = sorted(files)
    frames_count = len(files)

    print(f"Step 1: Processing {frames_count} frames from path {image_dir_path}")

    if arq == "VGG16":
        module = vgg16
        instance = vgg16.VGG16(weights='imagenet', include_top=False)
    
    elif arq == "InceptionV3":
        module = inception_v3
        instance = inception_v3.InceptionV3(weights='imagenet', include_top=False)

    model = Flatten(name="flatten") (instance.output)
    model = Model(inputs=instance.input, outputs=model)

    model.summary()
    
    X = list()
    X_Control = list()

    frame_chunks = __chunks(files,frames_chunks)

    for j,chunck in enumerate(frame_chunks):
        print(f"\tReading chunk {j+1}")
        for i in range(0, len(chunck)):
            if i*selection_factor < len(chunck):
                img_path=image_dir_path+chunck[i*selection_factor]
                img = image.load_img(img_path)
                loaded_img = image.img_to_array(img)
                img_dims = np.expand_dims(loaded_img, axis=0)
                X.append(img_dims)
                X_Control.append(chunck[i*selection_factor])
    
    print("Step 2: Images ready as numpy arrays")
    X = np.array(X).squeeze() #Removes the 'samples' part of the shape from each image, which is of shape=1, and the new amount of sample will be the first array, containing the amount of images.
    X = module.preprocess_input(X)

    print("Step 3: Extracting features with chucks")

    final_features = []
    arrays_chunks = __chunks(X,features_chunks) 

    x_chunks = sum([1 for i in  __chunks(X,features_chunks)])
    counter = 0

    for arr in arrays_chunks:
        print("Processing chunk %d of %d" % (counter,x_chunks))
        for e in model.predict(arr):
            final_features.append(e)
        counter += 1

    return np.array(final_features), X_Control,files