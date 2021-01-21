
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/Iterator
from PIL import Image
from tensorflow.keras.preprocessing.image import Iterator, img_to_array

class ImageIterator(Iterator):
# __init__()
# if data augmentation -> run and cache
# if prepare_subfunctions -> run on all data, prepare and cache
# init data augmentation

    def __init__(self):
        #asdasdasd todo
        super(ImageIterator, self).__init__(len(image_paths), batch_size, shuffle, seed)



    def _get_batches_of_transformed_samples(self, index_array):
        todo
    # for loop over index_array list:
    # -> load image np.array(Image.open(filepath))
    # -> data augmentation
    # -> get or run subfunctions on dataset (possible caching)
    # return batch(img, class, weight)


# multiprocessing: for loop via map
