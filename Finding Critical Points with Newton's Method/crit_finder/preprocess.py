import numpy as np
import skimage.transform

def downscale(dataset, factor=4):
    
    train_images, train_labels, test_images, test_labels = get_train_and_test(dataset)
        
    downscaled_train_images = downscale_by_local_mean(train_images, factor)
    downscaled_test_images = downscale_by_local_mean(test_images, factor)
    
    train = {"images":downscaled_train_images,
             "labels":train_labels}
    test = {"images":downscaled_test_images,
           "labels":test_labels}
    
    downscaled_dataset = {"train":train,
                      "test":test}
    
    return downscaled_dataset


def downscale_by_local_mean(images, factor):
    num_images, input_shape = images.shape
    image_shape = int(np.sqrt(input_shape))
    
    downscaled_input_shape = int(input_shape/factor**2)
    
    downscaled_data = np.zeros([num_images, downscaled_input_shape])
    
    for image_idx, image in enumerate(images):
        downscaled_data[image_idx,:] = np.ndarray.flatten(
                                            skimage.transform.downscale_local_mean(
                                            image.reshape([image_shape, image_shape]), 
                                            (factor, factor)
                                            ))
    return downscaled_data

def subsample(dataset, n_test=1000, n_train=1000):
    
    train_images, train_labels, test_images, test_labels = get_train_and_test(dataset)
    
    subsampled_train = subsample_dataset(train_images, train_labels, n_train)
    subsampled_test = subsample_dataset(test_images, test_labels, n_test)

    subsampled_dataset = {"train": subsampled_train,
                       "test": subsampled_test}
    
    return subsampled_dataset

def subsample_dataset(images, labels, n):
    
    random_indices = np.random.choice(images.shape[0], size=n, replace=False)
    subsampled_images = images[random_indices,:]
    subsampled_labels = labels[random_indices,:]
    
    subsampled_dataset = {"images":subsampled_images,
             "labels":subsampled_labels}
    
    return subsampled_dataset

def get_train_and_test(dataset):
    
    if hasattr(dataset, "train"):
        train_images, train_labels = dataset.train.images, dataset.train.labels
        test_images, test_labels = dataset.test.images, dataset.test.labels
    else:
        train = dataset["train"]
        test = dataset["test"]
        
        train_images, train_labels = train["images"], train["labels"]
        test_images, test_labels = test["images"], test["labels"]
        
    return train_images, train_labels, test_images, test_labels