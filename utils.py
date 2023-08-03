import shutil
import imghdr
import os
import random
from typing import List, Tuple, Dict
import numpy as np


def check_image(image_dir):
    """
    Checks images if they have correct extension.  Good when downloading from
    a web source to verify images are good valid.

    :param image_dir:
    :return:
    """
    data_dir = image_dir
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    # print(os.listdir(os.path.join(data_dir, 'vangogh')))  -> prints list of Van Gogh images

    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('Image not in list of extensions {}'.format(image_path))
                    os.remove(image_path)
            except IOError:
                print('Issue with image {}'.format(image_path))

    return None


# Count the number of images in a folder


def count_images(image_path):
    """
    Counts number of images in specified path
    :param image_path:
    :return:
    """
    dir_path = image_path
    count = 0

    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1

    folder = dir_path.split("/")[-1]
    print("The number of images in the {} folder = {}".format(folder, count))

    return count


# split images into 80% train, 20% test


def create_sets(class_name, to_path, from_path, pct_train, pct_test):
    """
    Divides set of images into train, test, and valid sets based on percentages
    :param class_name:
    :param to_path:
    :param from_path:
    :param pct_train:
    :param pct_test:
    :return:
    """
    num_images = count_images(from_path)
    image_select = os.listdir(from_path)

    for i in range(num_images):
        value = random.random()
        filename = image_select[i]
        from_loc = from_path + filename
        
        if value <= pct_train:
            train_path = to_path + '/train/' + class_name + filename
            shutil.copy(from_loc, train_path)
        elif pct_train < value < (pct_test + pct_train):
            test_path = to_path + '/test/' + class_name + filename
            shutil.copy(from_loc, test_path)
        else:
            valid_path = to_path + '/valid/' + class_name + filename
            shutil.copy(from_loc, valid_path)

    return


def check_raw_count():
    """
    Check raw image count
    :return:
    """
    raw_not_vangogh = count_images("raw_images/not_vangogh")
    raw_vangogh = count_images("raw_images/vangogh")
    total = raw_vangogh + raw_not_vangogh
    print(f"Total raw Images = {total}")


def check_train_count():
    """
    checks train image count
    :return:
    """
    train_images_vg = count_images("data/train/vangogh")
    train_images_nvg = count_images("data/train/not_vangogh")
    total_train = train_images_vg + train_images_nvg
    print(f"Total Train Images= {total_train}")
    return total_train


def check_test_count():
    """
    checks test image count
    :return:
    """
    test_images_vg = count_images("data/test/vangogh")
    test_images_nvg = count_images("data/test/not_vangogh")
    total_test = test_images_vg + test_images_nvg
    print(f"Total Test Images = {total_test}")
    return total_test


def check_valid_count():
    """
    checks valid image count
    :return:
    """
    valid_images_vg = count_images("data/valid/vangogh")
    valid_images_nvg = count_images("data/valid/not_vangogh")
    total_valid = valid_images_vg + valid_images_nvg
    print(f"Total Valid Images= {total_valid}")
    return total_valid


def check_dataset_count():
    """
    Checks image counts for data and raw set and prints results in run window

    :return:
    """
    print("**** Check Raw ****")
    check_raw_count()
    print("**** Check valid ****")
    total_valid = check_valid_count()
    print("**** Check test****")
    total_test = check_test_count()
    print("**** Check train ****")
    total_train = check_train_count()
    total = total_valid + total_test + total_train
    print("")
    print(f"Total Train/Test/Valid Images= {total}")


def get_class(file_path="data/valid/"):
    """
    Gets class list from a filepath
    :param file_path:
    :return:
    """
    # Setup path for target directory
    target_directory = file_path
    print(f"Target directory: {target_directory}")

    # Get the class names from the target directory
    class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
    print(f"Class names found: {class_names_found}")


# get_class(file_path)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array
   """
    # resize to 256x256
    resized = image.resize((256, 256))
    height, width = resized.size

    # crop the center of the image
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image_crop = resized.crop((left, top, right, bottom))

    # normalize
    image_norm = np.array(image_crop) / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_norm = (image_norm - mean) / std

    # reorder for PIL
    image_trans = image_norm.transpose((2, 0, 1))
    return image_trans

# print(os.listdir("data/train"))
