import torch
from typing import List
import argparse
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path, PurePath
import random


def parse_args():
    """
        Arg Parse Parameters
        - checkpoint is where the saved model is stored
        - class names: gets class names
        - filepath: path of image to check
        - custom: individual image to predict
        - gpu: set to gpu (cuda) or cpu
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', dest='checkpoint', action='store',
                        default='saved_models/RegNet_checkpoint.pth')
    parser.add_argument('--class_names', type=str, dest='class_names', default='data/valid/')
    parser.add_argument('--custom', dest='custom_image_path', default="data/valid/vangogh/f_0790.jpg")
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=False)

    return parser.parse_args()


def load_checkpoint(model_checkpoint):
    """ Loads saved model checkpoint """
    saved_model = torch.load(model_checkpoint)
    model = saved_model['arch']

    # Get the model size in bytes then convert to megabytes
    model_size = Path(model_checkpoint).stat().st_size // (1024 * 1024)
    print(f"model size: {model_size} MB")

    return model


def get_class(file_path="data/valid/"):
    """ Gets list of classes """
    # Setup path for target directory
    target_directory = file_path

    # Get the class names from the target directory
    class_names = sorted([entry.name for entry in list(os.scandir(target_directory))])

    return class_names


def predict_image(model, custom_image_path, class_names, gpu):
    """ predicts an individual image """
    pred_and_plot(model=model,
                  image_path=custom_image_path,
                  class_names=class_names,
                  transform=None,
                  gpu=gpu)


def predict_random(filepath, model, class_names, gpu):
    """ randomly selects images from filepath and predicts/prints selected
    images
    """
    # Get a random list of 3 images from valid folder
    num_images_to_plot = 5
    # print(os.listdir(filepath))
    image_list = list(Path(filepath).glob("*/*.jpg"))
    print(len(image_list))

    test_image_path_sample = random.sample(population=image_list,
                                           k=num_images_to_plot)
    print(test_image_path_sample)

    # Iterate through random test image paths, make predictions on them and plot them
    for image_path in test_image_path_sample:
        pred_and_plot(model=model,
                      image_path=image_path,
                      class_names=class_names,
                      transform=None,
                      gpu=gpu)


def check_valid(filepath, model, class_names, gpu):
    """ predicts validation dataset"""
    image_list = list(Path(filepath).glob("*/*.jpg"))

    # Iterate through random test image paths, make predictions on them and plot them
    for item in image_list:
        item_string = str(item)

        pred_and_print(model=model,
                       image_path=item_string,
                       class_names=class_names,
                       transform=None,
                       gpu=gpu)


def pred_and_plot(model, image_path, class_names, transform, gpu):
    """
    Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        transform (torchvision.transforms, optional): Transform to perform on image.
        Defaults to None which uses ImageNet normalization.
        gpu: cuda or cpu
   """

    # Open image

    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Predict on image

    # Make sure the model is on the target device
    cuda = torch.cuda.is_available()

    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image
        # (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Move input and label tensors to the (default device= cpu)
        if gpu and cuda:
            transformed_image = transformed_image.cuda()

        else:
            transformed_image = transformed_image.cpu()

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image)

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # check if prediction is correct
    image_is = check_prediction(image_path, target_image_pred_label, class_names)

    # Get model name
    model_name = model.__class__.__name__

    # Plot image with predicted label and probability
    plt.figure()

    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max() * 100:.1f}%"
        f" | Prediction is {image_is}"
    )
    plt.suptitle(f"Model: {model_name}")
    plt.axis(False)
    plt.show()


def pred_and_plot_image(model, image_path, class_names, transform, gpu):
    """
    Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        transform (torchvision.transforms, optional): Transform to perform on image.
        Defaults to None which uses ImageNet normalization.
        gpu: cuda or cpu
   """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Predict on image

    # Make sure the model is on the target device
    cuda = torch.cuda.is_available()

    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image
        # (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Move input and label tensors to the (default device= cpu)
        if gpu and cuda:
            transformed_image = transformed_image.cuda()

        else:
            transformed_image = transformed_image.cpu()

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image)

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Get model name
    model_name = model.__class__.__name__

    # image_is = check_prediction(image_path, target_image_pred_label, class_names)

    # Plot image with predicted label and probability
    plt.figure()

    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max() * 100:.1f}%"
    )
    plt.suptitle(f"Model: {model_name}")
    plt.axis(False)
    plt.show()


def pred_and_print(model, image_path, class_names, transform, gpu):
    """
    Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        transform (torchvision.transforms, optional): Transform to perform on image.
        Defaults to None which uses ImageNet normalization.
        gpu: cuda or cpu
   """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Predict on image

    # Make sure the model is on the target device
    cuda = torch.cuda.is_available()

    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image
        # (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Move input and label tensors to the (default device= cpu)
        if gpu and cuda:
            transformed_image = transformed_image.cuda()

        else:
            transformed_image = transformed_image.cpu()

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image)

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # check prediction
    image_is = check_prediction(image_path, target_image_pred_label, class_names)

    print(f"{class_names[target_image_pred_label]}"
          f" |{target_image_pred_probs.max() * 100:.1f}% "
          f"|Pred. is {image_is}")


def check_prediction(image_path, target_image_pred_label, class_names):
    # check if prediction is correct
    p = PurePath(image_path)
    p_tuple = p.parts
    image_is = "correct"

    for item in p_tuple:
        if item == 'not_vangogh' or item == 'vangogh':
            image_class = item

            if image_class == class_names[target_image_pred_label]:
                image_is = "correct"
            else:
                image_is = "not correct"
            break
        else:
            image_is = "unknown class"

    return image_is
