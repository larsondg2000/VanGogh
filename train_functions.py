"""
Performs all the model training functions:
    1) Gets arg parse values
    2) Loads datasets and transforms
    3) Trains model using VGG19,  Efficient Net, RegNet
    4) Calculates train loss/accuracy and test loss/accuracy
    5) Saves model as a checkpoint
    6) User can set epochs, gpu/cpu, learn rate, hidden layers, save and data dir
    7) User can select model VGG19, EfficientNet, or RegNet to train
"""

# Imports here
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict
from matplotlib import pyplot as plt
import argparse
from torchinfo import summary


def args_parse():
    """
        Arg Parse Parameters
        - save_dir: is where the model is stored after training
        - arch: model type (vgg19 or AlexNet)
        - learning_rate: training learning rate
        - epochs: number of epochs for training
        - gpu: set to gpu (cuda) or cpu
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", action="store", dest="save_dir",
                        default="saved_models/")
    parser.add_argument('--data_dir', type=str, dest='data_dir', action="store", nargs="*", default="data")
    parser.add_argument('--model', dest='model', default='efficientnet_v2',
                        choices=['efficientnet_v2', 'vgg19', 'regnet'])
    parser.add_argument('--hidden_layers', type=int, dest='hidden_layers', default='4096')
    parser.add_argument('--learn_rate', type=float, dest='learn_rate', default='0.00033')
    parser.add_argument('--epochs', type=int, dest='epochs', default='10')
    parser.add_argument('--gpu', dest='gpu', action="store_true", default=False)
    return parser.parse_args()


def train_setup(model, epochs, gpu, learn_rate, hidden_layers, save_dir, data_dir):
    """
    Train Setup Function
        - sets up data transformer and loaders
        - calls model to be trained (VGG19, EfficientNet, RegNet)
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {

        'train': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),

        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

        ]),

        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # set batch size so easy to adjust later
    b_size = 64

    train_loader = DataLoader(image_datasets['train'], batch_size=b_size, shuffle=True)
    # valid_loader = DataLoader(image_datasets['valid'], batch_size=b_size)
    test_loader = DataLoader(image_datasets['test'], batch_size=b_size)

    # Call Training Model (saves model at end of training)
    if model == 'efficientnet_v2':
        model_e_net(epochs, learn_rate, train_loader, test_loader,
                    image_datasets, gpu, hidden_layers, save_dir, b_size)

    if model == 'vgg19':
        model_vgg(epochs, learn_rate, train_loader, test_loader,
                  image_datasets, gpu, hidden_layers, save_dir, b_size)

    if model == 'regnet':
        model_regnet(epochs, learn_rate, train_loader, test_loader,
                     image_datasets, gpu, hidden_layers, save_dir, b_size)

    return model, save_dir


def model_vgg(epochs, learn_rate, train_loader, test_loader, image_datasets, gpu, hidden_layers, save_dir, b_size):
    """
    VGG19 Model function
    :param epochs: configurable from command line
    :param learn_rate: configurable from command line
    :param train_loader:
    :param test_loader:
    :param image_datasets: train, test and valid datasets
    :param gpu: cpu or gpu if available
    :param hidden_layers: configurable from command line
    :param save_dir: saved model location (configurable from command line)
    :param b_size: batch size
    :return:
    """
    # Set for torchvision v0.13+ (for earlier versions use "pre-trained")
    model = models.vgg19(weights='VGG19_Weights.DEFAULT')

    # Freeze the layers/parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    """
    Create new classifier:
        - in_features 25088 
        - hidden_layers are configurable (default=4096)
        - dropout = .45
        - out_features = 2
    """
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_layers)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('output', nn.Linear(hidden_layers, 2)),
    ]))

    # Replace classifier with new classifier
    model.classifier = classifier

    model_sum = summary(model, input_size=(b_size, 3, 224, 224),
                        verbose=0,
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])

    # check the model set up
    print("=============================================  VGG19 Model  ==========================================")
    print("====================================== Weights=VGG19_Weights.DEFAULT =================================")
    print(f"Epochs = {epochs}   Learn Rate = {learn_rate}   Hidden Layers = {hidden_layers}   Batch Size = {b_size}")
    print("======================================================================================================")
    print("")
    print(model_sum)

    # Define loss and optimizer, lear_rate is configurable
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # Call train_model function
    train_model(model, epochs, criterion, optimizer, train_loader,
                test_loader, image_datasets, gpu, save_dir)

    return


def model_e_net(epochs, learn_rate, train_loader, test_loader, image_datasets, gpu, hidden_layers, save_dir, b_size):
    """
    EfficientNet Model function
    :param epochs:
    :param learn_rate:
    :param train_loader:
    :param test_loader:
    :param image_datasets:
    :param gpu:
    :param hidden_layers:
    :param save_dir:
    :param b_size:
    :return:
    """
    # Set for torchvision v0.13+ (for earlier versions use "pre-trained")
    model = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
    # print(model)

    # Freeze the layers/parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    """

    Create new classifier:
        - in_features 25088 
        - hidden_layers are configurable (default=4096)
        - dropout = .5
        - out_features = 2

      (classifier): Sequential(
        (0): Dropout(p=0.45, inplace=True)
        (1): Linear(in_features=1280, out_features=1000, bias=True)
    """
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1280, hidden_layers)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.45)),
        ('output', nn.Linear(hidden_layers, 2)),
    ]))

    # Replace classifier with new classifier
    model.classifier = classifier

    model_sum = summary(model, input_size=(b_size, 3, 224, 224),
                        verbose=0,
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])

    # check the model set up
    print("=========================================  EfficientNet Model  =======================================")
    print("================================ Weights=EfficientNet_V2_L_Weights.DEFAULT ===========================")
    print(f"Epochs = {epochs}    Learn Rate = {learn_rate}    Hidden Layers = {hidden_layers}   Batch Size = {b_size}")
    print("======================================================================================================")
    print("")
    print(model_sum)

    # Define loss and optimizer, lear_rate is configurable (default=.001)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # Call train_model function
    train_model(model, epochs, criterion, optimizer, train_loader,
                test_loader, image_datasets, gpu, save_dir)

    return


def model_regnet(epochs, learn_rate, train_loader, test_loader, image_datasets, gpu, hidden_layers, save_dir, b_size):
    """
    RegNet Model function
    :param epochs:
    :param learn_rate:
    :param train_loader:
    :param test_loader:
    :param image_datasets:
    :param gpu:
    :param hidden_layers:
    :param save_dir:
    :param b_size:
    :return:
    """
    # Set for torchvision v0.13+ (for earlier versions use "pre-trained")
    model = models.regnet_y_16gf(weights='RegNet_Y_16GF_Weights.DEFAULT')

    # Freeze the layers/parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    """
    Create new classifier (fc):
        - in_features 3024
        - hidden_layers are configurable 
        - dropout = .45
        - out_features = 2

    """
    fc = nn.Sequential(OrderedDict([
        ('fc', nn.Linear(3024, hidden_layers)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),
        ('output', nn.Linear(hidden_layers, 2)),
    ]))

    # Replace classifier with new classifier
    model.fc = fc

    model_sum = summary(model, input_size=(b_size, 3, 224, 224),
                        verbose=0,
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])

    # check the model set up
    print("============================================= RegNet Model ===========================================")
    print("====================================== Weights=regnet_y_16gf.DEFAULT =================================")
    print(f"Epochs = {epochs}   Learn Rate = {learn_rate}   Hidden Layers = {hidden_layers}  Batch Size = {b_size}")
    print("======================================================================================================")
    print("")
    print(model_sum)

    # Define loss and optimizer, lear_rate is configurable
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learn_rate)

    # Call train_model function
    train_model(model, epochs, criterion, optimizer, train_loader,
                test_loader, image_datasets, gpu, save_dir)

    return


def train_model(model, epochs, criterion, optimizer, train_loader,
                test_loader, image_datasets, gpu, save_dir):
    """
    Trains function, gets train and test loss and accuracy results for plotting
    :param model: VGG19, RegNet, or EffecientNet
    :param epochs: configurable from command line
    :param criterion: loss function
    :param optimizer: optimizer used
    :param train_loader:
    :param test_loader:
    :param image_datasets:
    :param gpu:
    :param save_dir:
    :return:
    """
    # Create empty list to capture loss/accuracy values for all steps for plotting
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_accuracy_list = []
    print_every = 5

    # Training steps, each loop is one epoch
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # Call train_step functions, get back updated loss/accuracy/steps list
        train_loss_list, test_loss_list, train_acc_list, test_accuracy_list = \
            train_step(model, train_loader, test_loader, gpu, criterion, optimizer, epoch, epochs,
                       train_loss_list, test_loss_list, train_acc_list, test_accuracy_list, print_every)

    # plots test/train loss and test/train accuracy over the steps once all epochs complete
    plot_loss_curves(train_loss_list, test_loss_list, train_acc_list, test_accuracy_list, print_every)

    # gets model name and adds model name to checkpoint.pth filename
    model_name = model.__class__.__name__
    print(f"model name = {model_name}")

    # Updates model save path
    model_dir = save_dir + model_name + '_checkpoint.pth'

    # Call save_model and saves to saved_model folder as filename {model name}_checkpoint.pth
    if model_name == 'VGG':
        save_model(model, epochs, optimizer, image_datasets, model_dir)

    if model_name == 'EfficientNet':
        save_model(model, epochs, optimizer, image_datasets, model_dir)

    if model_name == 'RegNet':
        save_model_fc(model, epochs, optimizer, image_datasets, model_dir)

    return


def train_step(model, train_loader, test_loader, gpu, criterion, optimizer, epoch, epochs,
               train_loss_list, test_loss_list, train_acc_list, test_accuracy_list, print_every):
    """
    Training step
    :param model:
    :param train_loader:
    :param test_loader:
    :param gpu:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param epochs:
    :param train_loss_list:
    :param test_loss_list:
    :param train_acc_list:
    :param test_accuracy_list:
    :param print_every:
    :return:
    """
    # initialize parameters
    running_loss = 0
    running_acc = 0
    train_size_total = 0
    steps = 0

    model.train()

    # send to cuda or cpu
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()

    for inputs, labels in train_loader:
        steps += 1

        # Move input and label tensors to the (default device= cpu)
        if gpu and cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        else:
            inputs = inputs.cpu()
            labels = labels.cpu()

        # forward pass
        outputs = model(inputs)

        # Calculate train loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # zero grads
        optimizer.zero_grad()

        # loss back
        loss.backward()

        # optimizer step
        optimizer.step()

        # Calculate and accumulate correct predictions
        y_pred_class = outputs.argmax(dim=1)
        running_acc += (y_pred_class == labels).sum().item()

        # Get total size of test data set by getting size of batches
        # ***last batch in epoch might not equal batch size
        size_batch = inputs.size(0)

        # calculates total number of training inputs/labels per epoch
        train_size_total += size_batch

        if steps % print_every == 0:
            # call test_eval function using cross entropy loss calc.
            test_loss, test_accuracy = test_eval(model, test_loader, criterion, gpu)

            # Calculate train loss
            train_loss = running_loss / print_every
            train_acc = running_acc / train_size_total

            # Print results every "print_every" times
            print(f"Epoch: {epoch + 1}/{epochs} "
                  f"Train Loss: {train_loss:.3f} "
                  f"Train Accuracy: {train_acc * 100:.1f}% "
                  f"Test Loss {test_loss:.3f} "
                  f"Test Accuracy: {test_accuracy * 100:.1f}% "
                  )

            # Updates loss and accuracy list values for epoch
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            train_acc_list.append(train_acc)
            test_accuracy_list.append(test_accuracy)

            # Resets values to zero and sets model back to train
            running_loss = 0
            running_acc = 0
            train_size_total = 0
            model.train()

    # return updated list accuracy/loss values along with steps at end of each epoch
    return train_loss_list, test_loss_list, train_acc_list, test_accuracy_list


# Test eval run and calculations using CE loss
def test_eval(model, dataloader, loss_fn, gpu):
    """
    Test evaluation function
    :param model:
    :param dataloader:
    :param loss_fn:
    :param gpu:
    :return:
    """
    # Set to eval mode
    model.eval()

    # resets variables to zero
    total_test_correct = 0
    test_size_total = 0
    count = 0
    test_loss_running = 0

    # Turn on inference context manager (sim to no_grad())
    with torch.inference_mode():

        # Loop through test images
        for batch, (inputs, labels) in enumerate(dataloader):

            # Move input and label tensors to the (default device= cpu)
            cuda = torch.cuda.is_available()
            if gpu and cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            # calculates number of passes for loss average calculation
            count += 1

            # Forward pass
            test_predicted_logits = model(inputs)

            """
            Calculate and accumulate loss using CrossEntropy loss function
            - CE takes softmax of outputs and 
            - CE calculates mean of the losses per batch
            - Basically same as NLLLoss except NLLLoss softmax function needs to be in the output of the classifier
            """
            loss = loss_fn(test_predicted_logits, labels)

            # keeps running total of batch loss
            test_loss_running += loss.item()

            # Calculate and accumulate correct predictions
            test_correct = test_predicted_logits.argmax(dim=1)
            total_test_correct += ((test_correct == labels).sum().item())

            # Get total size of test data set by getting size of batches
            # ***last batch might not equal batch size
            size_batch = inputs.size(0)
            test_size_total += size_batch

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss_running / count
    test_acc = total_test_correct / test_size_total

    # returns test loss and accuracy
    return test_loss, test_acc


# Saves the model, directory is configurable
def save_model(model, epochs, optimizer, image_datasets, model_dir):
    """
    saves VGG19 and EfficientNet models and state dict.
    :param model:
    :param epochs:
    :param optimizer:
    :param image_datasets:
    :param model_dir:
    :return:
    """
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'epochs': epochs,
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'arch': model}

    torch.save(checkpoint, model_dir)

    return


def save_model_fc(model, epochs, optimizer, image_datasets, model_dir):
    """
    Saves RegNet model
    :param model:
    :param epochs:
    :param optimizer:
    :param image_datasets:
    :param model_dir:
    :return:
    """
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'state_dict': model.state_dict(),
        'fc': model.fc,
        'epochs': epochs,
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'arch': model}

    torch.save(checkpoint, model_dir)

    return


def plot_loss_curves(train_loss, test_loss, train_acc, test_accuracy, print_every):
    """Plots training and test loass and accuracy results.

    Args:
            train_loss
            train_acc
            test_loss
            test_accuracy
            print_every
    """
    # set values for x-axis
    step_list = []

    for x in range(1, len(train_loss) + 1):
        step_list.append(int(print_every * x))

    max_loss = max(train_loss + test_loss)

    # Set size
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(step_list, train_loss, label="train_loss")
    plt.plot(step_list, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Batch")
    plt.xlim([min(step_list), max(step_list)])
    plt.ylim([0, max_loss + .2])
    plt.xticks(step_list)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(step_list, train_acc, label="train_accuracy")
    plt.plot(step_list, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Batch")
    plt.ylim([0, 1])
    plt.xlim([min(step_list), max(step_list)])
    plt.xticks(step_list)
    plt.legend()
    plt.show()
