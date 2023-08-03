# Van Gogh Painting Classification using Pytorch and Pre-trained Models

## Models
The project uses three pre-trained models:
  1) VGG19
  2) RegNet
  3) EfficientNet
The models have pre-trained weights with a trainable classidier. It uses two python files to train the models:
  1) train.py
  2) train_functions.py

The models use an Adam optimizer and CrossEntropy as the loss function.  The epochs, learning rate, and hidden layers are configurable using the CLI and arg parse.  
The batch size is set to 64 and it can be modified in the train-functions.py file or youi can modify argparse to make it configurable from the CLI.
At the end of training, the test and train accuracy and losses are ploteed and the model checkpoint is saved to the saved_models folder as "model name"_checkpoint.pth.

The three models achieved >90% accuracy during the validation runs after training.

## Image data
Images are sorted into two classes- Van Gogh and not Van Gogh.  The raw images folder has all the images and this is a breakdown of thwe images.

Image source for this project:

    Van Gogh Images (total paintings/discarded/total in dateset): 605
        (http://www.vggallery.com/index.html)
        Paris (March 1886 - February 1888)- 225/17/208 paintings
        Saint-Remy (May 1889 - May 1890)- 143/4/139 paintings
        Auvers-sur-Oise (May 1890 - July 1890)- 76/1/75 paintings
        Arles (February 1888 - May 1889)- 186/4/182 paintings
        Totals: 630/26/604 (total paintings/discarded/total in dateset)

    Not Van Gogh  Images (random painting images from various time periods): 609
        Baroque- 80
        German Expressionist- 13
        Impressionist- 77
        Neo-classic- 23
        Post-Impressionist- 80
        Realist- 86
        Renaissance- 62
        Rococo- 64
        Romantic- 84
        Modern- 40
        Total: 609

        National Gallery of Art:
        https://www.nga.gov/collection/collection-search.html
        Modern Art:
        https://www.saatchiart.com/paintings/modern

The raw images can be segragated into train, test, and vaild folders using the setup.py along with functions in the utils.py file.

For my taining, testing, and validation I used the following folder structure:

Data
  Test
    not_vangogh
    vangogh
  Train
    not_vangogh
    vangogh
  valid
    not_vangogh
    vangogh

There are also some additional paintings in the sample_paintings folder.

## Predict
There are three different predict functions:
  1) predict.py:  uses the valid datasets to check the accuracy of the models
  2) predict_random.py: randomly selects five images to check and displays image along with probability and prediction.
  3) predict_image.py: used to check a single image.

The three predict modules load a saved model checkpoint which is configurable in the CLI.
    
## Utils.py
There are some useful functions in the utils.py file:
  1) Get the images counts in various folders.
  2) Check images for proper extension (useful if downloadingg images from google).
  3) Get class list and class_to_idx dictionary.
  4) There is a function to process images (this is automatically done in the predict_*.py files)
