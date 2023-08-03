# Van Gogh Painting Classification using Pytorch and Pre-trained Models

## Models
The project uses three pre-trained models:
  1) VGG19 (weights='VGG19_Weights.DEFAULT')
  2) RegNet (weights='RegNet_Y_16GF_Weights.DEFAULT')
  3) EfficientNet (weights='EfficientNet_V2_L_Weights.DEFAULT')

The models have pre-trained weights with a trainable classifier. It uses three python files to train the models:
  1) `train.py`: main() function call
  2) `train_functions.py`: all the required training functions
  3) `utils.py`: get_class()  function to get class list 

## Model Parameters
* Optimizer: Adam
* Loss Function : CrossEntropyLoss
* Epochs: configurable using CLI
* learning Rate: configurable using CLI
* Hidden Layers: configurable using CLI
* Dropout: set to .2 to .5 (depending on model)
* Batch Size: b_size set to 64
* Save Directory: configurable using CLI (saves as "model name"_checkpoint.pth.)

## Accuracy
All three models achieved >90% accuracy during the validation runs after training.  The models results were fairly close with the Regnet model performing the best. 

## Image Data
Images are sorted into two classes: "Van Gogh" and "not Van Gogh".  The raw images folder has all the images and this is a breakdown of the images.

Image sources for this project:

    Van Gogh Images (total paintings/discarded/total in dateset): 605
        (http://www.vggallery.com/index.html)
        Paris (March 1886 - February 1888)- 225/17/208 paintings
        Saint-Remy (May 1889 - May 1890)- 143/4/139 paintings
        Auvers-sur-Oise (May 1890 - July 1890)- 76/1/75 paintings
        Arles (February 1888 - May 1889)- 186/4/182 paintings
        Totals: 630/26/604 (total paintings/discarded/total in dateset)

    Not Van Gogh  Images (random paintings from various time periods): 609
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

        The images were downloded from the National Gallery of Art:
        https://www.nga.gov/collection/collection-search.html
        
        Modern Art:
        https://www.saatchiart.com/paintings/modern

## Dataset
The raw images can be segregated into train, test, and vaild folders using the `setup.py` along with functions in the `utils.py` file.  The setup randomly assigns raw images to the folders based on the following percentages:
* Train 80%
* Test 15%
* Validation 5%

At the end of the setup, the count_images() function counts all the images in the train, test, and valid folders as a check.

For my training, testing, and validation I used the following folder structure:

Data\

  test\
  
    not_vangogh\
    
    vangogh\
  
  train\
  
    not_vangogh\
    
    vangogh\
  
  valid\
  
    not_vangogh\
    
    vangogh\

There are some additional painting images in the sample_paintings folder that are not classified.

## Predictions
There are three different prediction functions, each with a main() that calls various functions in the
`predict_functions.py` file:
  1) `predict.py`:  uses the valid datasets to check the accuracy of the model using the valid dataset.
  2) `predict_random.py`: randomly selects five images from valid folder and displays image along with probability, prediction, and correct/incorrect prediction.
  3) `predict_image.py`: used to check a single unclassified image.

The three predict modules load a saved model checkpoint.  The model checkpoint is configurable in the CLI.
    
## Utilities
There are some useful functions in the `utils.py` file:
  1) Get the images counts in various folders.
  2) Check images for proper extension (useful if downloading images from google).
  3) Get class list and class_to_idx dictionary.
  4) There is a function to process images (Note: this is automatically done in the predict_*.py files)
