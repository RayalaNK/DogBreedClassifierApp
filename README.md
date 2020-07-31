# DogBreedClassifierApp

##  Overview


The objective of this application is to primarily predict whether a given image is of a dog or of a human using Convolutional Neural Networks. If it's either of them, the application will need to further determine the kind of breed the dog is, or which dog breed the human most resembles. When the given image is neither a dog nor a human, predicting the breed wouldn't execute!


**Jupyter Notebook**

- Interpreter: Python 3.6+
- Web : flask, plotly
- Processing : numpy, pandas, scikit-learn, NLTK, pickle, re
- DB : SQLalchemy


**File Descriptions**

 **folder/files:**
- dog_app.ipynb: File containing message categories
- haarcascades/haarcascade_frontalface_alt: Haar feature-based cascade classifiers for face detection
- images folder: Images for the notebook and to test the algorithm are found here
- data/dog_names.csv: Contains names of the dog 

 **model:**
- saved_models folder: contains the trained models such as weights.best.VGG19.hdf5, weights.best.VGG16.hdf5 and weights.best.from_scratch.hdf5 

 **app:**
- run.py: Runs the Flask web app
- templates: HTML files for web app


**Usage**

Clone the repo using:
> git clone https://github.com/RayalaNK/DisasterResponsePipeline.git

Run the following command to run your web app after navigating into the app directory.
> cd web_app

> python app.py

Go to the URL:

> http://0.0.0.0:3001/ or http://localhost:3001

Upload an image of a dog or a human to predict the breed or resemblence respectively:

**Acknowledgements**

> Figure Eight for providing pre-labeled messages dataset

**Copyright and License**
> Code is released under the MIT License.
