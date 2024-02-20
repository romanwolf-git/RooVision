# Project: Kangaroo Detection and Classification
The project aims to detect and classify species of the Macropodidae family native to Victoria, Australia :australia:. The Macropodidae are a family of marsupials that includes kangaroos :kangaroo:, wallabies :kangaroo: and wallaroos :kangaroo:.
## Table of Content
- [Overview](#overview)
- [Installation](#installation)
- [About the Data](#about-the-data)
- [Data Acquisition](#data-acquisition)
- [Extract from the Classification Report](#classfication-report)
- [Methods](#methods)
- [Web Application](#webpage)
- [Acknowledgements & Licensing](#acknowledgements--licensing)

## Overview <a name="overview"/>

When enjoying Victoria's wildlife, you will often come across species from the Macropodidae family. However, it is relatively difficult for the untrained eye to tell which species you are looking at. The aim of the project is to train an algorithm, which will then be used in an app on the camera of a mobile device, to detect and classify the species of an animal in the family in real time. In other words, the app will help nature lovers understand which "kangaroo" they are seeing in the wild.

## Installation <a name="installation"/>
If you want to run the application on your local machine, follow the installation steps below. Otherwise, the application can be accessed from www.xxx.xxx.
1. Clone repository and change to its directory.
```
git clone https://github.com/romanwolf-git/disaster_response
```

2. Install requirements.txt. Mac users may also need 'brew install cmake libomp'.
```
pip install -r requirements.txt
```
3. Change to 'data' directory and run 'process_data.py' to clean the data and store it in a database.
```
python process_data.py messages.csv categories.csv disaster_response.db
```
4. Change to 'models' directory and run 'data_pipeline.py' to train, tune and save the model.
```
python data_pipeline.py disaster_response.db
```
5. Run 'run.py' which outputs: "Serving Flask app 'run'" and the URL (http://127.0.0.1:3000) where its run. Open the URL in your browser.
```
python run.py
```

## About the Data <a name="about-the-data"/>
The model was trained using data from the Atlas of Living Australia :australia:. To avoid class imbalance, 100 images of each :kangaroo: species were used:
- 61 Bridled nail-tail wallaby (_Onychogalea fraenata_)
- 100 Brush-tailed rock-wallaby (_Petrogale penicillata_)
- 100 Eastern grey kangaroo (_Macropus giganteus_)
- 100 Red kangaroo (_Ospranter rufus_)
- 100 Red-necked wallaby (_Notamacropus rufogriseus_)
- 100 Swamp wallaby (_Wallabia bicolor_)
- 100 Western grey kangaroo (_Macropus fuliginosus_)

Unfortunately there are not many images of the bridled nail-tail wallaby (_Onychogalea fraenata_) in the database, so there are only 61 images of this species in the dataset.

A critical aspect of this dataset is that it is not always certain that the animal in the image is the species indicated. As the [Atlas of Living Australia](https://www.ala.org.au/) is an open source database, anyone can upload images and incorrectly identify species. Where obvious errors have been made, images have been removed from the dataset and replaced.

It is also important to note that some of the images in the database have found species in regions where they are not expected to occur. The decision on which species to include in the dataset is therefore based on information from the [Australian Department of Climate Change, Energy, Environment and Water](https://biodiversity.org.au/afd/search/names/).

The necessary annotation of the images was done on [roboflow](https://roboflow.com/), which made the process relatively quick and easy. The annotated dataset was downloaded to the local machine for training.

A notebook for more detailed exploratory data analysis can be found here.
## Methods <a name="methods"/>
The Python script 'process_data.py' is used to clean the data and save it to a SQLite database.

The Python script 'data_pipeline.py' loads the data from the database, builds a model, trains and tunes the model with resampled data, outputs a classification report, and saves the model in a serialized pickle binary. 
generates the machine learning model.

The model consists of scikit-learn's CountVectorizer, TFDIFTransformer and LightGBM's classifier. Resampling is done using the Multilabel Synthetic Minority Over-sampling Technique ([multi-label SMOTE](https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote)) with 500 additional training samples.

One of the project requirements is the use of a scikit-learn pipeline. However, resampling within a pipeline is only possible using imblearn, which currently does not support multi-labeling.

## Data Acquisition <a name="data-acquisition"/>
Although most of the images in the database are not very large, downloading them synchronously is time-consuming :hourglass_flowing_sand:. Therefore, asynchronous programming is used to download the required data. This has reduced the data acquisition time to just under 10 minutes :rocket: for about 1 GB of data. This was done using the [asyncio](https://docs.python.org/3/library/asyncio.html), [aiofiles](https://pypi.org/project/aiofiles/) and [aiohttp](https://docs.aiohttp.org/en/stable/) libraries. The data retrieval script can be found here.

## Extract from the Classification Report <a name="classification-report"/>
| Average      | Precision | Recall | F1-Score   |
|--------------|-----------|--------|------------|
| Micro Avg    | 0.7929    | 0.6267 | **0.7001** |
| Macro Avg    | 0.5918    | 0.354  | **0.4208** |
| Weighted Avg | 0.7554    | 0.6267 | 0.6632     |
| Samples Avg  | 0.6339    | 0.5259 | 0.5312     |

## App <a name="webpage"/>
Screenshots from the application:
1. Front screen/dashboard of the application with:
   - Navigation bar,
   - Message input field
   - button for message classification,
   - 2 Plotly plots for training data overview.
![Fron screen/dashboard](https://github.com/romanwolf-git/disaster_response/blob/main/images/screenshot_app.png)

2. Message classification target page with selected categories
![Message classification target page](https://github.com/romanwolf-git/disaster_response/blob/main/images/screenshot_classification.png)

## Acknowledgements & Licensing <a name="acknowledgements--licensing"/>
Thanks to Figure Eight Inc. for providing the data and to Udacity for providing the course and support.
