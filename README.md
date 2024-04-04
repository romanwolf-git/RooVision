# Project: Kangaroo Detection and Classification
The project aims to detect and classify species of the _Macropodidae_ family native to Victoria, Australia :australia:. The _Macropodidae_ are a family of marsupials that includes kangaroos :kangaroo:, wallabies :kangaroo: and wallaroos :kangaroo:.
## Table of Content
- [Motivation](#motivation)
- [Installation](#installation)
- [Libraries](#libraries)
- [Methods](#methods)
- [About the Data](#about-the-data)
- [Validation Results](#validation-results)
- [Analysis](#analysis)
- [Files](#files)
- [Acknowledgements](#acknowledgements)

## Motivation for the Project <a name="motivation"/>

When enjoying Victoria's wildlife, you will often come across species from the Macropodidae family. However, it is relatively difficult for the untrained eye to tell which species you are looking at. The aim of the project is to train an algorithm, which will then be used in an app on the camera of a mobile device, to detect and classify the species of an animal in the family in real time. In other words, the app will help nature lovers understand which "kangaroo" they are seeing in the wild.

## Installation <a name="installation"/>
If you want to run the application on your local machine, follow the installation steps below. Otherwise, the application can be accessed from https://roovision-6bnwkndfqa-km.a.run.app/.
1. Clone repository and change to its directory.
```
git clone https://github.com/romanwolf-git/RooVision
```

2. Install requirements.txt.
```
pip install -r requirements.txt
```
3. Change to `app` directory and run `wsgi.py` to start the app.
```
python wsgi.py
```
4. Click on the link or open a window in your browser at http://127.0.0.1:8080.
```
[[python data_pipeline.py disaster_response.db](http://127.0.0.1:8080)](http://127.0.0.1:8080)
```
5. Navigate to the prediction section and upload an image, set confidence and overlap, and infer the macropod species.
![Screenshot1](https://github.com/romanwolf-git/RooVision/blob/main/app/app_preview/Screenshot%202024-04-03%20at%2014-47-23%20Prediction.png)
## Libraries <a name="libraries"/>
The project used the 'galah' Python API from the Atlas of Living Australia to retrieve data. To improve download speed and use asynchronous programming, the asyncio, aiofiles and aiohttp libraries were used.

PyTorch, in particular its computer vision module Torch Vision, is used extensively in the project for data science tasks. YOLO uses albumentations for data preprocessing, OpenCV for image processing tasks, and Pillow for additional image manipulation capabilities. Results are visualized using matplotlib and seaborn.

Annotations were performed using Roboflow, which offers a convenient workflow management client. Ultralytics also provides its own client for seamless integration of YOLO.

Flask is the web framework used to develop the application, with extensions for forms (Flask-WTF), sessions (Flask-Session, redis) and uploads (Flask-Uploads). To ensure secure management of application secrets, Google Cloud Secret Manager is used.

Furthermore, an effort is made to keep Docker container sizes to a minimum, typically achieved by selecting lightweight images or implementing multi-stage builds. However, PyTorch requires several CUDA dependencies, which can make container images quite large. To ensure optimal performance and resource utilization, these dependencies may be pruned based on the allocated cloud resources for container deployment.

Additionally, smaller utility libraries have been used, although their specific functionality is not described.
## Methods <a name="methods"/>
The project methodology involved several key steps. Firstly, the [Atlas of Living Australia](https://www.ala.org.au/), an open database, was identified as a source of images relevant to the species of interest. Next, images relevant to the species of interest were downloaded using a custom script. Finally, we manually inspected the acquired data to identify and rectify any apparent anomalies. The characteristics of the image dataset were then explored using exploratory data analysis (EDA).

A subset of images suitable for annotation was selected based on pre-defined criteria to facilitate annotation. To make the annotation process faster and easier, these images were uploaded to Roboflow, a platform that provides fast annotation capabilities and additional features. An automated machine learning algorithm was also trained to speed up the annotation process. The data was split into training, validation, and test sets in a 70-20-10 ratio using a stratified approach.

Local model training was established, incorporating preprocessing steps and augmentation techniques to improve model performance. Two popular architectures, Faster R-CNN and YOLOv9, were trained locally for 25 epochs each. The training results were evaluated based on the mean average precision (mAP), and the superior architecture was determined for further training.

The model was trained on Google Colab, a cloud-based platform, for 300 epochs to take leverage of its computational resources. The training process was regularly monitored to ensure there were no overfitting or underfitting problems. Once the training was successful, the model was validated to assess its performance on unseen data.

The trained model's results, including training and validation metrics, and the model weights were downloaded for further analysis after validation. To deploy the model, a Flask application skeleton was coded using an application factory approach. Additionally, front-end functionality was integrated using JavaScript, and the trained model was seamlessly integrated into the Flask application to provide a user-friendly interface for end users.
## About the Data <a name="about-the-data"/>
The model was trained using data from the Atlas of Living Australia :australia:. To avoid class imbalance, 100 images of each :kangaroo: species were used:
- 64 Bridled nail-tail wallaby (_Onychogalea fraenata_)
- 100 Brush-tailed rock-wallaby (_Petrogale penicillata_)
- 100 Eastern grey kangaroo (_Macropus giganteus_)
- 100 Red kangaroo (_Ospranter rufus_)
- 100 Red-necked wallaby (_Notamacropus rufogriseus_)
- 100 Swamp wallaby (_Wallabia bicolor_)
- 100 Western grey kangaroo (_Macropus fuliginosus_)
- 100 background images without macropods.

Unfortunately there are not many images of the bridled nail-tail wallaby (_Onychogalea fraenata_) in the database, so there are only 64 images of this species in the dataset.

A critical aspect of this dataset is that it is not always certain that the animal in the image is the species indicated. As the [Atlas of Living Australia](https://www.ala.org.au/) is an open source database, anyone can upload images and incorrectly identify species. Where obvious errors have been made, images have been removed from the dataset and replaced.

It is also important to note that some of the images in the database have found species in regions where they are not expected to occur. The decision on which species to include in the dataset is therefore based on information from the [Australian Department of Climate Change, Energy, Environment and Water](https://biodiversity.org.au/afd/search/names/).

The necessary annotation of the images was done on [Roboflow](https://roboflow.com/), which made the process relatively quick and easy. The annotated dataset was downloaded to the local machine for training.

A notebook for more detailed exploratory data analysis [can be found here](https://github.com/romanwolf-git/RooVision/blob/main/notebooks/exploratory_data_analysis.ipynb).
## Validation Results <a name="validation-results"/>
The data was validated by partitioning 20% of it. The selected subset represents the different species in the same proportion as the training and testing data, thus validating its utility and reliability under real-world conditions.

| Class                      |  Images  |  Instances  |  Precision  |  Recall  |  mAP50  |               mAP50-95               |
|----------------------------|:--------:|:-----------:|:-----------:|:--------:|:-------:|:------------------------------------:|
| all                        |   153    |     146     |    0.878    |  0.874   |  **<font color="green">0.926</font>**  | **<font color="green">0.784</font>** |
| Bridled-nail-tail wallaby |   153    |     12      |    1.000    |  0.759   |  0.917  |                0.853                 |
| Brush-tailed rock-wallaby  |   153    |     21      |    0.867    |  1.000   |  0.990  |                0.800                 |
| Eastern grey kangaroo      |   153    |     23      |    0.714    |  0.867   |  0.904  |                0.785                 |
| Red kangaroo               |   153    |     24      |    1.000    |  0.846   |  0.938  |                0.793                 |
| Red-necked wallaby         |   153    |     20      |    0.863    |  0.850   |  0.899  |                0.714                 |
| Swamp wallaby              |   153    |     21      |    0.984    |  0.952   |  0.989  |                0.826                 |
| Western grey kangaroo      |   153    |     25      |    0.718    |  0.840   |  0.846  |                0.720                 |
## Analysis <a name="analysis"/>
The validation report indicates that the model's performance is encouraging across the different classes of Victorian macropods. The model demonstrates high precision, particularly in distinguishing between different species, achieving an overall precision of 0.878. However, there are variations in recall rates across classes, with certain species exhibiting lower recall rates than others. For example, the model achieves perfect precision for Bridled-nail-tail wallabies and Red kangaroos, but its recall for the Eastern grey kangaroo is comparatively lower at 0.867. 

The mean Average Precision (mAP) metrics highlight the model's effectiveness, particularly its robust performance with a mAP50 of 0.926 and mAP50-95 of 0.784 across all classes. The Brush-tailed rock-wallaby class has demonstrated exceptional performance, with a precision of 0.867 and perfect recall, resulting in a mAP50 of 0.990.

However, there is still room for improvement to ensure consistent and high-quality detections across all species. Further refinement and adjustments to the model could enhance its detection accuracy and address challenges associated with lower recall rates in certain classes.
## Conclusion <a name="conclusion"/>
The YOLOv9 model architecture is capable of detecting and classifying macropods in Victoria, Australia. Its performance analysis shows strong results across various classes, achieved with as few as 100 annotated images per class.

However, to enhance the model's performance, additional high-quality data is required to optimize its training. Ideally, YOLO should be trained on around 1,500 images per class or 10,000 annotations for optimal performance. However, because of limited data availability and computational constraints, a smaller dataset was used in this project.

To improve the model's performance on smaller objects, it is recommended to train the model at higher resolutions, such as 1280x1280 pixels compared to the 640x640 used. Additionally, the number of parameters in the model is minimised to reduce resource consumption and training time, although increasing the number of parameters typically improves performance at the expense of resource consumption and training time.

Manual inspection of misclassified images provides valuable insights into the model's weaknesses, enabling targeted adjustments to preprocessing techniques and augmentation strategies. Additionally, optimizing hyperparameters, such as selecting appropriate optimizers, enhances model training efficiency and can improve performance metrics such as mean average precision, precision, and recall.

As the model is integrated into a Flask web application, improving the overall project impression involves enhancing user experience through additional functionality and conducting A/B testing to evaluate new features' impact. A promising next step is to incorporate real-time macropod prediction via the user's camera using WebRTC in the app's frontend with JavaScript, leveraging YOLO's rapid inference capabilities.
## Files <a name="files"/>
Due to the large number of files, not all are listed in the overview. Only directory descriptions are provided where deemed sufficient. The file names are typically descriptive, often containing additional information about their content. The project is structured as follows:
```
├─app
│ ├─application             # Flask application directory
│ │ ├───index               # index directory registered as blueprint
│ │ │   ├───static
│ │ │   │     ├─css         # custom css styles
│ │ │   │     └─js          # custom javascript scripts
│ │ │   ├───templates       # jinja2 templates
│ │ │   └───routes.py       # routes for this blueprint
│ │ ├───prediction          # prediction directory registered as blueprint
│ │ │   ├───static
│ │ │   │     ├─img_test    # directory for test images
│ │ │   │     ├─img_upload  # directory for uploaded images
│ │ │   │     ├─img_output  # directory for output images (predictions)
│ │ │   │     ├─js          # directory for custom javascript scripts
│ │ │   │     └─models      # directory for pickled models
│ │ │   ├───templates       # jinja2 templates
│ │ │   └───routes.py
│ │ ├───data                # data directory registered as blueprint
│ │ │   ├───static
│ │ │   ├───templates       # jinja2 templates
│ │ │   └───routes.py
│ │ ├───model               # model directory registered as blueprint
│ │ │   ├───static
│ │ │   ├───templates      # jinja2 templates
│ │ │   └───routes.py
│ │ └───results             # results directory registered as blueprint
│ │     ├───static
│ │     │     ├─img         # directory for images
│ │     │     └─js          # directory for custom javascript scripts
│ │     ├───templates       # jinja2 templates
│ │     └───routes.py  
│ ├─__init__.py             # creates the app object
│ ├─config.py               # Flask configuration
│ ├─requirements.txt        # requirements for the app
│ └─wsgi.py                 # entrypoint to the app
├─data
│ └─images                  # annotated images
│   ├─test                  # test images and labels
│   │  ├─images         
│   │  └─labels
│   ├─train                 # train images and labels
│   │  ├─images         
│   │  └─labels
│   ├─valid                 # validation images and labels
│   │  ├─images         
│   │  └─labels
│   └─data.yaml             # YAML-file for YOLOv9 training
├─notebooks                 # notebook for exploratory data analysis 
└─src
  ├─data_retrieval          # scripts for data retrieval
  ├─faster_rcnn             # scripts for running Faster R-CNN
  │ ├─data_augmentation     # scripts for data augmentation
  │ ├─data_visualization    # scripts for data visualization
  │ ├─model_architecture    # scripts for the model architecture
  │ ├─model_inference       # scripts for model inference
  │ ├─model_training        # scripts for model training
  │ └─utils                 # scripts for data visualization
  └─yolov9                  # cloned from https://github.com/SkalskiP/yolov9 
```

## Acknowledgements <a name="acknowledgements"/>
Special thanks to the Atlas of Living Australia for providing the image dataset used in this project. Additionally, gratitude goes to [Piotr Skalski](https://github.com/SkalskiP) for the provision of the YOLOv9 fork, based on the work originally developed by [Wong Kin-Yiu](https://github.com/WongKinYiu). Their contributions have been invaluable to the success of this project.

