# ML-Hackathon-Techkriti-2021
**Techkriti** is an annual techfest conducted by **IIT Kanpur**. As a part of the fest, competitions are conducted is various themes like `Enterprenurial` , `Technology` , `Miscellaneous`. This year (March 14th, 2021) ML Hackathon is conducted as a part of Technology division.

I got `first position` (Team Name : `Aine`) in ML Hackathon and this repo gives details of my approach to problem statement.

[Check results here](https://github.com/shanmukh05/ML-Hackathon-Techkriti-2021/blob/main/Result%20T21.pdf)

[Visit official website](https://techkriti.org/)

# Problem Statement

Classify the images into following 6 categories 
- buildings
- forest
- glacier
- mountain
- sea
- street

[More details here](https://github.com/shanmukh05/ML-Hackathon-Techkriti-2021/blob/main/ML%20problem%20Statement.pdf)

# Dataset

[Dataset is uploaded in Kaggle](https://www.kaggle.com/shanmukh05/ml-hackathon)

## Data Distribution

![image](https://user-images.githubusercontent.com/65073329/123540781-5b160700-d75e-11eb-8e88-c96df7ed94b0.png)
- Total images in training data : 14034
- Total Images in Validation data : 3000
- Total images in test data : 7301

## Preprocessing Data
- Height, Width = 150, 150
- Augmentations are added to images using `tf.keras.preprocessing.image.ImageDataGenerator` 
    - random rotation
    - horizontal flip
    - width shift range
    - shear range
    - zoom range

# Training

- Metric : Accuracy
- GPU used : NVIDIA TESLA P100
- Framework : TensorFlow


