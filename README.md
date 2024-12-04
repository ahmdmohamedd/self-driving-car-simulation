# Self-Driving Car Simulation with CNNs

## Overview

This project demonstrates the development and training of a self-driving car simulation system using Convolutional Neural Networks (CNNs). The system is designed to predict steering angles based on images from a car’s camera. The model was trained using a dataset with camera images and corresponding steering angles, achieving exceptional performance with low Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## Project Highlights

- **Model Type**: Convolutional Neural Networks (CNNs)
- **Training Data**: Driving images and steering angles from a real-world self-driving dataset.
- **Evaluation Metrics**: 
  - **MSE**: 0.0087
  - **MAE**: 0.0566
- **Output**: A trained model capable of predicting steering angles based on input images.

## Dataset

The dataset used for training the model consists of camera images and the corresponding steering angles. The dataset is publicly available and can be found [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip).

### Key Dataset Components:
- **driving_log.csv**: A CSV file containing paths to the images and their corresponding steering angles, throttle, brake, and speed data.
- **IMG Folder**: Contains the images captured from three cameras: center, left, and right.

## Features

- **CNN Architecture**: The model uses a standard CNN architecture with multiple convolutional and pooling layers for feature extraction followed by fully connected layers for regression.
- **Model Performance**: The model achieves excellent performance, making it suitable for real-time applications in self-driving systems.
- **Visualization**: The project includes a visualization of true vs. predicted steering angles to evaluate the model’s accuracy.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/self-driving-car-simulation.git
   ```

2. Download the dataset from [this link](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip), and extract it into the project folder.

## Training the Model

1. Start by loading the dataset (driving_log.csv) and images.
2. Preprocess the data by normalizing and augmenting the images.
3. Train the CNN model on the dataset using the provided Jupyter notebook `self_driving_car_simulation.ipynb`.
4. The training process involves optimizing the model to minimize the loss function and improve the model's prediction of steering angles.

## Evaluation

After training, the model is evaluated using the validation data. The following metrics are computed to assess the performance of the model:

- **Mean Squared Error (MSE)**: 0.0087
- **Mean Absolute Error (MAE)**: 0.0566

The performance is visualized by plotting the true vs. predicted steering angles.

## Usage

Once the model is trained, it can be saved and used to simulate the car's steering behavior on new images. The saved model allows anyone to perform actual simulations by loading the model and passing new input images for steering angle prediction.

To load the saved model and run simulations:

1. Download the saved model from the repository.
2. Use the provided script to run the model on new input images.

```python
from keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Use the model to predict steering angles for new images
steering_angle = model.predict(image)
```

## Contributing

Contributions to this project are welcome! If you have improvements, bug fixes, or new features to add, please feel free to fork the repository and submit a pull request.
