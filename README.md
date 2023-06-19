# Face Recognition Classifier

This project is a face recognition classifier that aims to identify and classify faces as belonging to either a registered user or a non-user. It utilizes a simple perceptron implemented in the Keras library to accomplish this task.

## Introduction

Face Recognition Classifier is a machine learning project that utilizes computer vision techniques to recognize and classify faces. It is designed to identify whether a face belongs to a registered user or a non-user based on a trained model.

## Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Image preprocessing](#image-preprocessing)
- [Model](#model)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)


### Clone

If you want to quickly clone the repository to your local machine, you can use the following command:

```bash
git clone --depth 1 https://github.com/cesarvillegase/Face_recognition.git
```

## Image Preprocessing



In the image preprocessing step, we apply several techniques to enhance the dataset and improve the robustness of our face recognition system. These techniques include resizing (1), grayscale conversion (2), and data augmentation (3).

### Resizing of the Images (1)

To ensure consistency in the input data, we resize the images to a fixed dimension of 500x500 pixels. This resizing step helps in normalizing the image sizes and prepares them for further processing.

```python
def resize_imgs(input, output):

    # Create a new directory to save the resized images
    os.makedirs(output, exist_ok=True)

    # Define the desired dimension
    new_width = 500
    new_height = 500

    # Iterate through the images and resize them
    for filename in os.listdir(input):
        image_path = os.path.join(input, filename)
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize((new_width, new_height))

        # Save the resized image in the new directory
        resized_image_path = os.path.join(output, filename)
        resized_image.save(resized_image_path)
```

### Grayscale Conversion (2)

Converting the images to grayscale simplifies the data representation while preserving the essential features of the faces. By converting the images from RGB (Red, Green, Blue) to grayscale, we reduce the computational complexity and focus on the intensity variations that convey important facial information.
```python
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image_array = np.array(image)  # Convert the image to a numpy array
    image_array = image_array.flatten().reshape(1, -1) # Flatten the 2D array into a 1D array
    return image_array
```
### Data Augmentation (3)

Data augmentation is a technique used to artificially increase the size of the training dataset by applying various transformations to the existing images. This helps to introduce diversity in the training data and improves the model's ability to generalize to unseen faces. Some common data augmentation techniques we employ include random rotations, flips, and shifts.


```python
#Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate images randomly by 20 degrees
    width_shift_range=0.1,  # Shift images horizontally by 10% of the width
    height_shift_range=0.1,  # Shift images vertically by 10% of the height
    shear_range=0.2,  # Apply shearing transformations
    zoom_range=0.2,  # Zoom images by up to 20%
    horizontal_flip=True  # Flip images horizontally
)

# Fit the data generator on the data
datagen.fit(X)

# Generate augmented images and labels
augmented_data = datagen.flow(X, y, batch_size=len(X), shuffle=True)

# Retrieve the augmented data and labels
X_augmented, y_augmented = augmented_data.next()

# Concatenate the augmented data with the original data
X = np.concatenate((X, X_augmented))
y = np.concatenate((y, y_augmented))
```

These preprocessing steps, including resizing (1), grayscale conversion (2), and data augmentation (3), play a crucial role in enhancing the performance and robustness of our face recognition system. By incorporating these techniques, we ensure that our model can effectively handle variations in image sizes, lighting conditions, and facial expressions.

## Model

In this section, we describe the process of training our face recognition model using the preprocessed images. We utilize a multi-layer perceptron architecture implemented in Keras for this task.

### Data Splitting

Before training the model, we split our dataset into training and validation sets. The training set is used to train the model, while the validation set helps us monitor the model's performance and prevent overfitting.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Convert string labels to numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert numerical labels to one-hot encoded vectors
num_classes = len(label_encoder.classes_)
onehot_encoder = OneHotEncoder(categories='auto')
y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1)).toarray()
y_test_onehot = onehot_encoder.transform(y_test_encoded.reshape(-1, 1)).toarray()
```

### Model Architecture

Our face recognition model employs a multi-layer perceptron (MLP) implemented using the Keras library. The MLP consists of multiple layers, including input, hidden, and output layers. Each layer contains multiple neurons, and the connections between the neurons are weighted to learn the appropriate representations for face recognition.

```python
model = Sequential()

# Capa oculta 1 con ReLU
model.add(Dense(units=64, activation='relu', input_shape=(250000,)))

# Capa oculta 2 con ReLU
model.add(Dense(units=128, activation='relu'))

# Capa oculta 3 con ReLU
model.add(Dense(units=256, activation='relu'))

model.add(Dense(units=128, activation='relu'))

# Capa de salida con sigmoide
model.add(Dense(units=num_classes, activation='sigmoid'))

# Compilación del modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# We train the model
model.fit(X_train, y_train_onehot, epochs=20, batch_si
```

### Training Procedure

During training, we use various techniques to optimize the model's performance. These include setting appropriate learning rates and selecting an appropriate loss function and evaluation metrics.

We train the model using the training dataset and monitor its performance on the validation dataset. We iterate over multiple epochs, adjusting the model's parameters to minimize the loss and improve accuracy.

### Model Evaluation

Once the training is complete, we evaluate the trained model on a separate test dataset that the model has not seen during training. This allows us to assess the model's performance and its ability to generalize to unseen faces.

We measure the model's performance using various evaluation metrics such as accuracy and loss model. These metrics provide insights into the model's effectiveness in recognizing faces accurately and efficiently.

```python
# Reshape the image data
X_test_new_reshaped = X_test_new.reshape((-1, 500, 500))

num_rows = 7
num_cols = 10

# Create a figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Iterate through the images and their predicted labels
for i, ax in enumerate(axes.flat):
    # Display the image
    ax.imshow(X_test_new_reshaped[i], cmap='gray')
    ax.axis('off')

    # Set the color of the title based on prediction correctness
    if predicted_labels[i] == y_test[i]:
        ax.set_title(str(predicted_labels[i]), color='green')
    else:
        ax.set_title(str(predicted_labels[i]), color='red')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
```

## Examples

![Recognition](https://github.com/cesarvillegase/Face_recognition/assets/101744671/9ccf6ae4-65f0-4e5e-8630-6f0d7f9e0169)

## Contributing


Thank you for your interest in contributing to this project! We welcome contributions from the community to make this project even better. To contribute, please follow these guidelines:

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your contribution:
3. Make your changes, following the project's coding conventions and best practices.
4. Commit your changes with descriptive commit messages:
5. Push your branch to your forked repository:
6. Open a pull request from your branch to the main repository's `main` branch.
7. Provide a clear title and description for your pull request, explaining the changes you made.
8. Be responsive to any feedback or comments during the review process and make the necessary updates to your pull request.
9. Once your pull request is approved and merged, your contribution will become a part of the project.

Please note that we have a code of conduct in place. Be respectful and considerate of others when contributing to this project.

If you have any questions or need further assistance, feel free to reach out to us. We appreciate your contributions!

Happy coding!

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this project as per the terms of the license.

Please note that this project may utilize third-party libraries or tools, which may have their own licenses. Make sure to comply with the respective licenses when using those components.
