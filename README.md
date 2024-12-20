# Pytorch-Learning: MNIST Image Classification

This project demonstrates how to build and train a simple convolutional neural network (CNN) for image classification using the MNIST dataset. The model is implemented using PyTorch.

## Project Structure

- `image_classification.py`: Contains the model definition, training loop, and testing loop.
- `test_classifier.py`: Loads the trained model and evaluates it on the test dataset, displaying some predictions.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This file.

## Training the Model

To train the model, run the `image_classification.py` script:
```sh
python image_classification.py
```
This will train the model on the MNIST training dataset and save the model parameters to `classifier-parameters.pth`.

## Testing the Model

To test the model and visualize some predictions, run the `test_classifier.py` script:
```sh
python test_classifier.py
```
This will load the saved model parameters, evaluate the model on the MNIST test dataset, and display some test images along with their predicted labels.

## Results

The model achieves an accuracy of approximately 99% on the MNIST test dataset.

## License

This project is licensed under the MIT License.
