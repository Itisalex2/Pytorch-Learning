import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim


class ImageClassifier(nn.Module):
    """
    ImageClassifier: A simple convolutional neural network for image classification.

    This model consists of:
    - Two convolutional layers with ReLU activations
    - Max pooling and dropout for regularization
    - A fully connected layer followed by another dropout
    - LogSoftmax for the output layer to get log-probabilities
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.max_pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """ Input: (1, 28, 28) """
        x = self.conv1(x)  # (32, 26, 26)
        x = self.relu(x)  # (32, 26, 26)
        x = self.conv2(x)  # (64, 24, 24)
        x = self.relu(x)  # (64, 24, 24)
        x = self.max_pool(x)  # (64, 12, 12)
        x = self.dropout1(x)  # (64, 12, 12)
        x = self.flatten(x)  # (9216) -> 64 * 12 * 12
        x = self.fc1(x)  # (128)
        x = self.relu(x)  # (128)
        x = self.dropout2(x)  # (128)
        x = self.fc2(x)  # (10)
        output = self.log_softmax(x)  # (10)
        return output


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Compute the prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    # Evaluate
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # Get the correct device

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Get the data

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # Constants for MNIST
    TRAINING_DATA_SIZE = 60000
    TEST_DATA_SIZE = 10000
    IMAGE_DIMENSIONS = (1, 28, 28)

    # Make sure the dimensions are correct
    assert (len(training_data) == TRAINING_DATA_SIZE)
    assert (len(test_data) == TEST_DATA_SIZE)
    assert (training_data[0][0].shape == IMAGE_DIMENSIONS)
    assert (test_data[0][0].shape == IMAGE_DIMENSIONS)

    # Hyperperameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Set up model
    classifier = ImageClassifier().to(device)
    print(classifier)

    # Optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    for t in range(NUM_EPOCHS):
        print(f"Epoch {t+1}\n------------------------------")
        train_loop(train_dataloader, classifier, loss_fn, optimizer)
        test_loop(test_dataloader, classifier, loss_fn)

    # Save the model
    torch.save(classifier.state_dict(), "classifier-parameters.pth")
    print("Saved PyTorch Model State to classifier-parameters.pth")


if __name__ == "__main__":
    main()
