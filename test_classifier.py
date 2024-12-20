import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from image_classification import ImageClassifier
import matplotlib.pyplot as plt


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

    # Load the model
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load("classifier-parameters.pth"))
    model.eval()
    print("Loaded PyTorch Model State from classifier-parameters.pth")

    # Get the test data
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    test_dataloader = DataLoader(test_data, batch_size=64)

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Visualize some predictions
    X, y = next(iter(test_dataloader))
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
        pred = model(X).argmax(1)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f"True: {y[i].item()}, Pred: {pred[i].item()}")
        ax.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
