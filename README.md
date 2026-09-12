# EXP 04: Developing a Neural Network Classification Model using Transfer Learning

## Aim

To develop a neural network classification model using transfer learning with a pre-trained VGG19 model and classify images into defect and notdefect classes.

## Algorithm

1. Import the required PyTorch, torchvision, matplotlib, NumPy and sklearn libraries.
2. Define the image transformation and resize the images to 224 × 224.
3. Extract the chip dataset and load the training and testing images using `ImageFolder`.
4. Create DataLoaders with a batch size of 32.
5. Load the pre-trained VGG19 model.
6. Modify the final fully connected layer to match the two classes in the dataset.
7. Freeze the feature extraction layers of the pre-trained model.
8. Use Cross Entropy Loss and Adam optimizer with a learning rate of 0.001.
9. Train the model for 10 epochs and calculate the training and validation loss.
10. Test the model and calculate the test accuracy.
11. Display the confusion matrix and classification report.
12. Predict the class of a sample image.

## Program

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Extract dataset
import zipfile
import os

zip_path = '/content/chip_data.zip'
extract_dir = 'data'

os.makedirs(extract_dir, exist_ok=True)

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Successfully extracted '{zip_path}' to '{extract_dir}'")
except zipfile.BadZipFile:
    print(f"Error: '{zip_path}' is a bad zip file. Please ensure it's not corrupted.")
except Exception as e:
    print(f"An error occurred during extraction: {e}")

# Load dataset
dataset_path = "./data/dataset/"

try:
    train_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/train",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{dataset_path}/test",
        transform=transform
    )

    print(
        f"Train dataset loaded with {len(train_dataset)} "
        f"images and {len(train_dataset.classes)} classes."
    )

    print(
        f"Test dataset loaded with {len(test_dataset)} "
        f"images and {len(test_dataset.classes)} classes."
    )

except Exception as e:
    print(f"Error loading datasets: {e}")

# Create DataLoader for batch processing
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Load a pre-trained VGG19 model
model = models.vgg19(weights='VGG19_Weights.DEFAULT')

# Move model to GPU if available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(device)

# Modify the final fully connected layer
num_classes = len(train_dataset.classes)

model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    num_classes
)

model = model.to(device)

# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Train the model
def train_model(model, train_loader, test_loader, num_epochs=10):

    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_losses.append(
            running_loss / len(train_loader)
        )

        # Compute validation loss
        model.eval()

        val_running_loss = 0.0

        with torch.no_grad():

            for images, labels in test_loader:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item()

        val_losses.append(
            val_running_loss / len(test_loader)
        )

        model.train()

        print(
            f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {train_losses[-1]:.4f}, '
            f'Validation Loss: {val_losses[-1]:.4f}'
        )

    print("Name: SAJITH AHAMED F")
    print("Register Number: 212223240144")

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, num_epochs + 1),
        train_losses,
        label='Train Loss',
        marker='o'
    )

    plt.plot(
        range(1, num_epochs + 1),
        val_losses,
        label='Validation Loss',
        marker='s'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# Train the model
train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10
)

# Test the model
def test_model(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    accuracy = correct / total

    print(f'Test Accuracy: {accuracy:.4f}')

    # Confusion matrix
    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    print("Classification Report:")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=train_dataset.classes
        )
    )


# Evaluate the model
test_model(model, test_loader)

# Predict on a single image
def predict_image(model, image_index, dataset):

    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():

        image_tensor = image.unsqueeze(0).to(device)

        output = model(image_tensor)

        _, predicted = torch.max(output, 1)

        predicted = predicted.item()

    class_names = dataset.classes

    image_to_display = transforms.ToPILImage()(image)

    plt.figure(figsize=(4, 4))

    plt.imshow(image_to_display)

    plt.title(
        f'Actual: {class_names[label]}\n'
        f'Predicted: {class_names[predicted]}'
    )

    plt.axis("off")
    plt.show()

    print(
        f'Actual: {class_names[label]}, '
        f'Predicted: {class_names[predicted]}'
    )


# Example prediction
predict_image(
    model,
    image_index=55,
    dataset=test_dataset
)
```

## Output

<img width="375" height="140" alt="image" src="https://github.com/user-attachments/assets/9f28aae2-8985-49d5-857e-d96afe7ec659" />

<img width="521" height="190" alt="image" src="https://github.com/user-attachments/assets/f51716e7-bfa5-4bcb-9080-643e4723e4e1" />

<img width="662" height="782" alt="image" src="https://github.com/user-attachments/assets/c9cdc685-d4c1-446b-b808-b15fdef42058" />

<img width="682" height="792" alt="image" src="https://github.com/user-attachments/assets/5c459419-1682-4db8-9700-93d483206a1b" />

<img width="692" height="477" alt="image" src="https://github.com/user-attachments/assets/4db70a91-d7ea-444a-b0a3-4034836c746d" />

<img width="611" height="678" alt="image" src="https://github.com/user-attachments/assets/12f658e7-1d65-4992-a543-e7c4ee6918d2" />

<img width="602" height="743" alt="image" src="https://github.com/user-attachments/assets/fe229094-3903-4ed1-8b8b-d9a6f42fa81a" />




* Dataset loading output
* VGG19 model summary
* Training and validation loss graph
* Training output for 10 epochs
* Confusion matrix
* Classification report
* Sample image prediction

The notebook's training reached a final training loss of `0.0011` and validation loss of `0.2096` in epoch 10.

The final test accuracy obtained was `0.9587` (95.87%).

The classification report showed:

* `defect` — precision 0.94, recall 0.91, F1-score 0.
