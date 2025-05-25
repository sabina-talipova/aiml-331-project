import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

import dataset_wrapper

def train_model(config, model_class):
    # Select the device: use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s", device)

    # Initialize the model with parameters from config and move it to the selected device
    model = model_class(**config.get("model_kwargs", {})).to(device)

    # Load training, validation, and test datasets using the custom dataset wrapper
    print("Load dataset")
    train_dataset, val_dataset, test_dataset = dataset_wrapper.get_pet_datasets(
        img_width=config.get("img_width", 128),
        img_height=config.get("img_height", 128),
        root_path=config.get("root_path", "data")
    )

    # Create data loaders for efficient batch loading
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 32))
    test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size", 32))

    # Define loss function (cross-entropy) and optimizer (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))

    # Initialize TensorBoard summary writer for logging training metrics
    log_dir = config.get("log_dir", "runs/exp1")
    writer = SummaryWriter(log_dir)

    # Training loop
    for epoch in range(config.get("epochs", 20)):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()       # Reset gradients from previous step
            outputs = model(inputs)     # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()             # Backpropagation
            optimizer.step()            # Update model weights

            running_loss += loss.item() # Accumulate training loss

        # Calculate average training loss and log to TensorBoard
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)  # Get predicted classes
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total  # Compute validation accuracy
        writer.add_scalar("Accuracy/val", val_acc, epoch)  # Log to TensorBoard

        print(f"Epoch {epoch+1}: Train loss={avg_loss:.4f}, Val acc={val_acc:.4f}")

    # Final evaluation on the test dataset
    test_correct = 0
    test_total = 0
    model.eval()
    start = time.time()  # Start timing inference
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)
    end = time.time()

    # Compute final test accuracy and inference time per image
    test_acc = test_correct / test_total
    print(f'Final Test Accuracy: {test_acc:.4f}, Inference time: {(end-start)/test_total:.6f}s per image')

    # Close the TensorBoard writer
    writer.close()
