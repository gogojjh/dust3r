import matplotlib.pyplot as plt
import json
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Visualize training loss from a log file.")
parser.add_argument("--logdir", type=str, required=True, help="Path to the directory containing the log file.")
args = parser.parse_args()

# Construct the full path to the log file
log_file_path = os.path.join(args.logdir, "log.txt")

# Load data from log.txt
epochs = []
train_losses = []
try:
    with open(log_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if "train_loss" in data:
                epochs.append(data["epoch"])
                train_losses.append(data["train_loss"])

except FileNotFoundError:
    print(f"Error: The log file '{log_file_path}' does not exist.")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: The log file '{log_file_path}' contains invalid JSON.")
    exit(1)

# Plot train_loss as dots
plt.plot(epochs, train_losses, 'o', label="Train Loss")  # 'o' for dots
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()
