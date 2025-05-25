import argparse
from cnn.model import SimpleCNN, ResidualCNN
from cnn_attention.model import AttentionCNN
from vit.model import VisionTransformer
from train import train_model

def main():
    print("Training model...")

    # Create an argument parser to handle command line options
    parser = argparse.ArgumentParser()

    # Add an argument to select the model type; default is 'cnn'
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vit', 'cnn_attention', ])

    # Add an argument to set the number of training epochs; default is 20
    parser.add_argument('--epochs', type=int, default=20)

    # Parse the command line arguments
    args = parser.parse_args()

    # Based on the selected model, set up the training configuration and call the training function
    if args.model == 'cnn_attention':
        print("Training model CNN with Attention ...")

        # Configuration dictionary specific for AttentionCNN
        config = {
            "epochs": args.epochs,          # Number of training epochs
            "batch_size": 32,               # Batch size for data loading
            "lr": 0.001,                   # Learning rate for optimizer
            "model_kwargs": {"num_classes": 4},  # Model-specific arguments (output classes)
            "log_dir": "runs/attention"    # Directory to save training logs for TensorBoard
        }

        # Train AttentionCNN model with the given config
        train_model(config, AttentionCNN)

    elif args.model == 'vit':
        print("Training model VisionTransformer ...")

        # Configuration dictionary specific for VisionTransformer model
        config = {
            "epochs": args.epochs,
            "batch_size": 32,
            "lr": 0.001,
            "model_kwargs": {
                "image_size": 128,    # Input image size expected by the ViT model
                "patch_size": 16,     # Size of image patches for embedding
                "in_channels": 3,     # Number of input image channels (e.g. RGB)
                "embed_dim": 60,      # Dimension of the patch embeddings
                "num_classes": 4,     # Number of output classes
                "num_heads": 4,       # Number of attention heads in the transformer
            },
            "log_dir": "runs/vit_patch16_heads4"
        }

        # Train VisionTransformer model with the given config
        train_model(config, VisionTransformer)

    else:
        print("Training model SimpleCNN ...")

        # Configuration dictionary for simple CNN training
        config = {
            "epochs": args.epochs,
            "batch_size": 32,
            "lr": 0.001,
            "model_kwargs": {"num_classes": 4},
            "log_dir": "runs/simplecnn"
        }

        # Train SimpleCNN model
        train_model(config, SimpleCNN)

        print("Training model ResidualCNN ...")

        # Adjust config for ResidualCNN (different output classes and log directory)
        config["log_dir"] = "runs/residual"
        config["model_kwargs"] = {"num_classes": 6}

        # Train ResidualCNN model
        train_model(config, ResidualCNN)

    print("Training model DONE")

# Standard Python idiom to ensure main() runs when script is executed directly
if __name__ == "__main__":
    main()
