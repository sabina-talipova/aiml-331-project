import argparse
from cnn.model import SimpleCNN
from cnn_attention.model import AttentionCNN
from vit.model import VisionTransformer
from train import train_model

def main():
    print("Training model...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vit', 'cnn_attention'])
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    if args.model == 'cnn_attention':
        print("Training model CNN with Attention ...")
        config = {
            "epochs": args.epochs,
            "batch_size": 32,
            "lr": 0.001,
            "model_kwargs": {"num_classes": 4},
            "log_dir": "runs/attention"
        }
        train_model(config, AttentionCNN)

    elif args.model == 'vit':
        print("Training model VisionTransformer ...")
        config = {
            "epochs": args.epochs,
            "batch_size": 32,
            "lr": 0.001,
            "model_kwargs": {
                "img_size": 128,
                "patch_size": 16,
                "in_channels": 3,
                "embed_dim": 60,
                "num_classes": 4,
                "depth": 6,
                "num_heads": 4,
                "use_positional_embedding": True
            },
            "log_dir": "runs/vit_patch16_heads4"
        }
        train_model(config, VisionTransformer)
    else:
        print("Training model SimpleCNN ...")
        config = {
            "epochs": args.epochs,
            "batch_size": 32,
            "lr": 0.001,
            "model_kwargs": {"num_classes": 4},
            "log_dir": "runs/simplecnn"
        }
        train_model(config, SimpleCNN)

        print("Training model ResidualCNN ...")

        config["log_dir"] = "runs/residual"
        config["model_kwargs"] = {"in_channels": 3, "out_channels": 64}
        train_model(config, ResidualCNN)

    print("Training model DONE")

# Entry point to start the main function if this script is run directly
if __name__ == "__main__":
    main()