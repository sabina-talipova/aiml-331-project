import argparse
import vit.train as vit_train
import cnn.train as cnn_train
import cnn_attention.train as cnn_attention_train

import vit.train_model

def main():
    print("Training model...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vit', 'cnn_attention'])
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    if args.model == 'cnn_attention':
        cnn_attention.train_cnn_with_attention_model()
    elif args.model == 'vit':
        cnn_train.train_model({"epochs": args.model})
    else:
        cnn_train.train_model({"epochs": args.model})

    print("Training model DONE")

# Entry point to start the main function if this script is run directly
if __name__ == "__main__":
    main()