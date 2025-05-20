import vit.train_model

def main():
    print("Training model...")

    config = {
        "patch": 8,
        "heads": 4,
        "layers": 4,
        "pos": True,
        "epochs": 20
    }

    train_model(config)

    print("Training model DONE")

# Entry point to start the main function if this script is run directly
if __name__ == "__main__":
    main()