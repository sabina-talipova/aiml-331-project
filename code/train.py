class ModelTrainer:
    def __init__(self, dir, model):
        """
        Initializes the ModelTrainer instance.
        Args:
            dir (str): Directory containing image data.
            model (torch.nn.Module): PyTorch model to be trained.
        """
        self.dir = dir
        self.__model = model
        self.__train_loader = None  # DataLoader for training data
        self.__criterion = None  # Loss function
        self.__optimizer = None  # Optimizer for model training
        self.__device = None  # Device on which to train the model (CPU or GPU)

def main():
    """
    Main function to initiate model training.
    Loads environment variables, initializes the model trainer with the data folder path and
    a custom CNN model, and performs cross-validation training.
    """
    # Load environment variables (e.g., for data paths or other configurations)
    load_dotenv()
    print("Training model...")

    # Initialize ModelTrainer with the specified data folder and custom CNN model
    my_object = ModelTrainer(DATA_FOLDER_PATH, CustomCNN())

    # Train the model using cross-validation on the dataset
    my_object.train_model_CV()

    # Optional: Uncomment to train and create the model without cross-validation
    # my_object.create_model()

    print("Training model DONE")

# Entry point to start the main function if this script is run directly
if __name__ == "__main__":
    main()