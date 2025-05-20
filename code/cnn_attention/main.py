import argparse
from models.base_cnn import SimpleCNN
from models.cnn_with_attention import CNNWithCBAM
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'cbam'])
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

if args.model == 'cnn':
    model = SimpleCNN()
else:
    model = CNNWithCBAM()

train(model, train_loader, val_loader, criterion, optimizer, epochs=args.epochs)