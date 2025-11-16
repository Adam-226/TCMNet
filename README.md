## TCM Herb Image Classification using CNN
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify traditional Chinese medicine herb images.
The model is trained using a dataset organized into labeled folders and can run on both CPU and Apple MPS (Metal Performance Shaders) for accelerated training on macOS devices.

## Model Architecture
| Layer     | Details                 |                        |
| --------- | ----------------------- | ---------------------- |
| Conv2D    | 3 → 32                  | kernel: 3×3, padding:1 |
| Conv2D    | 32 → 64                 | kernel: 3×3            |
| Conv2D    | 64 → 128                | kernel: 3×3            |
| MaxPool2D | Applied after each conv |                        |
| FC Layer  | 128×16×16 → 512         |                        |
| Dropout   | 0.5                     |                        |
| FC Layer  | 512 → 5 categories      |                        |
