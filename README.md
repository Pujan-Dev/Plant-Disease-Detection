# Plant Disease Detection

A PyTorch image-classification project for identifying plant diseases from leaf
images. The project uses the directory structure of the dataset to derive class
labels, loads images with a custom `Dataset`, and trains a fully connected
neural network on resized RGB images.

The main workflow is implemented in [main.ipynb](./main.ipynb).

## What this project does

The notebook:

1. Finds the disease classes from the folders inside `Dataset/`.
2. Builds a table containing image file names and their class labels.
3. Undersamples classes with more than 1,000 images.
4. Converts class names into integer IDs.
5. Loads images as RGB PIL images and resizes them to `224 x 224`.
6. Creates stratified training, validation, and test splits:
   - 80% training
   - 10% validation
   - 10% testing
7. Trains a PyTorch model using cross-entropy loss and SGD.
8. Reports training and validation accuracy during training.
9. Evaluates the final model on the test set.
10. Displays a confusion matrix and calculates accuracy, macro precision,
    macro recall, and macro F1-score.

## Dataset structure

The dataset must be available at `./Dataset` relative to the notebook:

```text
Plant-Disease-Detection/
├── Dataset/
│   ├── Apple___Apple_scab/
│   │   ├── image-1.JPG
│   │   └── image-2.JPG
│   ├── Apple___healthy/
│   ├── Potato___Early_blight/
│   └── ...
├── main.ipynb
├── plant_disease_model.pth
└── README.md
```

Each immediate subdirectory of `Dataset/` is treated as one class. The folder
name is used as the label, so do not rename class folders after creating a
training split unless you rerun the notebook from the beginning.

The current dataset contains plant classes including Apple, Corn, Pepper,
Potato, and Tomato diseases and healthy categories. The notebook discovers the
available classes dynamically instead of hard-coding their names.

### Dataset requirements

- Images may be `.jpg`, `.JPG`, `.jpeg`, or `.png`.
- Every class should have at least one valid image.
- Images should be readable by Pillow.
- Folder names should be unique and should not contain accidental nested class
  directories.

The notebook converts every image to RGB, so grayscale and images with alpha
channels are handled consistently.

## Requirements

Use Python 3.9 or newer. The notebook requires:

- Python
- PyTorch
- torchvision
- pandas
- NumPy
- Matplotlib
- Pillow
- scikit-learn
- seaborn
- Jupyter Notebook or JupyterLab

Install the packages with:

```bash
python -m pip install torch torchvision pandas numpy matplotlib pillow scikit-learn seaborn jupyter
```

For a CUDA-enabled installation of PyTorch, use the installation command
recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/)
instead of blindly installing the CPU or CUDA packages shown above.

## Running the notebook

From the project directory:

```bash
cd Plant-Disease-Detection
jupyter notebook main.ipynb
```

In VS Code, open `main.ipynb`, select the Python environment that contains the
required packages, and run the cells in order.

For a clean run:

1. Restart the notebook kernel.
2. Run all cells from the top.
3. Wait for the dataset split and data loaders to be created.
4. Run the model and optimizer cells.
5. Run the training loop.
6. Run the test evaluation, confusion matrix, and score cells.

The notebook uses:

```python
path = "./Dataset"
```

Therefore, the notebook should be run with the project directory as the
working directory. If it is opened from another location, either change the
working directory or update `path` to the absolute or correct relative path.

## Preprocessing and labels

Class IDs are generated in sorted order:

```python
label_to_id = {
    label: index for index, label in enumerate(sorted(unique_labels))
}
```

Images are transformed with:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
```

`ToTensor()` converts pixel values to floating-point tensors in the range
`[0, 1]` and changes the layout from height-width-channel to
channel-height-width.

The model receives flattened image tensors with:

```text
3 channels x 224 height x 224 width = 150,528 input values
```

Labels are returned as `torch.long` integer tensors, which is the format
required by `torch.nn.CrossEntropyLoss`.

## Dataset balancing and splitting

To reduce the effect of very large classes, the notebook keeps at most 1,000
randomly selected images per class:

```python
if len(image_paths) > 1000:
    image_paths = np.random.choice(image_paths, 1000, replace=False)
```

The resulting table is split with stratification so that each split preserves
approximately the same class distribution:

```text
Training:   80%
Validation: 10%
Testing:    10%
```

The random split uses `random_state=42` for repeatable partitioning. The
notebook also checks that no image file name appears in more than one split.

## Model

The current model is a fully connected neural network, not a convolutional
neural network:

```text
Input:  150,528 values
fc1:    150,528 -> 500
ReLU
fc2:    500 -> 500
ReLU
fc3:    500 -> number of classes
```

The model is initialized with:

```python
PlantDiseaseNeuralNet(
    input_size=3 * 224 * 224,
    hidden_size=500,
    num_classes=len(unique_labels)
)
```

The optimizer is stochastic gradient descent:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)
```

The training loop currently runs for 8 epochs. Change the value in
`range(8)` if you want to train for more or fewer epochs.

## Metrics

During each epoch, the notebook prints:

- Training loss
- Training accuracy
- Validation accuracy

After training, the test set is evaluated once and the notebook calculates:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score

It also displays a confusion matrix to show which classes are being confused.
For a fair final estimate, do not use the test set to tune the model during
training.

## Existing model checkpoint

The repository includes [plant_disease_model.pth](./plant_disease_model.pth).
A `.pth` file may contain either a complete serialized model or a model state
dictionary, depending on how it was saved. Before loading it, make sure its
architecture matches the current `PlantDiseaseNeuralNet` definition and that
the class ordering used by `label_to_id` is the same.

For a state dictionary, loading typically looks like:

```python
model.load_state_dict(torch.load(
    "plant_disease_model.pth",
    map_location=device
))
model.to(device)
model.eval()
```

Only load model files from trusted sources. PyTorch checkpoints can contain
serialized Python objects, so avoid loading untrusted files.

## Troubleshooting

### `FileNotFoundError: ./Dataset`

The notebook is running with the wrong working directory. Confirm that
`Dataset/` is next to `main.ipynb`, then run the notebook from the project
directory or update the `path` variable.

### `AttributeError: 'tuple' object has no attribute 'to'`

The data loader is returning file names or tuples rather than image tensors.
Use `PlantDiseaseDataset` with the data loaders and run the dataset definition
cell before creating `trainloader`, `valloader`, and `testloader`.

### `mat1 and mat2 shapes cannot be multiplied`

The flattened input size and model definition do not match. Images resized to
`224 x 224` RGB must be flattened to 150,528 values, and the current model must
keep the hidden dimensions consistent:

```python
self.fc2 = nn.Linear(hidden_size, hidden_size)
self.fc3 = nn.Linear(hidden_size, num_classes)
```

After changing the architecture, recreate the model and optimizer by rerunning
those cells. A kernel restart avoids accidentally using an old model instance.

### CUDA out-of-memory

Try one or more of the following:

- Reduce the `batch_size` from 32 to 16 or 8.
- Reduce the image size, then update the model input size accordingly.
- Run on CPU by selecting a CPU-only environment.
- Restart the notebook kernel and call `torch.cuda.empty_cache()`.

### `ModuleNotFoundError`

Install the missing package in the same Python environment selected by the
notebook. For example:

```bash
python -m pip install pandas torch torchvision scikit-learn pillow seaborn
```

Then restart the kernel.

### Low validation accuracy

This model flattens all pixels and does not use convolutional layers, so it may
perform worse than a CNN or transfer-learning model. Possible improvements
include:

- Replacing the fully connected network with a CNN.
- Using a pretrained model such as ResNet or EfficientNet.
- Adding data augmentation.
- Increasing the number of training epochs.
- Tuning the learning rate and batch size.
- Reporting per-class metrics to identify underperforming classes.

## Reproducibility notes

The split uses `random_state=42`, but the class undersampling step and model
training also use random operations. For more reproducible experiments, set
seeds before building the data frame and model:

```python
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
```

Exact results can still differ between hardware and CUDA configurations.

## Limitations

- The project is notebook-based and does not currently provide a command-line
  training or inference script.
- The model is a fully connected network, so it does not explicitly capture
  local visual patterns as a CNN does.
- The dataset is undersampled rather than augmented or class-weighted.
- The notebook loads images during data-loader iteration, so training time
  depends on storage speed.
- The included checkpoint may not correspond to the latest notebook
  architecture unless it was saved after the architecture changes.

## Project status

This project is intended for experimentation and learning. Predictions should
not be treated as a definitive agricultural diagnosis without validation by a
qualified plant pathologist and testing on representative real-world images.