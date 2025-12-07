# Digit Classifier with TensorFlow

A simple, clean machine learning project that trains a neural network to classify handwritten digits (0–9) using the MNIST dataset. Built with TensorFlow and designed to be lightweight to run.

---

## Project Overview

This project demonstrates a basic deep learning workflow using TensorFlow and Keras. It includes:

- Loading and preprocessing the MNIST dataset
- Building a fully connected neural network
- Training, evaluating, and saving the model
- Visualizing data with Jupyter Notebook
- Clean, modular code with best practices

---

## Tech Stack

- Python 3.9+
- [TensorFlow](https://www.tensorflow.org/) (Keras API)
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Sample Output

Here’s a sample MNIST digit used during training:

![Sample MNIST Digit](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/digit-classifier.git
cd digit-classifier
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3. Run the Model Training
```bash
python main.py
```

### 4. (Optional) Launch Jupyter Notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

### Results

- Accuracy: ~97% on MNIST test set after 5 epochs
- Model saved to model/model.h5

### Best Practices Followed

- Clean project structure
- Reusable data loading module
- Virtual environment isolation
- Lightweight & laptop-friendly
- Documented and modular code

### License
This project is licensed under the MIT License. See LICENSE for details.
