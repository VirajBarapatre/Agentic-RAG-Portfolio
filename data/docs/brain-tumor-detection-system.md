# Brain Tumor Detection System

## Project Overview
Brain Tumor Detection System is a deep learning-based project that detects brain tumors from MRI images using Convolutional Neural Networks (CNN). This project uses a custom-trained CNN model to classify MRI images into two categories: Tumor Detected and No Tumor.

## Dataset
The dataset is downloaded from **Kaggle**:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### Dataset Structure
```
Brain_Tumor_Dataset/
│
├─ Training/         # Training Images
│
└─ Testing/          # Testing Images
```

## Tech Stack
- Python
- TensorFlow
- Keras
- Tkinter (GUI)
- Matplotlib
- NumPy
- Pandas
- OpenCV

## Project Folder Structure
```
Brain_Tumor_Detection_System/
│
├─ dataset/                # Brain Tumor Dataset Folder
├─ model/                  # Saved CNN Model
├─ preprocessing.py        # Preprocessing Code
├─ model_building.py       # CNN Model Building and Training
├─ evaluation.py           # Model Evaluation and Visualization
├─ gui.py                  # GUI Code with Tkinter
└─ README.md               # Project Documentation
```

## How to Run the Project
### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Pandas
- Matplotlib

### Install Dependencies
```bash
pip install tensorflow opencv-python pandas matplotlib
```

### Run the GUI Application
```bash
python gui.py

```

## Project Screenshots
| Training Accuracy & Loss | GUI Interface |
|--------------------------|---------------|
|![Figure_1](https://github.com/user-attachments/assets/0d9c12ea-4448-4cee-84ef-5501dc552682) | ![GUI Interface](https://github.com/user-attachments/assets/0598ab75-f10a-4e58-97e8-71399d7a3b4d)




## Results
| Metric    | Score  |
|-----------|-------|
| Training Accuracy | 99.91% |
| Validation Accuracy | 98.43% |

## Author
**Viraj Barapatre**

GitHub Profile: [VirajBarapatre](https://github.com/VirajBarapatre)

## License
This project is licensed under the **MIT License**.

