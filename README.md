# BrainTumor_MRI_ImageClassification
**Problem Statement**
The goal of this project is to develop a deep learning-based system for classifying brain MRI images by tumor type. The solution will include:

Designing and training a custom convolutional neural network (CNN) from scratch.
Enhancing performance through transfer learning with pretrained models.
Deploying a user-friendly Streamlit web application that enables real-time tumor type predictions from uploaded MRI scans.

**Key objectives**

Build a robust, end-to-end pipeline for data preprocessing, model training, evaluation, and selection.
Compare the performance of a bespoke CNN against transfer learning approaches using state-of-the-art pretrained architectures.
Create an intuitive interface for clinicians and researchers to upload MRI images and obtain rapid, reliable predictions.
Ensure model interpretability and provide clear visualization of results (e.g., class probabilities, confusion matrices and evaluation plots).

## Features

* **Multi-Class Classification:** Categorizes brain MRI images into 4 distinct classes: Glioma, Meningioma, No Tumor, and Pituitary.
* **Transfer Learning:** Implements and evaluates several pre-trained CNN architectures (VGG16, ResNet50, InceptionV3, DenseNet121, MobileNet) for enhanced performance.
* **Custom CNN:** Includes a custom-built Convolutional Neural Network for baseline comparison.
* **Performance Metrics:** Detailed evaluation using Accuracy, Precision, Recall, Loss, Classification Reports, and Confusion Matrices.
* **Model Comparison:** Comprehensive comparison of all trained models to identify the best-performing architecture.
* **Interactive Web App:** A user-friendly Streamlit application to upload MRI images and get instant tumor type predictions.

---

## Dataset

The project utilizes a dataset of brain MRI images.
* **Source:** [Brain Tumor MRI Multi-Class Dataset](https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju?usp=sharing)
* **Classes:**
    * `glioma`
    * `meningioma`
    * `no_tumor`
    * `pituitary`
* **Structure:** The dataset is organized into `train`, `validation`, and `test` directories, with subfolders for each tumor type.
* **Image Size:** Images are typically resized to 224x224 (or 299x299 for InceptionV3) pixels and normalized.

---

## Models Explored

The following CNN architectures were trained and evaluated:

1.  **Custom CNN:** A manually designed convolutional neural network.
2.  **VGG16:** A 16-layer deep convolutional network, pre-trained on ImageNet.
3.  **ResNet50:** A 50-layer Residual Network, pre-trained on ImageNet, known for its skip connections.
4.  **InceptionV3:** An Inception architecture with 48 layers, pre-trained on ImageNet, using inception modules for efficient computation.
5.  **DenseNet121:** A 121-layer Densely Connected Convolutional Network, pre-trained on ImageNet, promoting feature reuse.
6.  **MobileNetV2:** A lightweight, efficient model pre-trained on ImageNet, suitable for mobile and embedded vision applications.

---

## Performance Highlights

The **MobileNet** model emerged as the top performer based on test set evaluation.

| Model        | Test Accuracy | Test Precision | Test Recall | Test Loss |
| :----------- | :------------ | :------------- | :---------- | :-------- |
| **MobileNet** | **0.9065** | **0.9177** | **0.9065** | 0.2632     |
| InceptionV3  | 0.8740        | 0.8770         | 0.8699      | 0.4031    |
| VGG16        | 0.8252        | 0.8426         | 0.8049      | 0.4656    |
| DenseNet121  | 0.7724        | 0.8235         | 0.7398      | ~0.55     |
| CustomCNN    | 0.7276        | 0.7789         | 0.6301      | 0.6327    |
| ResNet50     | 0.5325        | 0.7059         | 0.2439      | 1.0787    |


---
---

## Setup and Installation

To set up the project locally or in a Google Colab environment:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Nithya-Satishkumar/BrainTumor_MRI_Imageclassification.git]
    cd BrainTumor_MRI_Imageclassification
    ```

2.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Download Dataset:**
    * Place your brain tumor image dataset into the `data/` directory with `train`, `validation`, and `test` subdirectories.

---

## Usage

### Training and Evaluation

All model training, evaluation, and comparison code is typically found in the Colab notebooks (`notebooks/BrainTumor_ImageClassification.ipynb`).

1.  **Open the Notebook:**
    * **Google Colab:** Upload the `.ipynb` file to Google Colab. Ensure your runtime type is set to GPU (`Runtime > Change runtime type`).
2.  **Mount Google Drive (Colab Only):** If your dataset or saved models are in Google Drive, execute the following in a Colab cell:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3.  **Run Cells:** Execute the notebook cells sequentially to:
    * Load and preprocess data using `ImageDataGenerator`.
    * Define and compile each model (Custom CNN, VGG16, ResNet50, InceptionV3, DenseNet121, MobileNet).
    * Train models using `model.fit()` with callbacks (Early Stopping, Model Checkpoint).
    * Evaluate models using `model.evaluate()`.
    * Generate classification reports and confusion matrices.
    * Populate and visualize the `model_comparison` DataFrame.

### Running the Streamlit App

The `TumorClassification.py` file contains the Streamlit application for inference.

1.  **Ensure Model is Saved:** Make sure your best-performing model (e.g., `MobileNetModel.h5`) is saved in the `models/` directory or at the path specified in `TumorClassification.py`.

2.  **Run the App (Local):**
    ```bash
    streamlit run TumorClassification.py
    ```
    This will open the app in your default web browser.

3.  **Run the App (Google Colab with localtunnel):**
    * First, save the `TumorClassification.py` code to a file in Colab (as shown in previous conversations using `%%writefile TumorClassification.py`).
    * Install `localtunnel` if you haven't already: `!npm install -g localtunnel`
    * Run the app and tunnel in a Colab cell:
        ```bash
        !streamlit run TumorClassification.py & npx localtunnel --port 8501
        ```
    * A public URL will be provided (e.g., `https://random-word-number.loca.lt`). Click this URL to access your app. You may need to enter your Colab instance's public IP as a "tunnel password" the first time. Get it by running `!wget -q -O - ipv4.icanhazip.com` in a separate Colab cell.

---

## Results and Model Comparison

The detailed `model_comparison` DataFrame (as presented in the "Performance Highlights" section) provides a quantitative overview. Visualizations from the notebooks further illustrate these comparisons, highlighting the strengths of MobileNet and InceptionV3 for this task. The confusion matrices offer insights into specific misclassification patterns for each model.

---

## Future Work

* **Data Augmentation:** Explore more advanced data augmentation techniques (e.g., CutMix, Mixup) to further enhance model robustness.
* **Hyperparameter Tuning:** Systematically tune hyperparameters (learning rate, batch size, optimizer) for each model.
* **Ensemble Methods:** Combine predictions from multiple models to potentially achieve higher accuracy.
* **Grad-CAM/LIME:** Implement interpretability techniques to visualize which parts of the image the model focuses on for predictions.
* **Deployment:** Explore deploying the model using Flask/Django or cloud platforms (e.g., TensorFlow Serving, Google Cloud AI Platform) for production-ready applications.
* **Larger Dataset:** Test models on a larger and more diverse dataset for better generalization.

---
