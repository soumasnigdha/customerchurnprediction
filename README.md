# Customer Churn Prediction 

This project showcases the use of an Artificial Neural Network (ANN) for classifying and predicting customer churn. The model is trained on the "Churn Dataset" from Kaggle. A Streamlit web application is also provided for interactive predictions.

**Dataset:** [Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/mervetorkan/churndataset)

---

## 🚀 Live Demo
Access the live Streamlit application here:
[https://customerchurnprediction-jt3eapp7slhjeq76nkygph.streamlit.app/](https://customerchurnprediction-jt3eapp7slhjeq76nkygph.streamlit.app/)

---

## 📋 Table of Contents
* [Project Overview](#project-overview)
* [Dataset Details](#dataset-details)
* [Model Architecture](#model-architecture)
* [Streamlit Application](#streamlit-application-️)
* [File Structure](#file-structure)
* [Setup and Installation](#setup-and-installation-⚙️)
* [Usage](#usage)
* [Technologies Used](#technologies-used-🛠️)

---

##  Project Overview
The project follows these main steps:
1.  **Data Loading & Exploration**: The `Churn_Modelling.csv` dataset is loaded.
2.  **Data Preprocessing**:
    * Unnecessary columns (`RowNumber`, `CustomerId`, `Surname`) are dropped.
    * Categorical features are encoded:
        * `Gender`: Transformed using `LabelEncoder`.
        * `Geography`: Transformed using `OneHotEncoder`.
    * The dataset is split into training (80%) and testing (20%) sets.
    * Numerical features are scaled using `StandardScaler`.
3.  **Model Development (ANN)**:
    * A sequential ANN model is built using TensorFlow/Keras.
    * The model is compiled with the Adam optimizer and binary cross-entropy loss.
    * Training includes `EarlyStopping` to prevent overfitting and `TensorBoard` for logging.
4.  **Saving Artifacts**: The trained model (`model.keras`), scaler (`scaler.pkl`), and encoders (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`) are saved for later use.
5.  **Interactive Prediction**: A Streamlit application (`app.py`) loads the saved artifacts to predict churn for new customer data entered by the user.

---

## Dataset Details
The dataset contains information about bank customers, with features including:
* `CreditScore`
* `Geography` (France, Spain, Germany)
* `Gender` (Male, Female)
* `Age`
* `Tenure` (Number of years the customer has been with the bank)
* `Balance`
* `NumOfProducts` (Number of products the customer uses)
* `HasCrCard` (1 if the customer has a credit card, 0 otherwise)
* `IsActiveMember` (1 if the customer is an active member, 0 otherwise)
* `EstimatedSalary`
* `Exited` (Target variable: 1 if the customer churned, 0 otherwise)

---

## Model Architecture
The Artificial Neural Network (ANN) is a sequential model with the following layers:
* **Input Layer**: Implicitly defined by `input_shape` in the first Dense layer, corresponding to the number of features after preprocessing (12 features).
* **Hidden Layer 1**: `Dense` layer with 64 units and 'relu' activation.
* **Dropout Layer 1**: `Dropout` with a rate of 0.2 to prevent overfitting.
* **Hidden Layer 2**: `Dense` layer with 32 units and 'relu' activation.
* **Dropout Layer 2**: `Dropout` with a rate of 0.2.
* **Output Layer**: `Dense` layer with 1 unit and 'sigmoid' activation for binary classification.

The model is compiled using:
* **Optimizer**: Adam (`learning_rate=0.001`).
* **Loss Function**: Binary Crossentropy.
* **Metrics**: Accuracy.

---

## Streamlit Application ️
The `app.py` file launches a web application built with Streamlit for real-time customer churn prediction.
You can access the deployed application here: [https://customerchurnprediction-jt3eapp7slhjeq76nkygph.streamlit.app/](https://customerchurnprediction-jt3eapp7slhjeq76nkygph.streamlit.app/)

**Features:**
* User-friendly interface to input customer details.
    * Geography (Dropdown: France, Spain, Germany)
    * Gender (Dropdown: Male, Female)
    * Age (Slider: 19-92)
    * Balance (Numerical input)
    * Credit Score (Numerical input)
    * Estimated Salary (Numerical input)
    * Tenure (Slider: 0-10 years)
    * Number of Products (Slider: 1-4)
    * Has Credit Card (Dropdown: Yes/No represented as 1/0)
    * Is Active Member (Dropdown: Yes/No represented as 1/0)
* Loads the pre-trained ANN model, scaler, and encoders.
* Preprocesses the input data (encoding and scaling) to match the model's training format.
* Displays the churn prediction (likely to churn or not) and the churn probability.

---

## File Structure
customerchurnprediction/
│
├── app.py                     # Streamlit application script
├── experiments.ipynb          # Jupyter Notebook for model training and experimentation
├── requirements.txt           # Python dependencies
├── dataset/
│   └── Churn_Modelling.csv    # Dataset file
├── models/                    # Saved model, scaler, and encoders
│   ├── model.keras
│   ├── scaler.pkl
│   ├── onehot_encoder_geo.pkl
│   └── label_encoder_gender.pkl
├── logs/                      # TensorBoard logs (generated during training)
│   └── fit/
│       └── ...
└── README.md                  # This file


---

## Setup and Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/soumasnigdha/customerchurnprediction.git](https://github.com/soumasnigdha/customerchurnprediction.git)
    cd customerchurnprediction
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies using the provided `requirements.txt` file:**
    Ensure you have a `requirements.txt` file in the root of your project with the following content:
    ```txt
    tensorflow==2.19.0
    pandas
    numpy
    scikit-learn
    tensorboard
    matplotlib
    streamlit
    ipykernel
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  **To run the model training and experimentation notebook:**
    Ensure you have Jupyter Notebook or Jupyter Lab installed (covered by `ipykernel` in `requirements.txt` if you install Jupyter front-end separately).
    ```bash
    jupyter notebook experiments.ipynb
    # or
    jupyter lab experiments.ipynb
    ```
    *The notebook `experiments.ipynb` uses `os.chdir("..")` in its initial setup, implying it might be intended to run from a subdirectory. If running from the project root, you might need to adjust paths within the notebook or comment out this line.*

2.  **To run the Streamlit web application locally:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

---

## Technologies Used 🛠️
* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn** (for StandardScaler, LabelEncoder, OneHotEncoder, train_test_split)
* **TensorFlow/Keras** (version 2.19.0 specified)
* **Streamlit**
* **TensorBoard** (for logging model training)
* **Matplotlib** (likely for plotting in the notebook)
* **Pickle** (for saving/loading scaler, encoders and final model)
* **Jupyter Notebook / ipykernel** (for model development and experimentation)

---
