# Universal AutoML for Tabular Data

## Overview
Universal AutoML is a Python-based automated machine learning (AutoML) system that detects the problem type (classification, regression, clustering, or anomaly detection) and applies suitable machine learning models using AutoSklearn, KMeans, or IsolationForest. It also includes data preprocessing features such as handling missing values, encoding categorical variables, and normalizing numerical data.

---

## Features
- Automatic problem type detection (binary classification, multi-class classification, regression, or clustering/anomaly detection).
- Data preprocessing using SimpleImputer, LabelEncoder, and StandardScaler.
- Model selection using AutoSklearn for classification and regression.
- KMeans clustering or IsolationForest for anomaly detection.
- Model evaluation and scoring.

---

## Requirements
Make sure you have the following installed:

- Python 3.8 - 3.11
- NumPy
- Pandas
- Scikit-Learn
- AutoSklearn

---

## Installation
1. **Create a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate # On Linux/macOS
    env\Scripts\activate   # On Windows
    ```

2. **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn auto-sklearn
    ```

3. **Additional Dependencies (Linux/macOS):**
    ```bash
    sudo apt-get install build-essential swig python3-dev
    ```
    **On Windows:** Ensure you have Visual Studio Build Tools installed.

---

## Usage

1. Save your dataset as a CSV or load a built-in dataset using `sklearn.datasets`.

2. Example for using the UniversalAutoML class with the Iris dataset:
    ```python
    from sklearn.datasets import load_iris
    import pandas as pd
    from UniversalAutoML import UniversalAutoML

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    automl = UniversalAutoML()
    automl.fit(X, y)

    predictions = automl.predict(X)
    print("Predictions:", predictions)
    ```

---

## Methods
- `detect_problem_type(target_column)`
  - Automatically detects the type of problem.
- `preprocess_data(X)`
  - Cleans and preprocesses data using imputation, encoding, and scaling.
- `train_model(X_train, y_train, problem_type)`
  - Trains an appropriate model using AutoSklearn, KMeans, or IsolationForest.
- `evaluate_model(X_test, y_test, problem_type)`
  - Evaluates the model using suitable metrics.
- `fit(X, y)`
  - Performs end-to-end training.
- `predict(X)`
  - Predicts outputs using the trained model.

---

## Notes
- AutoSklearn requires `SWIG` and a compatible C++ compiler.
- The default AutoSklearn training time is set to 120 seconds with a 30-second limit per run. Adjust as needed.

---

## License
This project is licensed under the MIT License. Feel free to modify and use it as per your requirements.

---

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue on the GitHub repository.

---

## Author
Jhon Lloyd F. Omblero

For questions or inquiries, contact me at omblero.jhonlloyd.04@gmail.com.

