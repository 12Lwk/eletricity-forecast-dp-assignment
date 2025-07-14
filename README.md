# Deep Learning for Short-Term Electricity Demand Forecasting

[![Project Status: Complete](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://www.repostatus.org/#inactive)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Frameworks](https://img.shields.io/badge/Frameworks-TensorFlow%20%7C%20Keras%20%7C%20Scikit--learn-orange.svg)](https://www.tensorflow.org/)

This project develops and evaluates a suite of deep learning models to accurately forecast short-term electricity demand. Using a comprehensive dataset from Spain, this work explores various neural network architectures—including LSTMs, CNNs, and advanced hybrid models—to predict hourly electricity load changes over a 24-hour horizon.

---

## **Table of Contents**

- [Project Motivation](#project-motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architectures](#model-architectures)
- [Performance Summary](#performance-summary)
- [Key Findings](#key-findings)
- [How to Run This Project](#how-to-run-this-project)
- [Future Work](#future-work)
- [Author](#author)

---

## **Project Motivation**

In the modern energy sector, accurate electricity demand forecasting is critical for maintaining grid stability, optimizing energy distribution, and integrating renewable energy sources. Traditional forecasting methods often struggle with the dynamic and non-linear patterns of energy consumption. This project leverages the power of deep learning to capture these complex temporal dependencies, aiming to produce a robust and highly accurate forecasting model for real-world applications.

---

## **Dataset**

This project utilizes the **"Hourly Energy Consumption, Generation, Prices, and Weather"** dataset available on Kaggle. It provides a rich, multivariate time series environment for model training.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
- **Content:** The dataset includes over 35,000 hourly records, merging two primary sources:
    - `energy_dataset.csv`: Data on electricity generation from various sources (fossil fuels, nuclear, solar, wind), market prices, and total load.
    - `weather_features.csv`: Hourly weather data from multiple cities in Spain, including temperature, humidity, wind speed, and cloud cover.

The target variable for this forecasting task is `delta_load`—the hourly change in electricity demand—which was found to be more effective for capturing short-term fluctuations than predicting the total load directly.

---

## **Methodology**

The project follows a standard data science workflow:

1.  **Data Preprocessing:** Loading the datasets, synchronizing timestamps, merging energy and weather data, and handling missing values using time-based interpolation.
2.  **Feature Engineering:** Creating new features to capture cyclical patterns and temporal dependencies, including:
    - Time-based features (hour, weekday).
    - Cyclical encoding (`sin`, `cos` transformations for hour and weekday).
    - Lag features and rolling statistics (mean, standard deviation).
3.  **Data Splitting:** Chronologically splitting the data into training (60%), validation (20%), and test (20%) sets to ensure the model is evaluated on unseen future data.
4.  **Model Development:** Building and training several deep learning architectures to identify the most effective approach.
5.  **Hyperparameter Tuning:** Using KerasTuner and Optuna to systematically find the optimal hyperparameters for the best-performing models.
6.  **Evaluation:** Assessing model performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Trend Direction Accuracy.

---

## **Model Architectures**

Several deep learning models were developed and evaluated, increasing in complexity:

-   **LSTM Models:**
    -   A baseline stacked LSTM network.
    -   An **LSTM with Attention** mechanism to focus on the most relevant time steps.
    -   A Multi-Task LSTM to simultaneously predict load values (regression) and trend direction (classification).
-   **CNN & Hybrid Models:**
    -   A baseline 1D-CNN for fast, local pattern extraction.
    -   A hybrid **CNN + BiLSTM + Attention** model, designed to leverage the strengths of all three components. This architecture proved to be the most effective.
    -   A Multi-Task version of the hybrid model.

---

## **Performance Summary**

After extensive training and tuning, the models demonstrated strong performance in predicting load values. The hybrid architectures consistently outperformed the simpler models.

| Model Architecture                | Test MAE | Test MSE | Trend Direction Accuracy |
| --------------------------------- | -------- | -------- | ------------------------ |
| Basic LSTM                        | 0.0120   | 0.0004   | 20.17%                   |
| LSTM + Attention                  | 0.0066   | 0.0002   | 31.01%                   |
| Basic CNN                         | 0.0075   | 0.0003   | 29.44%                   |
| **Conv1D + BiLSTM + Attention** | **0.0060** | **0.0002** | **31.67%** |
| Conv1D + BiLSTM + Attention (Tuned) | 0.0061   | 0.0002   | 31.80%                   |

The **Conv1D + BiLSTM + Attention** model emerged as the top performer, achieving the lowest prediction error while maintaining high directional accuracy.

---

## **Key Findings**

1.  **Hybrid Models Excel:** Combining CNNs (for local pattern detection) with LSTMs (for temporal dependencies) and Attention (for focus) yields the best results.
2.  **`delta_load` is a Better Target:** Predicting the *change* in load (`delta_load`) resulted in models that were more responsive to sharp fluctuations compared to predicting the absolute `total_load_actual`.
3.  **The Challenge of Trend Direction:** While the models achieved very low numerical error (MAE/MSE), accurately predicting the *direction* (up or down) of the load change at every hour proved to be the most difficult task, with the best models achieving ~32% accuracy. This is a common challenge in forecasting, as loss functions like MSE prioritize minimizing error magnitude over directional correctness.

---

## **How to Run This Project**

To replicate this analysis, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/electricity-forecasting.git](https://github.com/your-username/electricity-forecasting.git)
    cd electricity-forecasting
    ```

2.  **Set up a Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing libraries like tensorflow, pandas, scikit-learn, etc.)*

4.  **Download the dataset:**
    -   Download the data from the [Kaggle link](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) and place the `.csv` files in a `data/` directory.

5.  **Run the analysis:**
    -   Open and run the Jupyter Notebook or Python script (`main.py`) to perform the data preprocessing, model training, and evaluation.

---

## **Future Work**

While this project successfully developed a robust forecasting model, future enhancements could include:

-   **Transformer-Based Architectures:** Implementing a Transformer model, which has shown state-of-the-art performance in sequence-to-sequence tasks.
-   **Custom Loss Functions:** Designing a loss function that explicitly penalizes incorrect trend direction predictions to improve directional accuracy.
-   **Long-Term Forecasting:** Extending the model to support seasonal and long-term forecasting horizons.

---

## **Author**

-   **Lee Wen Kang**
-   [![Connect on LinkedIn](https://www.linkedin.com/in/your-profile-url](https://www.linkedin.com/in/lee-wen-kang-3b76b6188/))
