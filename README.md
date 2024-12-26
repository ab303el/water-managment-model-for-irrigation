# Water Management Model for Irrigation

## Project Overview

This project focuses on predicting irrigation decisions for water management in agriculture. By using machine learning algorithms, such as MLP (Multi-layer Perceptron), LSTM (Long Short-Term Memory), and GRU (Gated Recurrent Unit), the model predicts whether irrigation is necessary based on 18 soil chemical properties. The model was developed as part of a thesis project aimed at improving irrigation efficiency in agricultural systems.

## Problem Statement

In many parts of the world, water resources are limited, and efficient water management in irrigation is essential. Traditional irrigation practices often rely on predetermined schedules and manual intervention, which may not be optimal. By leveraging soil chemical data and machine learning techniques, this project seeks to automate irrigation decisions and optimize water usage.

## Dataset

The dataset used in this project includes soil chemical properties such as pH, phosphorus (P), potassium (K), calcium (Ca), magnesium (Mg), manganese (Mn), sulfur (S), copper (Cu), boron (B), zinc (Zn), sodium (Na), iron (Fe), aluminum (Al), silicon (Si), cobalt (Co), molybdenum (Mo), and electrical conductivity (EC). Additionally, the dataset includes information about the crop type, soil type, and the location of the samples.

- Features: 18 chemical properties of soil.
- Target variable: Irrigation decision (1 for irrigate, 0 for no irrigation).

## Technologies Used

- Programming Language: Python
- Machine Learning Libraries: 
  - TensorFlow/Keras (for MLP, LSTM, and GRU models)
  - scikit-learn (for preprocessing and evaluation)
- Data Analysis: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Environment: Jupyter Notebook, Anaconda

## Model Architecture

The project utilizes three deep learning models for classification:

1. MLP (Multi-Layer Perceptron): A feed-forward neural network with multiple layers for predicting irrigation decisions based on soil chemical properties.
2. LSTM (Long Short-Term Memory): A type of recurrent neural network that captures temporal dependencies in the data, used for modeling time-series data of soil chemical properties.
3. GRU (Gated Recurrent Unit): Similar to LSTM, used for sequence prediction with fewer parameters, making it computationally efficient.

## Steps

1. Data Preprocessing: The dataset is cleaned, missing values are handled, and features are normalized.
2. Model Training: The models (MLP, LSTM, and GRU) are trained using the preprocessed dataset.
3. Model Evaluation: The models are evaluated on their accuracy and performance using various metrics (e.g., accuracy, precision, recall, F1-score).
4. Prediction: The best-performing model is used to predict irrigation decisions based on new input data.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ab303el/water-managment-model-for-irrigation.git
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook or Python scripts to explore the data and run the models:
   ```bash
   jupyter notebook
   ```

4. The code will process the data and generate predictions for irrigation.

## Results

The models are evaluated based on their ability to predict irrigation decisions, and the final model's performance is reported. The best model is used to make predictions, and its results can be used to automate irrigation decisions in real-world agricultural systems.

## Future Work

- Implementing real-time prediction using live sensor data.
- Integrating the model with an irrigation control system for automated water management.
- Expanding the dataset to include more features like weather conditions and crop growth stages.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

