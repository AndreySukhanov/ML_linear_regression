# ML_linear_regression: Fuel prediction

## Predicting fuel consumption based on machine power and weight

## Description
This project predicts fuel consumption (in liters per 100 km) for cars based on their horsepower and weight. Linear regression from Scikit-learn is used.

## Data
- Source: UCI Machine Learning Repository (Auto MPG dataset).
- Traits: power, weight
- Target variable: fuel consumption (liters per 100 km)

## Results
- Accuracy (R^2): 0.8-0.9
- Error (MSE): 5-10

## Visualization
- [Graph: fuel consumption vs power](https://github.com/AndreySukhanov/ML_linear_regression/blob/61aebb97ffaf013154eab7c04590e09d2f453993/plot_horsepower.png)
- [Histogram: fuel consumption distribution](https://github.com/AndreySukhanov/ML_linear_regression/blob/922e7f0c4042aa89afc230dfedb2d561ae0608a6/histogram.png)
- [Graph: fuel consumption vs weight](https://github.com/AndreySukhanov/ML_linear_regression/blob/d46f4bc2a21d5490df65beb0b31d66b228819267/plot_weight.png)

## Requirements
- Python 3.x
- Libraries: numpy, pandas, matplotlib, sklearn

## How to run
1. Install the libraries: `pip install numpy pandas matplotlib scikit-learn`.
2. Download the dataset or use the URL
3. Run `fuel_prediction.py`.
