import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin', 'Car Name']
df = pd.read_csv(url, sep='\s+', names=column_names, na_values="?")

# Remove missing values
df = df.dropna()

# Convert MPG to liters per 100 km
df['Fuel_Consumption_L_per_100km'] = 235.215 / df['MPG']

# 1. Linear regression with one feature (horsepower) for visualization
X_horsepower = df['Horsepower'].values.reshape(-1, 1)  # Horsepower as the feature
y = df['Fuel_Consumption_L_per_100km'].values  # Fuel consumption in liters per 100 km

# Clean data by removing outliers (fuel consumption between 5 and 25 L/100 km)
mask = (y > 5) & (y < 25)
X_horsepower = X_horsepower[mask]
y = y[mask]

# Split the data
X_train_hp, X_test_hp, y_train, y_test = train_test_split(X_horsepower, y, test_size=0.2, random_state=42)

# Train the model
model_hp = LinearRegression()
model_hp.fit(X_train_hp, y_train)
print("\n=== Results for Horsepower (Visualization) ===")
print("Slope (w):", model_hp.coef_)
print("Intercept (b):", model_hp.intercept_)

# Make predictions and evaluate
y_pred_hp = model_hp.predict(X_test_hp)
print("Predictions (L/100 km):", y_pred_hp[:5])
print("Actual values (L/100 km):", y_test[:5])

mse_hp = mean_squared_error(y_test, y_pred_hp)
r2_hp = r2_score(y_test, y_pred_hp)
print("Mean Squared Error (MSE):", mse_hp)
print("R^2 Score:", r2_hp)

# Plot: Fuel consumption vs Horsepower
plt.figure(figsize=(10, 6))
plt.scatter(X_test_hp, y_test, color="blue", label="Actual Data")
plt.plot(X_test_hp, y_pred_hp, color="red", label="Model Prediction")
plt.xlabel("Horsepower")
plt.ylabel("Fuel Consumption (Liters per 100 km)")
plt.legend()
plt.title("Fuel Consumption Prediction by Horsepower")
plt.grid(True)
plt.savefig("plot_horsepower.png")  # Save the plot for GitHub
plt.show()

# 2. Linear regression with two features (horsepower + weight) for accuracy
X = df[['Horsepower', 'Weight']].values
y = df['Fuel_Consumption_L_per_100km'].values

# Clean data by removing outliers
mask = (y > 5) & (y < 25)
X = X[mask]
y = y[mask]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n=== Results for Horsepower + Weight (Accuracy) ===")
print("Slope (w):", model.coef_)
print("Intercept (b):", model.intercept_)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print("Predictions (L/100 km):", y_pred[:5])
print("Actual values (L/100 km):", y_test[:5])

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)

# 3. Additional visualization: Fuel consumption vs Weight
X_weight = df['Weight'].values.reshape(-1, 1)
y_fuel = df['Fuel_Consumption_L_per_100km'].values

# Clean data
mask = (y_fuel > 5) & (y_fuel < 25)
X_weight = X_weight[mask]
y_fuel = y_fuel[mask]

model_weight = LinearRegression()
model_weight.fit(X_weight, y_fuel)
y_pred_weight = model_weight.predict(X_weight)

plt.figure(figsize=(10, 6))
plt.scatter(X_weight, y_fuel, color="blue", label="Actual Data")
plt.plot(X_weight, y_pred_weight, color="red", label="Model Prediction")
plt.xlabel("Weight")
plt.ylabel("Fuel Consumption (Liters per 100 km)")
plt.legend()
plt.title("Fuel Consumption Prediction by Weight")
plt.grid(True)
plt.savefig("plot_weight.png")  # Save the plot for GitHub
plt.show()

# 4. Histogram of fuel consumption distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Fuel_Consumption_L_per_100km'], bins=20, color='green', alpha=0.7)
plt.xlabel("Fuel Consumption (Liters per 100 km)")
plt.ylabel("Frequency")
plt.title("Distribution of Fuel Consumption")
plt.grid(True)
plt.savefig("histogram.png")  # Save the histogram for GitHub
plt.show()
