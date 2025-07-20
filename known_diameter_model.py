import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv('dataset/known_diameter_input.csv')
output = pd.read_csv('dataset/known_diameter_output.csv')

# Combine input and output
df = pd.concat([data, output[['z1', 'z2', 'b', 'BHN1', 'BHN2', 'weight']]], axis=1)

# Prepare features and labels
X = data.drop(columns=['material'])  # drop non-numeric/categorical columns
y = output[['z1', 'z2', 'b', 'BHN1', 'BHN2', 'weight']]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

sample = X_test.iloc[[0]]
prediction = model.predict(sample)
print("Sample prediction output:", prediction)