# Example: Data preprocessing (you'll need to adapt this to your specific dataset)
import pandas as pd

# Load historical data
historical_data = pd.read_csv('historical_train_data.csv')

# Load real-time data
real_time_data = pd.read_csv('real_time_train_data.csv')

# Merge historical and real-time data
merged_data = pd.merge(historical_data, real_time_data, on='train_id', how='inner')
 

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X = merged_data.drop('arrival_time', axis=1)
y = merged_data['arrival_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
 

# Make predictions on real-time data
real_time_input = preprocess_real_time_data(real_time_input)  # Replace with your preprocessing function
predicted_arrival_time = model.predict(real_time_input)

print(f'Predicted Arrival Time: {predicted_arrival_time}')

