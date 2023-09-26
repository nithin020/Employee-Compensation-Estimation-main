# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.linear_model import LinearRegression  # To build a linear regression model
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation
from sklearn.preprocessing import StandardScaler  # For feature scaling
import numpy as np  # For numerical operations

# Load the dataset
data = pd.read_csv(r"C:/Users/LENOVO/Documents/Python Projects/Employee Compensation/train_set.csv")

# Select relevant features and target variable
features = ['Salaries', 'Overtime', 'H/D']  # Features used for prediction
target = 'Total_Compensation'  # Target variable to predict

X = data[features]  # Features
y = data[target]  # Target variable

# Handle outliers
def remove_outliers(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.25)  # 1st quartile
        q3 = df[col].quantile(0.75)  # 3rd quartile
        iqr = q3 - q1  # Interquartile range
        lower_bound = q1 - 1.5 * iqr  # Lower bound for outliers
        upper_bound = q3 + 1.5 * iqr  # Upper bound for outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Filter outliers
    return df

data = remove_outliers(data, features + [target])  # Remove outliers from the dataset

X = data[features]  # Update features after removing outliers
y = data[target]  # Update target variable after removing outliers

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but often recommended)
scaler = StandardScaler()  # Initialize the feature scaler
X_train = scaler.fit_transform(X_train)  # Scale features for training data
X_test = scaler.transform(X_test)  # Scale features for testing data

# Build and train a Linear Regression model
model = LinearRegression()  # Create a linear regression model
model.fit(X_train, y_train)  # Train the model on the training data

# Make predictions
y_pred = model.predict(X_test)  # Use the trained model to make predictions on the test data

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error
r2 = r2_score(y_test, y_pred)  # Calculate R-squared

print("Mean Squared Error:", mse)  # Print mean squared error
print("R-squared:", r2)  # Print R-squared

# Add estimated Total Compensation to the dataset
data['Estimated_Total_Compensation'] = model.predict(scaler.transform(data[features]))

# Save the updated dataset with estimated values
data.to_csv(r"C:/Users/LENOVO/Documents/Python Projects/Employee Compensation/updated_dataset.csv", index=False)
