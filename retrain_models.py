import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data
import os
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, 'Database', '6accident_data.csv')
df = pd.read_csv(data_path)

print(f"Loaded data with shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Add season column based on date
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['month'] = df['Date'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 3  # Winter
    elif month in [3, 4, 5]:
        return 0  # Autumn (Spring)
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 1  # Rainy (Fall)

df['season'] = df['month'].apply(get_season)

# Select features for the model (using correct column names from CSV)
feature_columns = ['Age_Band_of_Driver', 'Vehicle_Type', 'Age_of_Vehicle', 
                   'Weather_Conditions', 'Day_of_Week', 'Road_Surface_Conditions', 
                   'Light_Conditions', 'Sex_of_Driver', 'season', 'Speed_limit']

# Drop rows with missing values
df_clean = df[feature_columns + ['Accident_Severity']].dropna()
print(f"After dropping NaN: {df_clean.shape}")

# Encode categorical features
label_encoders = {}
categorical_features = ['Age_Band_of_Driver', 'Vehicle_Type', 'Weather_Conditions', 
                        'Day_of_Week', 'Road_Surface_Conditions', 'Light_Conditions', 
                        'Sex_of_Driver']

for feature in categorical_features:
    le = LabelEncoder()
    df_clean[feature] = le.fit_transform(df_clean[feature].astype(str))
    label_encoders[feature] = le

# Also encode target variable
le_target = LabelEncoder()
df_clean['Accident_Severity'] = le_target.fit_transform(df_clean['Accident_Severity'])

X = df_clean[feature_columns]
y = df_clean['Accident_Severity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)
print(f"Random Forest accuracy: {rf_score:.4f}")

# Save Random Forest model
model_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Random Forest model saved to {model_path}")

print("\nTraining Decision Tree model...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_score = dt_model.score(X_test, y_test)
print(f"Decision Tree accuracy: {dt_score:.4f}")

# Save Decision Tree model
dt_model_path = os.path.join(base_dir, 'models', 'decision_tree_model.pkl')
with open(dt_model_path, 'wb') as f:
    pickle.dump(dt_model, f)
print(f"Decision Tree model saved to {dt_model_path}")

print("\nModels retrained successfully!")
