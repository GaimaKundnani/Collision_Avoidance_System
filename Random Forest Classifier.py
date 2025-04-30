import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = "Motor_Vehicle_Collisions_-_Crashes.csv"
df = pd.read_csv(file_path)

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Create target variable: Injury or Not (1 if injury, else 0)
df['INJURY_FLAG'] = df['NUMBER OF PERSONS INJURED'].apply(lambda x: 1 if x > 0 else 0)

# Optional: parse date
if 'CRASH DATE' in df.columns and 'CRASH TIME' in df.columns:
    df['CRASH DATETIME'] = pd.to_datetime(df['CRASH DATE'] + ' ' + df['CRASH TIME'], errors='coerce')
    df['HOUR'] = df['CRASH DATETIME'].dt.hour
    df['DAY_OF_WEEK'] = df['CRASH DATETIME'].dt.dayofweek

# Plot 1: Number of collisions by hour
if 'HOUR' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='HOUR', palette='coolwarm')
    plt.title("Collisions by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Number of Collisions")
    plt.tight_layout()
    plt.show()

# Plot 2: Collisions by borough
if 'BOROUGH' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='BOROUGH', palette='Set2', order=df['BOROUGH'].value_counts().index)
    plt.title("Collisions by Borough")
    plt.xlabel("Borough")
    plt.ylabel("Number of Collisions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot 3: Top contributing factors
if 'CONTRIBUTING FACTOR VEHICLE 1' in df.columns:
    top_factors = df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().drop('Unspecified', errors='ignore').head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(y=top_factors.index, x=top_factors.values, palette='magma')
    plt.title("Top 10 Contributing Factors")
    plt.xlabel("Number of Collisions")
    plt.ylabel("Contributing Factor")
    plt.tight_layout()
    plt.show()
# Plot Injury vs No Injury
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='INJURY_FLAG', palette='Set2')
plt.title("Injury vs No Injury")
plt.xlabel("Injury Occurred")
plt.ylabel("Number of Collisions")
plt.xticks([0, 1], ["No Injury", "Injury"])
plt.tight_layout()
plt.show()

vehicle_cols = ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 
                'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5']
df['NUMBER OF VEHICLES INVOLVED'] = df[vehicle_cols].notnull().sum(axis=1)
# Select features (you can modify these based on available columns)
features = ['HOUR', 'DAY_OF_WEEK', 'NUMBER OF VEHICLES INVOLVED']
df['NUMBER OF VEHICLES INVOLVED'] = df[['NUMBER OF VEHICLES INVOLVED']].fillna(0)

# Drop rows with missing feature values
df_ml = df.dropna(subset=features + ['INJURY_FLAG'])

X = df_ml[features]
y = df_ml['INJURY_FLAG']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
#model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Accuracy Score ---")
print(accuracy_score(y_test, y_pred)*100)
