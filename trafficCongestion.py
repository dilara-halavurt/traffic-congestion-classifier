from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# Assuming your data is in a DataFrame df
df = pd.read_csv('CSV3.csv')  # replace with your filename

df['Speed'] = df['Speed'].str.replace(' km/h', '').astype(float)
# Convert categorical variables into numeric representations
le = LabelEncoder()

for col in ['Name', 'Direction', 'day/night', 'weather']:
    df[col] = le.fit_transform(df[col])

# Define features and target variable
# assuming 'congestion' is your target variable
df = df.drop('notes', axis=1)
X = df.drop('class', axis=1)
y = df['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Define the model
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Assuming we have true labels as y_test and predicted labels as y_pred
# y_test = [...]
# y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
