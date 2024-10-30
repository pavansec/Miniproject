import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Feature scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data_scaled, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier()

# Perform grid search for best parameters
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(x_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict and evaluate the model
y_predict = best_model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model and scaler
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler}, f)
