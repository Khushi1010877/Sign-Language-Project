import pickle
import numpy as np
import collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

data_dict=pickle.load(open('./data.pickle','rb'))  #reopen the same dataset
EXPECTED_LENGTH = 42

# Ensure all samples have the same length
fixed_data = []
for sample in data:
    if len(sample) < EXPECTED_LENGTH: #sample is too short
        sample.extend([0] * (EXPECTED_LENGTH - len(sample)))  # fill with zeros
    elif len(sample) > EXPECTED_LENGTH:  #sample is too long
        sample = sample[:EXPECTED_LENGTH]  # Trim extra values
    fixed_data.append(sample)

# Convert to NumPy arrays
data = np.array(fixed_data, dtype=np.float32)
labels = np.array(labels)

# Ensure data is not empty after conversion
if data.size == 0:
    print("Error: Data preprocessing resulted in an empty dataset!")
    exit()

#normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)
# Check class distribution
label_counts = collections.Counter(labels)
print("Class Distribution Before Splitting:", label_counts)

# Ensure there are at least two unique labels
if len(label_counts) < 2:
    print("Error: Not enough unique labels for training!")
    exit()

# train the model
model = RandomForestClassifier()
X_train,X_test,y_train,y_test = train_test_split(data,labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('{}% of accuracy'.format(accuracy*100))

unique_predictions = set(y_pred)
print("Unique Predictions:", unique_predictions)

f = open('model.p', 'wb')  #open model.p in write binary mode wb
pickle.dump({'model': model}, f)  # save model 
f.close()