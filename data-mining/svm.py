import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

data_path = 'data-mining/sample_data.csv'
data = pd.read_csv(data_path)

features = ['Age', 'Purchase_Frequency', 'Total_Spent']
target = 'High_Spender'

def run_svm(data, target_column, test_size = 0.4, random_state=1):
    x = data[features]
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

    svm = SVC(kernel='linear', random_state=random_state)
    svm.fit(x_train, y_train)

    y_pred = svm.predict(x_test)

    test_results = pd.DataFrame(x_test, columns=features)
    test_results['Actual'] = y_test.values
    test_results['Predicted'] = y_pred

    print("\nPREDICTED VALUES")
    print(test_results)

    accuracy = accuracy_score(y_test, y_pred)
    print("Modelin doğruluğu:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print("\nCONFUSION MATRIX")
    print(cm)


print("\nORIGINAL DATAS SVM")
run_svm(data, target)

data.iloc[0:5, [0,1,2]] = np.nan
data.iloc[20:45, [1,2]] = np.nan
data.iloc[17, [0,1,2]] = np.nan

print("\nAFTER REPLACEMENT WITH NAN")
print(data)

data_filled = data.copy()
data_filled[features[0]] = data_filled[features[0]].fillna(data_filled[features[0]].mean().astype(int))
data_filled[features[1]] = data_filled[features[1]].fillna(data_filled[features[1]].mean().astype(int))
data_filled[features[2]] = data_filled[features[2]].fillna(data_filled[features[2]].mean().astype(int))


print("\nAFTER FILLING WITH COLUMN'S AVERAGE")
print(data_filled)

print("\nFILLED DATAS SVM")
run_svm(data_filled, target)
