import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel('/Users/Mihir/Documents/GeneratedData.xlsx')

features = ['About how many social events do you attend per week?', 'Do you live On or Off Campus?']
target = 'Do you consider yourself an introvert or extrovert?'

campusLE = LabelEncoder()
personalityLE = LabelEncoder()

df['Do you live On or Off Campus?'] = campusLE.fit_transform(df['Do you live On or Off Campus?'])
df[target] = personalityLE.fit_transform(df[target])

def tenplus(x):
    if isinstance(x, str) and '+' in x:
        return 10  
    try:
        return float(x)  
    except ValueError:
        return None  

df['About how many social events do you attend per week?'] = df['About how many social events do you attend per week?'].apply(tenplus)

df = df.dropna(subset=features)

X = df[features]
Y = df[target]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)

RandomForest.fit(Xtrain, Ytrain)

Ypredict = RandomForest.predict(Xtest)

accuracy = accuracy_score(Ytest, Ypredict)
print(f"Accuracy: {accuracy:.2f}")

FeatureWeight = pd.DataFrame({'feature': features, 'importance': RandomForest.feature_importances_})
print("\nFeature Importance:")
print(FeatureWeight.sort_values('importance', ascending=False))

Xall = df[features]
Yall = df[target]

Predictions = RandomForest.predict(Xall)
PredictionsLabels = personalityLE.inverse_transform(Predictions)

print("\nPredictions for all rows:")
for i, (true, pred) in enumerate(zip(personalityLE.inverse_transform(Yall), PredictionsLabels), 1):
    print(f"Row {i}: True - {true}, Predicted - {pred}")

Accuracy = accuracy_score(Yall, Predictions)
print(f"\nOverall Accuracy: {Accuracy:.2f}")
