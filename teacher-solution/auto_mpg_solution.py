# Section block 1
# Install packages first if the Colab environment needs them.
# %pip install numpy pandas matplotlib scikit-learn keras tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import Sequential, Input
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

keras.utils.set_random_seed(42)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
print("Keras backend:", keras.config.backend())

from pathlib import Path
DATA_FILE = Path(__file__).resolve().parent / "data" / "mpg.csv" if "__file__" in globals() else Path("mpg.csv")


# Section block 2
mpg_data = pd.read_csv(DATA_FILE if DATA_FILE.exists() else DATA_URL)
print(mpg_data.head().to_string(index=False))
print("Shape:", mpg_data.shape)
print("Columns:", mpg_data.columns.tolist())
print("Types:\n", mpg_data.dtypes.to_string())
print("Missing:\n", mpg_data.isna().sum().to_string())
working = mpg_data.copy().dropna(subset=["mpg"])
print("Rows removed for missing target:", len(mpg_data)-len(working))
assert mpg_data.shape == (398, 9)
assert len(working) == 398


# Section block 3
feature_columns = ["weight", "horsepower", "displacement", "model_year", "origin"]
X = working[feature_columns].copy()
y = working["mpg"].copy()
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
for name, features, targets in [("train",X_train,y_train),("validation",X_val,y_val),("test",X_test,y_test)]:
    print(name, features.shape, targets.shape)
assert set(X_train.index).isdisjoint(X_val.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_val.index).isdisjoint(X_test.index)
assert [len(X_train),len(X_val),len(X_test)] == [254,64,80]


# Section block 4
numeric_columns = ["weight", "horsepower", "displacement", "model_year"]
print("Training summaries:\n", pd.concat([X_train[numeric_columns],y_train],axis=1).describe().to_string())
print("Missing training horsepower:", X_train.horsepower.isna().sum())
print("Heaviest training vehicle:\n", working.loc[[X_train.weight.idxmax()],feature_columns+["mpg"]].to_string(index=False))
repeated = pd.concat([X_train,y_train],axis=1).duplicated().sum()
print("Repeated selected feature/target combinations:", repeated)
baseline_mpg = float(y_train.mean())
correlation = float(X_train.weight.corr(y_train))
print("Training mean MPG:", baseline_mpg)
print("Training weight/MPG correlation:", correlation)


# Section block 5
plt.figure(figsize=(7, 4))
plt.hist(y_train, bins=18, edgecolor="black")
plt.xlabel("MPG (miles per US gallon)")
plt.ylabel("Number of training vehicles")
plt.title("Training target distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.scatter(X_train["weight"], y_train, alpha=0.7)
plt.xlabel("Weight (pounds)")
plt.ylabel("MPG (miles per US gallon)")
plt.title("Training vehicle weight versus MPG")
plt.tight_layout()
plt.show()


# Section block 6
numeric_columns = ["weight", "horsepower", "displacement", "model_year"]
categorical_columns = ["origin"]

X_train_encoded = pd.get_dummies(
    X_train, columns=categorical_columns, dtype=np.float32
)

# Use the columns learned from training for held-out observations.
def encode_like_training(frame):
    encoded = pd.get_dummies(
        frame, columns=categorical_columns, dtype=np.float32
    )
    return encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

X_val_encoded = encode_like_training(X_val)
X_test_encoded = encode_like_training(X_test)


# Section block 7
print("Encoded columns:", X_train_encoded.columns.tolist())
for frame in [X_train_encoded,X_val_encoded,X_test_encoded]:
    print("Encoded shape:", frame.shape)
    assert frame.columns.equals(X_train_encoded.columns)
assert X_train_encoded.shape[1] == 7


# Section block 8
X_train_prepared = X_train_encoded.copy()
X_val_prepared = X_val_encoded.copy()
X_test_prepared = X_test_encoded.copy()
imputer = SimpleImputer(strategy="median")
X_train_prepared[numeric_columns] = imputer.fit_transform(X_train_prepared[numeric_columns])
for frame in [X_val_prepared,X_test_prepared]:
    frame[numeric_columns] = imputer.transform(frame[numeric_columns])
scaler = StandardScaler()
X_train_prepared[numeric_columns] = scaler.fit_transform(X_train_prepared[numeric_columns])
for frame in [X_val_prepared,X_test_prepared]:
    frame[numeric_columns] = scaler.transform(frame[numeric_columns])
train_X, val_X, test_X = [f.to_numpy(dtype="float32") for f in [X_train_prepared,X_val_prepared,X_test_prepared]]
train_y, val_y, test_y = [s.to_numpy(dtype="float32") for s in [y_train,y_val,y_test]]
for features,targets in [(train_X,train_y),(val_X,val_y),(test_X,test_y)]:
    print(features.shape,targets.shape,"finite:",bool(np.isfinite(features).all()))
    assert np.isfinite(features).all() and np.isfinite(targets).all()
np.testing.assert_allclose(imputer.statistics_, X_train[numeric_columns].median().to_numpy())
np.testing.assert_allclose(train_X[:,:4].mean(axis=0),0,atol=1e-6)
np.testing.assert_array_equal(train_X[:,4:],X_train_encoded.iloc[:,4:].to_numpy(dtype="float32"))
print("Training medians:", imputer.statistics_)
print("Training-only scaling means:", scaler.mean_)


# Section block 9
keras.utils.set_random_seed(42)
model = Sequential([Input(shape=(train_X.shape[1],)), Dense(16,activation="relu"),Dense(1)],name="Auto_MPG_baseline")
model.summary()
assert model.count_params() == (7*16+16)+(16+1) == 145
print("Manual total: (7*16+16)+(16+1) =",model.count_params())


# Section block 10
model.compile(optimizer="adam",loss="mean_squared_error",metrics=["mae"])
print("Compiled: Adam, mean squared error loss, mean absolute error metric")


# Section block 11
history = model.fit(train_X,train_y,epochs=300,batch_size=32,validation_data=(val_X,val_y),verbose=0)
assert len(history.history["loss"]) == 300
print("History keys:",list(history.history))
for key,values in history.history.items():
    assert np.isfinite(values).all()
    print(key,"final:",values[-1])


# Section block 12
train_mse = history.history["loss"]
val_mse = history.history["val_loss"]
epochs = range(1, len(train_mse) + 1)

plt.figure(figsize=(7, 4))
plt.plot(epochs, train_mse, label="Training MSE")
plt.plot(epochs, val_mse, label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE (MPG squared)")
plt.title("Auto MPG MLP: training and validation loss")
plt.legend()
plt.tight_layout()
plt.show()


# Section block 13
train_mae = history.history["mae"]
val_mae = history.history["val_mae"]
epochs = range(1, len(train_mae) + 1)

plt.figure(figsize=(7, 4))
plt.plot(epochs, train_mae, label="Training MAE")
plt.plot(epochs, val_mae, label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (MPG)")
plt.title("Auto MPG MLP: training and validation MAE")
plt.legend()
plt.tight_layout()
plt.show()


# Section block 14
print("Prediction before experiment: fewer hidden neurons reduce capacity; validation MAE could improve through less overfitting or worsen through underfitting.")
keras.utils.set_random_seed(42)
candidate = Sequential([Input(shape=(train_X.shape[1],)),Dense(8,activation="relu"),Dense(1)],name="Auto_MPG_candidate")
candidate.compile(optimizer="adam",loss="mean_squared_error",metrics=["mae"])
candidate_history = candidate.fit(train_X,train_y,epochs=300,batch_size=32,validation_data=(val_X,val_y),verbose=0)
assert len(candidate_history.history["loss"]) == 300
assert candidate.count_params() == 73
comparison = pd.DataFrame({"model":["baseline: 16 hidden","candidate: 8 hidden"],"epochs":[300,300],"final_val_mae":[history.history["val_mae"][-1],candidate_history.history["val_mae"][-1]],"final_val_mse":[history.history["val_loss"][-1],candidate_history.history["val_loss"][-1]]})
print(comparison.to_string(index=False))
candidate_wins = candidate_history.history["val_mae"][-1] < history.history["val_mae"][-1]
final_model = candidate if candidate_wins else model
selected_name = "8-neuron candidate" if candidate_wins else "16-neuron baseline"
print("Selected by validation MAE:",selected_name)


# Section block 15
test_mse, test_mae = [float(v) for v in final_model.evaluate(test_X,test_y,verbose=0)]
baseline_predictions = np.full(test_y.shape,baseline_mpg,dtype="float64")
baseline_test_mae = float(np.mean(np.abs(baseline_predictions-test_y)))
improvement = baseline_test_mae-test_mae
print("Selected model:",selected_name)
print("Test MSE (MPG squared):",test_mse)
print("Test MAE (MPG):",test_mae)
print("Training-mean baseline test MAE (MPG):",baseline_test_mae)
print("Baseline MAE minus MLP MAE (positive means improvement):",improvement)
assert np.isfinite([test_mse,test_mae,baseline_test_mae]).all()


# Section block 16
five_predictions = final_model.predict(test_X[:5],verbose=0).ravel()
five_results = pd.DataFrame({"row_index":X_test.index[:5],"actual_mpg":test_y[:5],"predicted_mpg":five_predictions,"signed_error_mpg":five_predictions-test_y[:5],"absolute_error_mpg":np.abs(five_predictions-test_y[:5])})
print(five_results.to_string(index=False))
print("Largest error among these five, original row index:",five_results.loc[five_results.absolute_error_mpg.idxmax(),"row_index"])


# Section block 17
new_vehicle = pd.DataFrame([{"weight":3000.,"horsepower":100.,"displacement":200.,"model_year":76,"origin":"usa"}])
new_encoded = encode_like_training(new_vehicle)
new_encoded[numeric_columns] = imputer.transform(new_encoded[numeric_columns])
new_encoded[numeric_columns] = scaler.transform(new_encoded[numeric_columns])
assert new_encoded.columns.equals(X_train_encoded.columns)
new_X = new_encoded.to_numpy(dtype="float32")
assert new_X.shape == (1,7) and np.isfinite(new_X).all()
new_mpg = float(final_model.predict(new_X,verbose=0).ravel()[0])
print("Hypothetical vehicle estimated MPG:",new_mpg)
print("This is a model estimate, not a measured fuel-efficiency value.")
