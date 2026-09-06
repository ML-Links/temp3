
### Part 3: Guided Regression (Auto MPG Dataset)

**Goal:** Learn the regression pipeline by predicting a car's fuel efficiency (Miles Per Gallon, or MPG) using a Linear Regression algorithm. 

Because we are predicting a continuous number (MPG) rather than a category, this is a **Regression** task.

#### Step 3.1: Load Pre-Cleaned Auto MPG Data
This dataset contains attributes of various cars (like weight, horsepower, and cylinders). The target variable we want to predict is `mpg`.

```python
# Load pre-cleaned Auto MPG dataset
url_mpg = "https://raw.githubusercontent.com/ML-Course-2026/session3/refs/heads/main/datasets/cars/auto-mpg.csv"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']

# The na_values="?" parameter tells Pandas to treat "?" as a missing value (NaN)
mpg_df = pd.read_csv(url_mpg, names=column_names, na_values="?", header=0)

# Basic Cleaning (We will dive much deeper into this in Part 5)
# 1. Drop the car name because it is a text identifier, not a mathematical feature.
mpg_df = mpg_df.drop('car_name', axis=1) 

# 2. Fill missing horsepower values with the median.
mpg_df['horsepower'].fillna(mpg_df['horsepower'].median(), inplace=True)

# 3. Drop 'origin' for now. It is a category (1=USA, 2=Europe, 3=Japan), but 
# treating it as a raw number confuses the model. We will fix this properly in Part 5.
mpg_df = mpg_df.drop('origin', axis=1)

# Drop any remaining rows with missing data
mpg_df.dropna(inplace=True) 

print("Auto MPG Data (first 5 rows):")
print(mpg_df.head())
print("\nData Info:")
mpg_df.info()
```

#### Step 3.2: Define Features (X) and Target (y)
We separate the target we want to predict (`mpg`) from the features we will use to make the prediction.

```python
# Define features (X) and target (y)
X_mpg = mpg_df.drop('mpg', axis=1)
y_mpg = mpg_df['mpg']

print("\nFeatures (X):")
print(X_mpg.head(2))
print("\nTarget (y):")
print(y_mpg.head(2))
```

#### Step 3.3: Split Data
We split the data into 80% training and 20% testing. 

```python
# Split the data
X_train_mpg, X_test_mpg, y_train_mpg, y_test_mpg = train_test_split(
    X_mpg, y_mpg, test_size=0.2, random_state=42
)

print(f"\nTraining set shape: X={X_train_mpg.shape}, y={y_train_mpg.shape}")
print(f"Testing set shape: X={X_test_mpg.shape}, y={y_test_mpg.shape}")
```

> [!NOTE]  
> Notice that we did **not** use `stratify=y_mpg` here like we did in Parts 1 and 2. Stratification ensures an equal balance of categories. Because regression deals with infinite possible continuous numbers (e.g., 25.1 MPG, 25.2 MPG), stratification does not mathematically apply here.

#### Step 3.4: Create and Train a Linear Regression Model
Unlike a Decision Tree that creates a flowchart of "if/then" rules, a Linear Regression model uses math to draw a "line of best fit" through the data points.

```python
# Create a Linear Regression model
lr_mpg = LinearRegression()

# Train the model on the training data
lr_mpg.fit(X_train_mpg, y_train_mpg)

print("\nLinear Regression model trained successfully.")
```

#### Step 3.5: Make Predictions on Test Data
We pass the hidden test features (`X_test_mpg`) to our trained model and ask it to guess the MPG for those cars.

```python
# Make predictions
y_pred_mpg = lr_mpg.predict(X_test_mpg)

# Display first 5 predictions vs actual values
print("\nFirst 5 Predictions:", y_pred_mpg[:5])
print("First 5 Actual Values:", y_test_mpg[:5].values)
```

#### Step 3.6: Evaluate the Model
Because we are predicting a continuous number, concepts like "Accuracy" (being 100% exactly right) do not apply. If a car's actual MPG is 25.0, and our model predicts 24.9, it is technically "wrong," but it is still a very good prediction. 

Therefore, for regression, we evaluate the model by measuring the **Error** (the distance between the prediction and the actual value).

```python
# Evaluate the model
mae_mpg = mean_absolute_error(y_test_mpg, y_pred_mpg)
mse_mpg = mean_squared_error(y_test_mpg, y_pred_mpg)
rmse_mpg = np.sqrt(mse_mpg) # We calculate RMSE by taking the square root of MSE
r2_mpg = r2_score(y_test_mpg, y_pred_mpg)

print(f"\nMean Absolute Error (MAE): {mae_mpg:.4f}")
print(f"Mean Squared Error (MSE): {mse_mpg:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_mpg:.4f}")
print(f"R-squared (R²): {r2_mpg:.4f}")
```

> [!IMPORTANT]  
> **Understanding Regression Metrics:**
> *   **MAE:** The average of all the errors. If MAE is 2.5, it means our model's predictions are off by an average of 2.5 MPG. It is the easiest to understand.
> *   **MSE:** Averages the *squares* of the errors. Squaring the numbers heavily penalizes the model for making massive mistakes, but it changes the units (e.g., "MPG squared").
> *   **RMSE:** The square root of MSE. This brings the error back into the original units (MPG), making it readable while still penalizing large errors.
> *   **R² (R-Squared):** A score from 0 to 1 indicating how well the model fits the data. 1.0 is a perfect fit. 0.0 means the model is no better than just guessing the average MPG every single time.

#### Step 3.7: Visualize Predictions vs Actuals
Numbers are great, but plotting the data helps us truly understand how the model behaves. 

```python
# Plot Actual vs Predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_mpg, y_pred_mpg, alpha=0.7)

# Draw a red dashed line representing "Perfect Prediction"
plt.plot([y_test_mpg.min(), y_test_mpg.max()], [y_test_mpg.min(), y_test_mpg.max()], '--r', linewidth=2) 
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title('Actual vs. Predicted MPG')
plt.show()

# Plot Residuals (The Errors)
residuals_mpg = y_test_mpg - y_pred_mpg
plt.figure(figsize=(10, 6))
sns.histplot(residuals_mpg, kde=True)
plt.xlabel('Residuals (Actual MPG - Predicted MPG)')
plt.title('Distribution of Residuals')
plt.show()
```

> [!TIP]  
> **How to read these charts:**
> 1.  **Scatter Plot:** You want all the blue dots to hug the red dashed line as tightly as possible. Dots far away from the line are poor predictions.
> 2.  **Residual Plot:** This shows the distribution of your model's mistakes. Ideally, this should look like a bell curve centered perfectly on `0`. If the peak is shifted to the left or right, your model is systematically predicting too high or too low.

###  Knowledge Check (Part 3)

Review the concepts covered in this section. Try to answer the questions before viewing the solutions.

**Question 1: The Accuracy Trap**
If you try to run the code `accuracy_score(y_test_mpg, y_pred_mpg)` on this model, Python will throw a massive error and crash. Why?

<details>
<summary><strong>View Answer</strong></summary>

`accuracy_score` is strictly a **Classification** metric. It checks if the prediction perfectly matches the target (True or False). 

In Regression, predictions are almost never perfectly exact. If the actual MPG is 20.0 and the model predicts 20.000001, `accuracy_score` will mark it as completely wrong, which is useless. For continuous numbers, we must measure the *distance* of the error (MAE/RMSE), not boolean correctness.
</details>

<br>

**Question 2: MAE vs RMSE**
Imagine you have a model predicting housing prices. The MAE is `$10,000`, but the RMSE is `$50,000`. Why would the RMSE be so much higher than the MAE, and what does this tell you about the model's performance?

<details>
<summary><strong>View Answer</strong></summary>

RMSE penalizes *large* errors significantly more than MAE because it squares the errors before averaging them. 

If RMSE is much higher than MAE, it tells you that while the model is usually quite accurate (off by about $10,000 on average), there are a handful of predictions where the model makes **massive, catastrophic mistakes** (e.g., overvaluing a house by $200,000). The squaring effect of RMSE highlights these outliers.
</details>

<br>

**Question 3: Missing Values**
In Step 3.1, we filled the missing `horsepower` values with the median horsepower of the entire dataset. Why did we use the median instead of simply filling the missing spaces with `0`?

<details>
<summary><strong>View Answer</strong></summary>

If you fill a missing horsepower value with `0`, you are telling the machine learning algorithm that the car literally has zero horsepower. The algorithm will treat that `0` as a real, mathematical fact and adjust its line of best fit accordingly, which will ruin the prediction (a car with 0 horsepower would not move). 

Filling missing values with the median (or mean) is a safe way to preserve the row of data without wildly distorting the mathematical average of the dataset.
</details>


<br>


**Question 4: Why Is Stratification Not Used in Regression?**
Why did we **not** use `stratify=y_mpg` in the regression example (e.g., predicting MPG) like we did in the classification examples (e.g., Iris, Titanic datasets)? Explain why stratification doesn't apply in the context of regression.

<details>  
<summary><strong>View Answer</strong></summary>

In classification problems, stratification is often used to ensure that the data is split evenly across the different classes, making sure each split (training and testing) has the same proportion of categories. This is crucial when working with imbalanced datasets to avoid biased models that overfit or underfit due to uneven class distributions.

However, **stratification does not apply in regression** problems, such as predicting continuous values like **MPG** (miles per gallon). In regression, the target variable is continuous, meaning it can take on any value within a range (e.g., 25.1 MPG, 25.2 MPG, etc.). This makes it impossible to categorize the target variable into discrete "classes" for stratification.

**Example:**
In the Iris dataset (classification), you have classes like 'setosa', 'versicolor', and 'virginica'. If you stratify by the target variable, the data will be split proportionally to ensure each category is equally represented. In the Titanic dataset (classification), you also have categories like 'survived' or 'did not survive'.

In contrast, with a regression problem like predicting **MPG**, where values like 25.1, 25.2, and 25.3 are all possible and continuous, there's no meaningful way to "split" the data into strata. Stratifying by the target would not help because the distribution of values is not discrete.

Thus, in regression tasks, it's more common to rely on **random splitting** of the dataset into training and test sets, without stratification, to ensure that all possible values of the target variable are present in both splits but without forcing a specific balance between categories.

**In summary:**

* **Stratification** works for **classification** (discrete categories).
* **Regression** deals with continuous values, so **stratification is not necessary** or meaningful.

For example, in the regression task predicting **MPG**, we would use a simple split without stratification:

```python
train_test_split(X, y_mpg, test_size=0.2, random_state=42)
```

This ensures a proper random distribution of the data without trying to force balance over continuous values.

</details>

<br>


---

### Part 4: Practice Regression (Diabetes Dataset)

> [!NOTE]  
> Just like in Part 2, the original plan for this lab was for you to write this code from scratch. We have provided the sample solution below. Your task is to run the code and study how the regression pipeline is applied to a completely different set of data.

**Goal:** Apply the regression pipeline to the `scikit-learn` Diabetes dataset. This dataset contains physiological measurements of patients, and the goal is to predict how much their disease will progress one year later.

#### Step 4.1: Load Diabetes Data
This dataset is built into `sklearn`. The features (like BMI, blood pressure, etc.) have already been scaled and cleaned by the dataset creators.

```python
# Load the Diabetes dataset
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Add the target column to the dataframe
diabetes_df['target'] = diabetes.target 

print("Diabetes Data (first 5 rows):")
print(diabetes_df.head())
print("\nData Info:")
diabetes_df.info()
```

> [!TIP]  
> **What exactly is the "Target" here?** 
> In the Auto MPG dataset, the target was obvious (Miles Per Gallon). In this dataset, the target is a numerical score (ranging from roughly 25 to 346) that represents how much the patient's diabetes progressed exactly one year after the baseline measurements were taken. A higher number means worse disease progression.

#### Step 4.2: Define Features (X) and Target (y)
We separate the target score from the patient features.

```python
# Define X_diabetes and y_diabetes
X_diabetes = diabetes_df.drop('target', axis=1)
y_diabetes = diabetes_df['target']

# Print shapes to verify
print(f"\nDiabetes features shape: {X_diabetes.shape}")
print(f"Diabetes target shape: {y_diabetes.shape}")
```

#### Step 4.3: Split Data
We split into 80% training and 20% testing. Again, because this is regression, we do not use `stratify`.

```python
# Split the data into X_train, X_test, y_train, y_test
X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

# Print shapes to verify
print(f"\nTraining set shape: X={X_train_diabetes.shape}, y={y_train_diabetes.shape}")
print(f"Testing set shape: X={X_test_diabetes.shape}, y={y_test_diabetes.shape}")
```

#### Step 4.4: Create and Train a Linear Regression Model
We initialize a new Linear Regression model and fit it strictly to the training data.

```python
# Create and train the model 
lr_diabetes = LinearRegression()
lr_diabetes.fit(X_train_diabetes, y_train_diabetes)

print("\nDiabetes Linear Regression model trained.")
```

#### Step 4.5: Make Predictions
We pass the hidden test data to the model and ask it to predict the disease progression scores.

```python
# Make predictions 
y_pred_diabetes = lr_diabetes.predict(X_test_diabetes)

# Display first 5 predictions vs actual values
print("\nFirst 5 Predictions:", y_pred_diabetes[:5])
print("First 5 Actual Values:", y_test_diabetes[:5].values)
```

#### Step 4.6: Evaluate the Model
We calculate the error metrics to see how close our predictions are to the actual patient outcomes.

```python
# Evaluate the model 
mae_diabetes = mean_absolute_error(y_test_diabetes, y_pred_diabetes)
mse_diabetes = mean_squared_error(y_test_diabetes, y_pred_diabetes)
rmse_diabetes = np.sqrt(mse_diabetes)
r2_diabetes = r2_score(y_test_diabetes, y_pred_diabetes)

print(f"\nDiabetes Model Evaluation:")
print(f"MAE: {mae_diabetes:.4f}")
print(f"MSE: {mse_diabetes:.4f}")
print(f"RMSE: {rmse_diabetes:.4f}")
print(f"R²: {r2_diabetes:.4f}")
```

#### Step 4.7: Visualize Predictions vs Actuals
Visualizing the results helps us identify if our model has any specific biases (e.g., if it struggles to predict very high disease progression).

```python
# Plot Actual vs Predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_diabetes, y_pred_diabetes, alpha=0.7)
plt.plot([y_test_diabetes.min(), y_test_diabetes.max()], [y_test_diabetes.min(), y_test_diabetes.max()], '--r', linewidth=2)
plt.xlabel('Actual Progression')
plt.ylabel('Predicted Progression')
plt.title('Actual vs. Predicted Diabetes Progression')
plt.show()

# Plot Residuals
residuals_diabetes = y_test_diabetes - y_pred_diabetes
plt.figure(figsize=(10, 6))
sns.histplot(residuals_diabetes, kde=True)
plt.xlabel('Residuals (Actual - Predicted)')
plt.title('Distribution of Residuals (Diabetes)')
plt.show()
```

### Knowledge Check (Part 4)

Review the concepts covered in this section. Try to answer the questions before viewing the solutions.

**Question 1: Interpreting the R-Squared (R²) Score**
Compare the R² score you received for the Auto MPG model (Part 3) with the R² score for the Diabetes model (Part 4). 
1. Which model performed better? 
2. What does a lower R² score tell you about the dataset itself?

<details>
<summary><strong>View Answer</strong></summary>

1. The Auto MPG model performed significantly better. The Auto MPG R² score usually hovers around `0.80` (80% of the variance explained), while the Diabetes R² score usually hovers around `0.45` (only 45% of the variance explained).
2. A lower R² score tells you that the features provided to the model (BMI, blood pressure, etc.) are **not sufficient** to perfectly predict disease progression. Human biology is incredibly complex. A score of ~0.45 means our model captures some real underlying trends, but there is a massive amount of "noise" or unmeasured factors (like genetics or daily diet) affecting the patient that our model simply cannot see.
</details>

<br>

**Question 2: Making Sense of the MAE**
In Step 4.6, you printed the Mean Absolute Error (MAE). If the MAE is roughly `42.7`, how do you explain that number to a doctor who has no background in data science?

<details>
<summary><strong>View Answer</strong></summary>

You would explain it by saying: *"When this AI predicts a patient's disease progression score for next year, its guess will be off by an average of 43 points, either too high or too low."*

Because the disease progression scores range from roughly 25 to 346, being off by 43 points means the model provides a decent "ballpark" estimate, but it is not precise enough to make highly exact medical decisions.
</details>

<br>

**Question 3: Examining the Scatter Plot**
Look closely at the first chart generated in Step 4.7 (Actual vs. Predicted). Notice that as the Actual Progression (the x-axis) gets higher, the blue dots tend to fall *below* the red dashed line. What does this indicate about the model's behavior?

<details>
<summary><strong>View Answer</strong></summary>

When dots fall below the red line of perfect prediction, it means the model is **underpredicting**. 

This chart visually proves that our Linear Regression model struggles with the sickest patients. When a patient's actual disease progression is very severe (e.g., above 250), our model consistently guesses numbers that are too low. In a medical context, this is a critical flaw, because underpredicting disease severity could lead to a patient not receiving enough care.
</details>



---

### Part 5: The Reality - Data Preprocessing

**Goal:** Understand *why* preprocessing is crucial and see *how* the raw Auto MPG data needed cleaning before we could use it effectively in Part 3.

**The Scenario:** In Parts 1-4, we used data that was mostly ready for modeling. Real-world data is almost never that clean! It often has missing values, non-numeric data types, and features on different scales. Models usually require numerical input and perform better when data is scaled.

**Let's revisit the Auto MPG dataset, but load the *raw* version this time.**

**Step 5.1: Load RAW Data and Initial Inspection**
Notice the `na_values='?'`: this tells pandas to treat '?' as missing.

```python
import pandas as pd

# Load the raw data again
url_mpg = "https://raw.githubusercontent.com/ML-Course-2026/session3/main/datasets/cars/auto-mpg.csv"

# Define column names (to ensure consistency)
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']

# Read CSV correctly (comma-separated)
raw_mpg_df = pd.read_csv(url_mpg, names=column_names, na_values="?", header=0)

# Display first rows
print("Raw Auto MPG Data (first 5 rows):")
print(raw_mpg_df.head())

# Data info
print("\nRaw Data Info:")
raw_mpg_df.info()

# Check for missing values
print("\nMissing values count:")
print(raw_mpg_df.isnull().sum())
```
*   **Observation:** `horsepower` has missing values (6). Also notice `horsepower` is an `object` (text) type, because of the '?' before we handled `na_values`. `origin` is numeric but represents categories (1: USA, 2: Europe, 3: Japan). `car_name` is text and probably not useful as a raw feature.

**Step 5.2: Handling Missing Values (Imputation)**
We need to fill or remove missing values. For `horsepower`, let's fill with the median.

```python
# Calculate median horsepower (ignoring NaNs)
median_hp = raw_mpg_df['horsepower'].median()
print(f"\nMedian horsepower: {median_hp}")

# Fill missing horsepower values
raw_mpg_df['horsepower'].fillna(median_hp, inplace=True)

# Verify missing values are handled
print("\nMissing values count after imputation:")
print(raw_mpg_df.isnull().sum())
```

**Step 5.3: Handling Categorical Features (Encoding)**
`origin` is categorical. Treating it as a number (1, 2, 3) implies an order and distance that doesn't exist. We should use One-Hot Encoding. `car_name` is too unique; we'll drop it.

```python
# Drop the car name column
raw_mpg_df = raw_mpg_df.drop('car_name', axis=1)

# Use One-Hot Encoding for 'origin'
# This creates new columns like 'origin_1', 'origin_2', 'origin_3'
raw_mpg_df = pd.get_dummies(raw_mpg_df, columns=['origin'], prefix='origin', drop_first=False) # drop_first=False keeps all origins explicit

print("\nData after One-Hot Encoding 'origin' (first 5 rows):")
print(raw_mpg_df.head())
print("\nData Info after Encoding:")
raw_mpg_df.info()
```
*   **Observation:** `origin` is gone, replaced by `origin_1`, `origin_2`, `origin_3`. All columns are now numeric.

**Step 5.4: Feature Scaling (Standardization)**
Features like `weight` (thousands) and `acceleration` (tens) have vastly different scales. Many models (including Linear Regression, though it's less sensitive) benefit from scaling features to have zero mean and unit variance (Standardization).

```python
# Separate target from features *before* scaling
y_processed = raw_mpg_df['mpg']
X_processed = raw_mpg_df.drop('mpg', axis=1)

# Identify numerical columns to scale (exclude the one-hot encoded origin columns for this example, though scaling them doesn't hurt)
numerical_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

# Create the scaler
scaler = StandardScaler()

# Fit and transform the numerical columns
X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])

print("\nFeatures after Standardization (first 5 rows):")
print(X_processed.head())

print("\nDescription of Scaled Features:")
print(X_processed[numerical_cols].describe()) # Mean should be close to 0, std dev close to 1
```

**Step 5.5: Data is Ready!**
*Now* the `X_processed` and `y_processed` dataframes are truly ready for the Split -> Train -> Predict -> Evaluate steps we performed in Part 3. The preprocessing steps (handling missing values, encoding categoricals, scaling) are essential for most real-world machine learning tasks.

**Key Takeaways from Part 5:**
*   Real-world data requires cleaning and preparation (preprocessing).
*   Common steps include handling missing data (imputation), converting categorical features to numbers (encoding), and scaling numerical features.
*   Tools like `pandas` for data manipulation and `scikit-learn`'s `StandardScaler` and `OneHotEncoder` (often used within `ColumnTransformer` and `Pipeline` for more complex workflows) are essential.
*   Preprocessing makes data suitable for modeling and often improves model performance.

---

**4. Regression Concepts (Parts 3 & 4)**

*   **Goal:** Predict a numerical value.
*   **Example Model:** Linear Regression (`sklearn.linear_model.LinearRegression`).
    *   Finds the best linear relationship between features and the target.
*   **Evaluation Metrics:**
    *   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values. Easy to interpret.
    *   **Mean Squared Error (MSE):** Average squared difference. Penalizes larger errors more.
    *   **Root Mean Squared Error (RMSE):** Square root of MSE. In the same units as the target.
    *   **R-squared (R²):** Proportion of target variance explained by the model (0 to 1, higher is better).
    *   `sklearn.metrics.mean_absolute_error`, `sklearn.metrics.mean_squared_error`, `sklearn.metrics.r2_score`.
*   **Visualization:**
    *   **Actual vs. Predicted Plot:** Scatter plot to visually check prediction quality (points near diagonal line are good).
    *   **Residual Plot:** Histogram or scatter plot of errors (Actual - Predicted). Ideally centered around zero with no clear pattern.
