# Customer Churn Prediction using ANN

## Project Story

This project started with a simple goal:
**predict whether a customer will leave a telecom service or stay.**

At first it sounded easy… but once I opened the dataset, reality kicked in. Data was messy, columns were weird, and some values were not even in the correct datatype.

So this project slowly turned into a **complete machine learning pipeline** where I cleaned the data, explored patterns, prepared features, trained a neural network, and finally evaluated the model.

What started as curiosity ended up becoming a **full churn prediction model built with TensorFlow/Keras**.

---

# Dataset

The dataset contains **7043 telecom customers** and includes information such as:

* Customer tenure
* Monthly charges
* Contract type
* Payment method
* Internet services
* Streaming services
* Technical support
* Whether the customer churned or not

Target variable:

```
Churn
```

Values:

```
Yes → Customer left
No  → Customer stayed
```

---

# Step 1 — Data Cleaning

The first thing I noticed was that **TotalCharges** was stored as an object instead of numeric.

Example values looked like:

```
'1889.5'
'29.85'
'108.15'
```

So I converted it using:

```python
pd.to_numeric(df["TotalCharges"], errors="coerce")
```

Rows with invalid values became `NaN`, and those rows were removed.

Another issue was columns containing values like:

```
No internet service
```

For consistency I replaced them with:

```
No
```

because from a modelling perspective both mean the same thing.

---

# Step 2 — Exploratory Data Analysis

Before building any model I wanted to see **patterns in the data**.

One interesting observation was **tenure vs churn**.

Customers who stayed longer with the company had **much lower churn probability**.

I plotted the distribution using a histogram.

```python
plt.hist([tenure_yes, tenure_no], label=["Churn=Yes","Churn=No"])
```

This helped visually understand the relationship between tenure and churn.

---

# Step 3 — Feature Encoding

Machine learning models cannot understand text.

So categorical columns were converted using **One-Hot Encoding**.

Example:

```
Contract
```

became

```
Contract_Month-to-month
Contract_One year
Contract_Two year
```

I used:

```python
pd.get_dummies()
```

After encoding the dataset contained **26 numerical features**.

---

# Step 4 — Feature Scaling

Since neural networks work better with scaled values,
I used **MinMaxScaler**.

This converts all values into the range:

```
0 → 1
```

Code used:

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

---

# Step 5 — Train Test Split

The dataset was divided into training and testing data.

```
80% → Training
20% → Testing
```

Code:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5
)
```

---

# Step 6 — Building the Neural Network

I built a simple **Artificial Neural Network (ANN)** using TensorFlow/Keras.

Architecture:

```
Input Layer  → 26 features
Hidden Layer → 200 neurons (ReLU)
Hidden Layer → 150 neurons (ReLU)
Hidden Layer → 50 neurons (ReLU)
Output Layer → 1 neuron (Sigmoid)
```

Model:

```python
model = keras.Sequential([
    keras.layers.Dense(200, input_shape=(26,), activation="relu"),
    keras.layers.Dense(150, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
```

Compilation:

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Training:

```python
model.fit(X_train, y_train, epochs=5)
```

---

# Step 7 — Prediction

The model outputs **probabilities between 0 and 1**.

Example:

```
0.90 → High chance of churn
0.05 → Very low chance of churn
```

To convert probabilities into classes:

```python
y_pred = (model.predict(X_test) > 0.5).astype(int)
```

---

# Step 8 — Model Evaluation

To evaluate performance I used:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score

Confusion matrix helps visualize how many predictions were correct vs incorrect.

Example metrics obtained:

```
Accuracy
Precision
Recall
F1 Score
```

These metrics help understand the model beyond just accuracy.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

---

# What I Learned

While building this project I learned several important things:

* Real-world data is messy
* Data preprocessing is more important than the model itself
* Neural networks require properly scaled inputs
* Evaluation metrics matter more than raw accuracy

Most importantly, this project showed how **a complete machine learning pipeline works in practice**.

---

# Future Improvements

Possible improvements for this project:

* Handle class imbalance
* Add dropout layers
* Perform hyperparameter tuning
* Compare ANN with other models like Random Forest or XGBoost
* Deploy the model as a web application

---

# Final Thoughts

This project started as an experiment and slowly evolved into a full **end-to-end churn prediction system**.

It was a great exercise in understanding how data flows from **raw dataset → preprocessing → model → evaluation**.

And honestly… debugging the mistakes along the way taught more than the final model itself.
