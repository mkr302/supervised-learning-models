# Supervised Learning

Supervised learning is a type of machine learning where a model learns from labeled data. In this approach, the algorithm is trained on a dataset that consists of input-output pairs, where each input (feature) is associated with a corresponding correct output (label). The goal is for the model to learn the mapping function from inputs to outputs so it can make accurate predictions on new, unseen data.

### **Key Characteristics of Supervised Learning:**
1. **Labeled Data** – The training dataset contains labeled examples (i.e., known input-output pairs).
2. **Learning Function** – The model learns to map inputs to outputs based on patterns in the data.
3. **Performance Evaluation** – The model is evaluated using metrics like accuracy, precision, recall, and mean squared error.
4. **Prediction** – Once trained, the model can predict outputs for new inputs.

### **Types of Supervised Learning:**
1. **Classification** – The model predicts discrete categories or classes.
   - **Examples**:
     - Spam detection (Email --> Spam or Not Spam)
     - Image recognition (Cat or Dog)
     - Sentiment analysis (Positive, Negative, Neutral)

2. **Regression** – The model predicts continuous values.
   - **Examples**:
     - Predicting house prices based on features like size, location, and number of rooms.
     - Forecasting stock prices.
     - Estimating customer lifetime value.

### **Common Algorithms for Supervised Learning:**
- **Linear Regression** – Used for predicting continuous values.
- **Logistic Regression** – Used for binary classification.
- **Decision Trees** – Used for both classification and regression.
- **Random Forest** – An ensemble method for better accuracy.
- **Support Vector Machines (SVM)** – Finds the best decision boundary.
- **Neural Networks** – Used for complex problems like image recognition and NLP.
- **k-Nearest Neighbors (k-NN)** – A simple method based on similarity.

### **Workflow of Supervised Learning:**
1. **Data Collection** – Gather labeled training data.
2. **Data Preprocessing** – Clean and transform data (handle missing values, normalize, encode categorical variables).
3. **Model Selection** – Choose an appropriate algorithm.
4. **Training** – Train the model on the dataset.
5. **Evaluation** – Assess performance using a test dataset.
6. **Hyperparameter Tuning** – Optimize parameters for better results.
7. **Prediction** – Use the trained model to make predictions on new data.

