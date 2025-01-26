# Melodious Medley: Song Recommendation Model

**Introduction**  
Melodious Medley is a predictive model designed to recommend songs based on user preferences and listening habits. This documentation outlines the methodology and algorithms used to build the model.

---

## Feature Extraction from the Dataset  

The dataset comprises 15 features, of which 14 were used for prediction (excluding `ID` as it doesn't significantly impact the outcome). The dataset is complete with no missing values, verified using:  
```python
dataset.isnull().sum().sum()
```  

A key feature in the dataset is the `ts_listen` field, which captures time-series data of song listening events. This field was transformed into a standardized datetime format to extract valuable temporal features such as:  

- **Day of the Week**:  
  Using the following code, a new column `Weekday` was created:  
  ```python
  dataset['Weekday'] = pd.to_datetime(dataset['ts_listen']).dt.weekday
  ```  
  Values range from `0` (Monday) to `6` (Sunday). This feature is particularly useful as song-listening behavior varies between weekdays and weekends.

- **Dummy Variables**:  
  To use `Weekday` in model training, it was converted into dummy variables, avoiding the dummy variable trap:  
  ```python
  dataset = pd.get_dummies(dataset, columns=['Weekday'], drop_first=True)
  ```

- **Normalization**:  
  Features were normalized using StandardScaler:  
  \[
  x' = \frac{x - \text{mean}}{\text{std. deviation}}
  \]
  The same scaling parameters were applied consistently across training, validation, and test datasets.

---

## Algorithms and Training  

Various algorithms were evaluated, including **Random Forest**, **XGBoost**, **LightGBM**, and **ANN**. The best results (accuracy ~0.87%) were achieved with **LightGBM** and **ANN** after parameter tuning.  

### **LightGBM**  
LightGBM is a gradient-boosting framework based on decision trees. Unlike traditional algorithms that split trees depth-wise, LightGBM employs a leaf-wise approach, optimizing for better accuracy and speed. Key highlights:  
- Model trained for 100 rounds.  
- Grid search for parameter tuning confirmed default parameters yielded optimal performance.  

### **Artificial Neural Network (ANN)**  
An ANN was implemented with the following architecture:  
- **Layers**: 5 layers with 30 neurons each.  
- **Activation**: ReLU for hidden layers, Sigmoid for the output layer.  
- **Loss Function**: Binary Cross-Entropy.  
- **Regularization**: Dropout (p=0.2) to prevent overfitting.  
- **Training**: Batch size = 16, Epochs = 10.  

---

## Model Evaluation  

The dataset was split into 80% training and 20% validation subsets. Evaluation metrics included:  
- **Accuracy Score**  
- **ROC AUC Score**  

### **Model Ensembling**  
To improve predictions, outputs from both models were ensembled by averaging probabilities:  
```python
ypred = (y_pred1 + y_pred2) / 2
```
Final predictions were determined with a threshold of 0.5.

---

## Remarks  

Both LightGBM and ANN performed similarly on the test set. By combining their strengths through ensembling, the final model achieved robust performance, making it well-suited for recommending songs tailored to user preferences.
