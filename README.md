# Obesity Prediction Using Machine Learning

## 🎯 Project Title
**Obesity Level Prediction using Random Forest Classifier**

---

## 📖 Objective

To develop a machine learning model that can accurately classify individuals into various obesity levels based on their demographic characteristics and lifestyle habits. The model aims to support healthcare practitioners in early identification of obesity trends and facilitate proactive health management.

---

## 🧠 Problem Statement

Obesity is a growing public health concern globally. Identifying individuals at risk before serious complications arise can help improve health outcomes. Manual assessments are often time-consuming and subjective. This project automates obesity level prediction using data-driven approaches to assist in scalable, consistent, and quick evaluations.

---

## 📊 Dataset Features

The model uses a dataset (`Obesity prediction.csv`) that typically includes the following types of features:

- Age
- Gender
- Height, Weight
- Eating habits (frequency of fast food, vegetable consumption, etc.)
- Physical activity (daily activity level, time spent exercising, etc.)
- Lifestyle habits (smoking, alcohol consumption)
- Family history and other health indicators

**Target Variable:**
- `Obesity` (Categorical values such as Underweight, Normal Weight, Overweight, Obese, etc.)

---

## 🔍 Approach

1. **Data Preprocessing**
   - Handle categorical variables using Label Encoding.
   - Standardize numerical data using `StandardScaler`.

2. **Splitting the Data**
   - Train-test split (80/20) for evaluation.

3. **Model Training**
   - Trained using `RandomForestClassifier` with `class_weight='balanced'` to handle imbalanced classes.

4. **Prediction**
   - Input data is preprocessed and passed to the trained model to get an encoded class prediction.
   - Prediction is decoded back to human-readable obesity labels.

5. **Evaluation**
   - Accuracy
   - Precision (Weighted)
   - Recall (Weighted)

---

## 🧪 Model Output

The model returns the **predicted obesity level** and supports performance evaluation using common classification metrics.

---

## 💡 Key Highlights

- Fully automated preprocessing and encoding
- Easy integration and extensibility
- Well-suited for real-world health monitoring systems
- Modular code with reusable components

---

## ✅ Applications

- **Healthcare & Clinics**: Use as a screening tool during routine health checkups.
- **Wellness Programs**: Targeted recommendations based on predicted obesity category.
- **Research**: Analyze obesity patterns across populations.
