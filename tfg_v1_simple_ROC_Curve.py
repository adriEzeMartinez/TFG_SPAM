import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# Generación de datos de ejemplo (simulando dataset de correos)
data = {
    'text': [
        'Congratulations! You have won a $1000 gift card. Click here to claim it.', 
        'Your Amazon package is on its way. Expected delivery tomorrow.',
        'Win big cash prizes. Click the link to enter now.',
        'Your electricity bill is due next week. Please pay promptly to avoid charges.',
        'Limited time offer: Buy one get one free!',
        'Meeting rescheduled to 3 PM today.',
        'Claim your reward now, only a few spots left!',
        'Don’t forget your doctor’s appointment tomorrow.',
        'Earn $500 from home easily. Click to find out more.',
        'Your monthly subscription will renew tomorrow.'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for SPAM, 0 for HAM
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Preprocesamiento: Vectorización del texto usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de Regresión Logística
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
y_prob_logistic = logistic_model.predict_proba(X_test)[:, 1]

# Modelo de Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear_cont = linear_model.predict(X_test)
y_pred_linear = [1 if y > 0.5 else 0 for y in y_pred_linear_cont]  # Clasificación binaria basada en umbral

# Evaluación de los Modelos
metrics = {
    'Model': ['Logistic Regression', 'Linear Regression'],
    'Accuracy': [accuracy_score(y_test, y_pred_logistic), accuracy_score(y_test, y_pred_linear)],
    'Recall': [recall_score(y_test, y_pred_logistic), recall_score(y_test, y_pred_linear)],
    'Precision': [precision_score(y_test, y_pred_logistic), precision_score(y_test, y_pred_linear)],
    'F1 Score': [f1_score(y_test, y_pred_logistic), f1_score(y_test, y_pred_linear)],
    'AUC': [roc_auc_score(y_test, y_prob_logistic), roc_auc_score(y_test, y_pred_linear_cont)]
}
metrics_df = pd.DataFrame(metrics)

# Mostrar la tabla de métricas al usuario
#import ace_tools as tools; tools.display_dataframe_to_user(name="Evaluation Metrics for Regression Models", dataframe=metrics_df)
print(metrics_df)

# Curva ROC para los dos modelos
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_logistic)
fpr_lin, tpr_lin, _ = roc_curve(y_test, y_pred_linear_cont)

# Plot de comparación de modelos
plt.figure(figsize=(10, 6))
plt.plot(fpr_log, tpr_log, label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_score(y_test, y_prob_logistic)))
plt.plot(fpr_lin, tpr_lin, label='Linear Regression (AUC = {:.2f})'.format(roc_auc_score(y_test, y_pred_linear_cont)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Logistic Regression vs Linear Regression')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
