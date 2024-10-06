import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Directorios de los correos SPAM y HAM
spam_dir = 'enron/spam/'
ham_dir = 'enron/ham/'

# Lectura de los correos y etiquetas
emails = []
labels = []

# Leer archivos SPAM
for filename in os.listdir(spam_dir):
    with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1') as file:
        emails.append(file.read())
        labels.append(1)  # Etiqueta para SPAM

# Leer archivos HAM
for filename in os.listdir(ham_dir):
    with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1') as file:
        emails.append(file.read())
        labels.append(0)  # Etiqueta para HAM

# Crear DataFrame con los datos
df = pd.DataFrame({'text': emails, 'label': labels})

# Preprocesamiento: Vectorización del texto usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
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
