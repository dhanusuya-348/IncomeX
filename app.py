import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, render_template, request

# Initialize Flask App
app = Flask(__name__)

# Load and Preprocess the Dataset
def load_and_preprocess_data():
    # Loading the Dataset
    file_path = r'D:\SEM - 6\DM project\IncomeX\adult_income_census.csv'
    df = pd.read_csv(file_path)
    
    # Handling the Missing Values and Inconsistencies
    df.replace('?', np.nan, inplace=True)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Removing the Duplicates
    df = df.drop_duplicates()

    # Outlier Detection and Handling
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    outliers = (z_scores > 3).any(axis=1)
    df = df[~outliers]

    # Data Smoothing
    bins = np.linspace(df['age'].min(), df['age'].max(), 10)
    df['age_bin_mean'] = pd.cut(df['age'], bins, labels=False)
    df['age_smooth_mean'] = df.groupby('age_bin_mean')['age'].transform('mean')
    df['age_smooth_median'] = df.groupby('age_bin_mean')['age'].transform('median')

    def boundary_smooth(x, bins):
        bin_edges = np.linspace(min(x), max(x), bins)
        return [min(bin_edges, key=lambda b: abs(b - i)) for i in x]
    df['age_smooth_boundary'] = boundary_smooth(df['age'], 10)

    # Data Normalization
    scaler = MinMaxScaler()
    df['normalized_age'] = scaler.fit_transform(df[['age']])
    scaler = StandardScaler()
    df['zscore_age'] = scaler.fit_transform(df[['age']])
    max_abs = df['age'].abs().max()
    df['decimal_scaled_age'] = df['age'] / (10**len(str(int(max_abs))))

    # Feature Engineering and Encoding
    encoder = LabelEncoder()
    df['income'] = encoder.fit_transform(df['income'])

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Filling out any NaN values after encoding
    df.fillna(0, inplace=True)

    # Train-Test Split
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Model Evaluation
    accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    return model, X, encoder, accuracy, cm, cr


# Feature Importance
def get_feature_importance(model, X_columns):
    importance = pd.DataFrame({'Feature': X_columns, 'Importance': model.coef_[0]})
    return importance.sort_values('Importance', ascending=False)


# Convert Matplotlib figure to base64 for HTML rendering
def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')


# Career Suggestions based on Features' Importance
def career_suggestions(importance):
    low_income_factors = importance[importance['Importance'] < 0].sort_values('Importance')
    suggestions = [
        "Suggested Careers for Higher Income: Data Science, Software Engineering, Finance, Healthcare Management"
    ]
    return low_income_factors, suggestions


# Prediction Function
def predict_income(user_input, model, X_columns):
    # Transforming the user input to match training data
    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df)
    user_df = user_df.reindex(columns=X_columns, fill_value=0)
    
    user_prediction = model.predict(user_df)[0]
    return "Income > 50K" if user_prediction == 1 else "Income <= 50K"


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_input = {}
    input_fields = [
        'age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'sex', 'native.country'
    ]

    # Getting data from form
    for field in input_fields:
        user_input[field] = request.form[field]

    # Preprocessing model and prediction
    model, X, encoder, accuracy, cm, cr = load_and_preprocess_data()
    prediction = predict_income(user_input, model, X.columns)

    # Feature importance for career suggestions
    importance = get_feature_importance(model, X.columns)
    low_income_factors, suggestions = career_suggestions(importance)

    # Generate confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    cm_url = fig_to_base64(fig)

    # Generate feature importance plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    importance.plot(kind='bar', ax=ax2, legend=False)
    ax2.set_title('Feature Importance')
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('Importance')

    fi_url = fig_to_base64(fig2)

    return render_template('index.html',
                           prediction=prediction,
                           low_income_factors=low_income_factors,
                           suggestions=suggestions,
                           accuracy=accuracy,
                           cm_url=cm_url,
                           cr=cr,
                           fi_url=fi_url,
                           importance=importance) # Pass the full 'importance' DataFrame


if __name__ == '__main__':
    app.run(debug=True)
