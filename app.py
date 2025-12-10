from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import re
from nltk.corpus import stopwords
import nltk

# -------------------- Initialize Flask App --------------------
app = Flask(__name__)
app.secret_key = "secret123"

# -------------------- Download Stopwords --------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------- Load Models and Vectorizer --------------------
nb_model = joblib.load('models/models/naive_bayes_model.pkl')
lr_model = joblib.load('models/models/logistic_regression_model.pkl')
rf_model = joblib.load('models/models/random_forest_model.pkl')
vectorizer = joblib.load('models/models/tfidf_vectorizer.pkl')

# -------------------- Text Cleaning Function --------------------
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# -------------------- FORM ROUTE --------------------
@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        message = request.form.get('problem', '').strip()
        if not message:
            flash('Please enter a message to analyze.', 'warning')
            return render_template('form.html')

        try:
            cleaned_message = clean_text(message)
            data = vectorizer.transform([cleaned_message])

            # Predictions
            prediction_nb = nb_model.predict(data)[0]
            prediction_lr = lr_model.predict(data)[0]
            prediction_rf = rf_model.predict(data)[0]

            def to_numeric(pred):
                if isinstance(pred, str):
                    return 1 if pred.lower() == 'spam' else 0
                return int(pred)

            predictions = [to_numeric(prediction_nb), to_numeric(prediction_lr), to_numeric(prediction_rf)]

            # Individual results
            result_nb = 'Spam' if predictions[0] == 1 else 'Not Spam'
            result_lr = 'Spam' if predictions[1] == 1 else 'Not Spam'
            result_rf = 'Spam' if predictions[2] == 1 else 'Not Spam'

            # Majority Vote
            spam_count = sum(predictions)
            majority_result = 'Spam' if spam_count > len(predictions)/2 else 'Not Spam'

            result = {
                'Naive Bayes': result_nb,
                'Logistic Regression': result_lr,
                'Random Forest': result_rf,
                'Spam Count': spam_count,
                'Total Models': len(predictions),
                'Overall Result (Majority Vote)': majority_result
            }

            # Redirect to result page
            return render_template('result.html', result=result, message=message)

        except Exception as e:
            flash(f'Error during prediction: {str(e)}', 'danger')
            return render_template('form.html')

    return render_template('form.html')

# -------------------- Run Flask App --------------------
if __name__ == '__main__':
    app.run(debug=True, port=5065)
