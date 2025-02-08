from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Initial example sales data (last 7 days)
days = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
sales = np.array([500, 550, 600, 650, 700, 850, 900])

# Train initial model
model = LinearRegression()
model.fit(days, sales)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get new sales data from form
        new_sales = request.json.get('sales')
        if len(new_sales) != 7:
            return jsonify({'error': 'Please enter sales for exactly 7 days.'})

        # Convert to NumPy arrays
        new_sales = np.array(new_sales)
        new_days = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)

        # Retrain model with new data
        model.fit(new_days, new_sales)

        # Predict next two days
        future_days = np.array([[8], [9]])
        predicted_sales = model.predict(future_days)

        return jsonify({
            'predicted_day_8': round(predicted_sales[0], 2),
            'predicted_day_9': round(predicted_sales[1], 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
