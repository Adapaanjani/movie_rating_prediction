from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model/random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_id = int(request.form['user_id'])
    movie_id = int(request.form['movie_id'])
    
    # Initialize genre list with 0s for 17 genres
    genres = [0] * 17
    
    # Set genres based on form input (you'll need to create inputs in your HTML for these)
    if request.form.get('Action'):
        genres[0] = 1
    if request.form.get('Adventure'):
        genres[1] = 1
    if request.form.get('Animation'):
        genres[2] = 1
    if request.form.get('Children'):
        genres[3] = 1
    if request.form.get('Comedy'):
        genres[4] = 1
    if request.form.get('Crime'):
        genres[5] = 1
    if request.form.get('Documentary'):
        genres[6] = 1
    if request.form.get('Drama'):
        genres[7] = 1
    if request.form.get('Fantasy'):
        genres[8] = 1
    if request.form.get('Film-Noir'):
        genres[9] = 1
    if request.form.get('Horror'):
        genres[10] = 1
    if request.form.get('Musical'):
        genres[11] = 1
    if request.form.get('Mystery'):
        genres[12] = 1
    if request.form.get('Romance'):
        genres[13] = 1
    if request.form.get('Sci-Fi'):
        genres[14] = 1
    if request.form.get('Thriller'):
        genres[15] = 1
    if request.form.get('War'):
        genres[16] = 1
    # Note: Make sure you have 17 genres in total

    # Prepare input for prediction
    input_data = np.array([[user_id, movie_id] + genres])
    
    # Print the shape of input_data for debugging
    print("Input Data Shape:", input_data.shape)  # Should be (1, 20)

    # Predict the rating
    predicted_rating = model.predict(input_data)[0]
    
    return render_template('index.html', prediction=predicted_rating)

if __name__ == '__main__':
    app.run(debug=True)
