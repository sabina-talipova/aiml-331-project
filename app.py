from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

# @app.route('/predict', methods=['GET'])
# def predict():
#     # img = request.files['file']
#     # img = image.load_img(img, target_size=(224, 224))
#     # img_array = image.img_to_array(img)
#     # img_array = np.expand_dims(img_array, axis=0) / 255.0
#     #
#     # predictions = model.predict(img_array)
#     # predicted_class = np.argmax(predictions)
#     #
#     # return jsonify({'predicted_class': str(predicted_class)})
#
#     return "Test"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)