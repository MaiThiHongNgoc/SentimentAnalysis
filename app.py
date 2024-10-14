from flask import Flask, request, jsonify, render_template
import pickle
from nltk.tokenize import word_tokenize
import nltk

# Tải về nltk data nếu chưa có
nltk.download('punkt')

# Tải mô hình và vectorizer đã lưu trữ
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

# Tiền xử lý văn bản (cùng hàm như trong file ai_model.py)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalpha()]
    return ' '.join(filtered_tokens)

# Trang chủ (HTML)
@app.route('/')
def home():
    return render_template('index.html')

# API phân tích cảm xúc
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu văn bản từ request
    data = request.form['text']
    processed_data = preprocess_text(data)

    # Vector hóa dữ liệu đầu vào
    vectorized_input = vectorizer.transform([processed_data]).toarray()

    # Dự đoán cảm xúc
    prediction = model.predict(vectorized_input)

    # Chuyển đổi kết quả dự đoán thành cảm xúc
    result = 'Positive' if prediction[0] == 1 else 'Negative'

    return jsonify({'sentiment': result})

if __name__ == '__main__':
    app.run(debug=True)
