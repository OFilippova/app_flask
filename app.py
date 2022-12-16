from flask import Flask, request, render_template
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVR
import tensorflow as tf
import pickle


app = Flask(__name__, template_folder='templates')


def get_prediction_nn(material_nn):
    model_nn = tf.keras.models.load_model(r"models/model_nn")
    y_pred_nn = round(float(model_nn.predict(material_nn)), 3)

    return f"Соотношение матрица-наполнитель {y_pred_nn}"


def get_prediction_reg(material_reg):
    model_reg = pickle.load(open('models/svr_regression.pkl', 'rb'))
    minmax_x = pickle.load(open('models/minmax_X.pkl', 'rb'))
    minmax_y = pickle.load(open('models/minmax_y.pkl', 'rb'))
    material_reg_norm = minmax_x.transform(material_reg.reshape(1, 11))
    y_pred_reg_norm = model_reg.predict(material_reg_norm)
    y_pred_reg_inv = minmax_y.inverse_transform(y_pred_reg_norm.reshape(1, 2))
    y_pred_1 = round(y_pred_reg_inv[0][0], 3)
    y_pred_2 = round(y_pred_reg_inv[0][1], 3)

    return f"Модуль упругости при растяжении: {y_pred_1} ГПа | Прочность при растяжении: {y_pred_2} МПа"


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/predict_nn.html', methods=['post', 'get'])
def processing_nn():
    message = ''
    if request.method == 'POST':
        material_nn = [float(request.form.get('feature_1')), float(request.form.get('feature_2')),
                       float(request.form.get('feature_3')), float(request.form.get('feature_4')),
                       float(request.form.get('feature_5')), float(request.form.get('feature_6')),
                       float(request.form.get('feature_7')), float(request.form.get('feature_8')),
                       float(request.form.get('feature_9')), float(request.form.get('feature_10')),
                       float(request.form.get('feature_11')), float(request.form.get('feature_12'))]

        material_nn = np.array([material_nn])

        message = get_prediction_nn(material_nn)

    return render_template('predict_nn.html', message=message)


@app.route('/predict_reg.html', methods=['post', 'get'])
def processing_reg():
    message = ''
    if request.method == 'POST':
        material_reg = [float(request.form.get('feature_1')), float(request.form.get('feature_2')),
                        float(request.form.get('feature_3')), float(request.form.get('feature_4')),
                        float(request.form.get('feature_5')), float(request.form.get('feature_6')),
                        float(request.form.get('feature_7')), float(request.form.get('feature_8')),
                        float(request.form.get('feature_9')), float(request.form.get('feature_10')),
                        float(request.form.get('feature_11'))]

        material_reg = np.array([material_reg])

        message = get_prediction_reg(material_reg)

    return render_template('predict_reg.html', message=message)


if __name__ == '__main__':
    app.run()
