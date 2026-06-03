import pickle
from flask import Flask, request, render_template

ridge_model = pickle.load(open(r'models\ridge.pkl', 'rb'))
standardscaler = pickle.load(open(r'models\scaler.pkl', 'rb'))

application = Flask(__name__)
app = application


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'POST':

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data = standardscaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        result = ridge_model.predict(new_data)

        return render_template(
            'home.html',
            results=round(result[0], 2)
        )

    return render_template('home.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)