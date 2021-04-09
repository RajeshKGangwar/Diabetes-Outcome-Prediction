from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pickle


filename = "diabetes-prediction.pkl"
model = pickle.load(open(filename,'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():

    sample = list()
    scale = StandardScaler()

    if request.method == "POST":

        pregnancies = request.form["pregnancies"]
        sample.append(pregnancies)
        glucose = request.form["glucose"]
        sample.append(glucose)
        blood_pressure = request.form["bloodpressure"]
        sample.append(blood_pressure)
        skin_thickness = request.form["skinthickness"]
        sample.append(skin_thickness)
        insulin = request.form["insulin"]
        sample.append(insulin)
        bmi = request.form["bmi"]
        sample.append(bmi)
        diabetespedigreefunction = request.form["diabetespedigreefunction"]
        sample.append(diabetespedigreefunction)
        age = request.form["age"]
        sample.append(age)



    result = int(model.predict([sample]))

    if result == 1:
        value = "You suffer from Diabetes"
    else:
        value = "You are Diabetic Free"
    return render_template('result.html',value=value)



if __name__ == '__main__':
    app.run(debug=True)