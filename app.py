from flask import Flask
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
#from sklearn.externals import joblib
import os
import numpy as np
os.chdir(r"D:\spoj\tweet sentiment\New folder")
import numpy as np
import joblib
app = Flask(__name__)



loaded_model=joblib.load("model.pkl")
loaded_stop=joblib.load("stopwords.pkl")
loaded_vec=joblib.load("vectorizer.pkl")

def classify(document):
     X = loaded_vec.transform([document])
     y =loaded_model.predict((X)[0]) 
     proba = np.max(loaded_model.predict_proba(X)) 
     return y
 

def prob(document):
     X = loaded_vec.transform([document])
     y =loaded_model.predict((X)[0]) 
     proba = np.max(loaded_model.predict_proba(X)) 
     return proba


class ReviewForm(Form):
    text = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])
    
@app.route('/')
def index():
 form = ReviewForm(request.form)
 return render_template('reviewform.html', form=form)    


@app.route('/results', methods=['POST'])
def results():
    
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['text']
        y = classify(review)
        m=prob(review)
        
 
        return render_template('results.html',content=review,prediction=y,probability=int(m*100))
    return render_template('reviewform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)