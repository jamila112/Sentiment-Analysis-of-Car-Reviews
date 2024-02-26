from flask import Flask, jsonify, render_template, request
import joblib
import pickle 

from spacy.lang.fr.stop_words import STOP_WORDS as fr_stopwords
import pandas as pd
import nltk
nltk.download('punkt')

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import string
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stopwords


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import string
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stopwords



nlp = spacy.load('fr_core_news_md')


def preprocess_text(df):
    # Convert comments to lowercase and remove punctuation
    df['rev_clean'] = df['comments'].str.lower().str.replace('[^\w\s]', '')
    
    # Tokenize the text
    df['rev_clean'] = df['rev_clean'].apply(word_tokenize)
    
    # Remove stopwords
    df['rev_clean'] = df['rev_clean'].apply(lambda x: [word for word in x if word.lower() not in fr_stopwords])
    
    # Join the tokens back into a string
    df['rev_clean'] = df['rev_clean'].apply(lambda x: ' '.join(x))
    
    # Lemmatize the text using spacy
    lemmatized = []
    for s in df['rev_clean']:
        doc = nlp(s)
        lemmatized_sentence = ' '.join([token.lemma_ for token in doc])
        lemmatized.append(lemmatized_sentence)
    
    df['rev_clean'] = lemmatized
    
    return df






app = Flask(__name__)


with open('SVM_dep.pkl', 'rb') as file:
    model = pickle.load(file)
with open('SVM_vec.pkl', 'rb') as file:
    vecto = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    excel_file = request.files['fileInput']
    if excel_file:
        # Read the Excel file using pandas
        df = pd.read_excel(excel_file)
        #dd=df['comments']
        #data_clean=preprocess_text(df)
        data=df['comments']
        print("file readed")

        if 'svm_self' in request.form:

            rev= vecto.transform(data)
            # Load the SVM model from the .pkl file
            predictions = model.predict(rev)
            predictions_str = list(map(str, predictions))
            count_2_pos = predictions_str.count('2')
            count_1_neu = predictions_str.count('1')
            count_0_nag = predictions_str.count('0')
            list_sentiment=[count_2_pos,count_1_neu,count_0_nag]
            
            print("********************************",len(predictions_str))
            # Return the results or render a template with the results
            return render_template('index.html', predictions=list_sentiment)
        elif 'xgboost_slef' in request.form:
            return 'hello xgboost slef'
        elif 'svm_co' in request.form:
            return 'svm co'
        elif 'xgboost_co' in request.form:
            return 'xgboost_co'
        else:
            return "No method selected"
    else:
        return 'file not selected'
        


        
import random
if __name__ == '__main__':
    #app.run()

    port = 5000 + random.randint(0, 999)
    print(port)
    url = "http://127.0.0.1:{0}".format(port)
    print(url)
    app.run(use_reloader=False, debug=True, port=port)
    






"""@app.route('/predict', methods=['POST','GET'])
def predict():
        text = request.form.get('review')
        #text=[text]
        ###########################################################################
        
        #######################################################################
        rev= vecto.transform([text])
        prediction = model.predict(rev)[0]
        prediction = str(prediction)
        print("***************************************************************** ",prediction)
        #response = {'prediction': prediction}
        #return jsonify(response)
        return render_template('index.html', prediction=text)
if __name__ == '__main__':
    app.run()
"""

"""from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
@app.route('/index')

def home():
    if request.method=='POST':
        # Load the SVM model
        with open('nb.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('vectorizer.pkl', 'rb') as file:
            vecto = pickle.load(file)
        user_input=request.form.get('size')
        rev= vecto.transform([user_input])
        print(rev)
        #vect_message = vectorizer.transform([message])
        #result = spam_ham_model.predict(vect_message)[0]
        
        prediction = model.predict(rev)[0]
        print("**************************************************************** ",prediction)
        #prediction = model.predict([user_input])

    return  prediction


if  __name__ == '__main__':
    app.run(debug=True)
"""



"""from flask import Flask, request
import joblib
import pickle

app = Flask(__name__)
with open('nb.pkl', 'rb') as file:
    model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vecto = pickle.load(file)


@app.route('/')
def hello_world():
    return("Hello World")


@app.route('/prediction', methods=['GET','POST'])
def prediction():
    user_input=request.form.get('size')
    rev= vecto.transform([user_input])
    result = model.predict(rev)[0]
    print("******************************************** ",result)
    

if __name__ == '__main__':
    app.run()
"""




















<!DOCTYPE html>
<html>
<head>
    <title> Upload your file for prediction </title>

    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">


</head>
<body>
    
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="header">
        <h1>Upload your file for prediction</h1>
        
        <input type="file"  id="fileInput" name="fileInput" >

        </div>

        <div class="methods">
            <div class="self">
            <label>SELF-TRAINING</label>
            <br>
            <input type="submit" name="svm_self" value="SVM">
            <br>
            <input type="submit" name="xgboost_slef" value="XGBoost">
            </div>

            
            <div class="co">
            <label>CO-TRAINING</label>
            <br>
            <input type="submit" name="svm_co" value="SVM">
            <br>
            <input type="submit" name="xgboost_co" value="XGBoost">
            </div>

        </div>
        
    </form>


    <tbody>
            {% for prediction in predictions %}

            <div>{{ prediction }}</div>
            
            {% endfor %}
            <br>

            
    </tbody>

</body>
</html>









<div class="emoji-container">
           <p>Positive</p>
           <p class="emoji">&#128516;</p>
        </div>
        
        <div class="emoji-container">
            <h3>Neutral</h3>
        <p class="emoji">&#128528;</p>
        </div>
        
        <div class="emoji-container">
            <h3>Negative</h3>
        <p class="emoji">&#128542;</p>
        </div>













.header input[type="file"] {
  display: inline-block;
  vertical-align: middle;
}



.self{
  margin-left: 300px;
  background-color: #F0FFFF;
  display: inline-block;

}



label {
  display: block;
  margin-bottom: 10px;
  color: #333;
  font-weight: bold;
}

input[type="file"] {
  display: block;
  margin-bottom: 10px;

}

.methods {
  margin-top: 60px;

}

.self,
.co {

  padding: 10px 20px;
  float: center;

}



.self label,
.co label {
  display: inline-block;
  margin-right: 10px;
  color: #333;
}

input[type="submit"] {
  padding: 5px 10px;
  background-color: #333;
  color: #fff;
  border: none;
  margin-top: 10px;
  margin-left: 20px;
}

input[type="submit"]:hover {
  background-color: #555;
}

/* Results styles */
tbody {
  margin: 20px auto;
  max-width: 400px;
  padding: 20px;
  background-color: #fff;
  border-radius: 5px;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
}

tbody div {
  margin-bottom: 10px;
  color: #333;
}






