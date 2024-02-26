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
import matplotlib.pyplot as plt
import mpld3
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
        print("hello jamila")
        #dd=df['comments']
        #data_clean=preprocess_text(df)
        data=df['comments']
        print("file readed")

        if 'svm_self' in request.form:

            rev= vecto.transform(data)
            # Load the SVM model from the .pkl file
            predictions = model.predict(rev)
            predictions_str = list(map(str, predictions))
            df['sentiment']=predictions_str
            grouped_reviews = df.groupby(['makes', 'sentiment']).size().reset_index(name='count')
            print(grouped_reviews)
            count_2_pos = predictions_str.count('2')
            count_1_neu = predictions_str.count('1')
            count_0_nag = predictions_str.count('0')
            print(" count ",count_2_pos)
            print(" count ",count_1_neu)
            print(" count ",count_0_nag)
            #list_sentiment=[count_2_pos,count_1_neu,count_0_nag]
            #print("********************************",len(predictions_str))
            pivot_table = grouped_reviews.pivot(index='makes', columns='sentiment', values='count')

            # Plotting the bar chart
            ax = pivot_table.plot(kind='bar', figsize=(8, 5))

            # Adding labels and title
            ax.set_xlabel('Make')
            ax.set_ylabel('Count')
            ax.set_title('Reviews by Make and Sentiment')

            # Convert the chart to an interactive HTML format using mpld3
            chart_html = mpld3.fig_to_html(ax.figure)
            # Return the results or render a template with the results
            return render_template('index.html', chart_html=chart_html ,positive=count_2_pos,neutre=count_1_neu,negative=count_0_nag)
        
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






rev= vecto.transform(data)
                # Load the SVM model from the .pkl file
                predictions = model.predict(rev)
                predictions_str = list(map(str, predictions))
                df['sentiment']=predictions_str
                grouped_reviews = df.groupby(['makes', 'sentiment']).size().reset_index(name='count')
                print(grouped_reviews)
                count_2_pos = predictions_str.count('2')
                count_1_neu = predictions_str.count('1')
                count_0_nag = predictions_str.count('0')
                print(" count ",count_2_pos)
                print(" count ",count_1_neu)
                print(" count ",count_0_nag)
                #list_sentiment=[count_2_pos,count_1_neu,count_0_nag]
                #print("********************************",len(predictions_str))
                pivot_table = grouped_reviews.pivot(index='makes', columns='sentiment', values='count')

                # Plotting the bar chart
                ax = pivot_table.plot(kind='bar', figsize=(8, 5))

                # Adding labels and title
                ax.set_xlabel('Make')
                ax.set_ylabel('Count')
                ax.set_title('Reviews by Make and Sentiment')

                # Convert the chart to an interactive HTML format using mpld3
                chart_html = mpld3.fig_to_html(ax.figure)
                # Return the results or render a template with the results
                return render_template('index.html', chart_html=chart_html ,positive=count_2_pos,neutre=count_1_neu,negative=count_0_nag)
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
