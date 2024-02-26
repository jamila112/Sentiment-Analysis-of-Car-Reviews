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
import matplotlib.pyplot as plt
import base64
import io
nlp = spacy.load('fr_core_news_md')
app = Flask(__name__)
with open('model_best.pkl', 'rb') as file:
    model = pickle.load(file)
with open('vectorizer_best.pkl', 'rb') as file:
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
        user_make = request.form.get('make')
        user_modele = request.form.get('modele')
        rev= vecto.transform(data)
        # Load the SVM model from the .pkl file
        predictions = model.predict(rev) 
        if 'svm_self' in request.form:
            if user_make and user_modele:  
                predictions_str = list(map(str, predictions))
                df['sentiment']=predictions_str
                grouped_reviewss = df.groupby(['makes', 'modeles', 'sentiment']).size().reset_index(name='count')
                grouped_reviewss['count'] = grouped_reviewss['count'].astype('int')
                grouped_reviewss['sentiment'] = grouped_reviewss['sentiment'].astype('int')
                print(grouped_reviewss)
                #grouped_reviewss['modeles'] = grouped_reviewss['modeles'].str.strip()
                filtered_data = grouped_reviewss[(grouped_reviewss['makes'] == user_make) & (grouped_reviewss['modeles'] == user_modele)]
                if not filtered_data.empty:
                    print("-----------------------------------------------------------",filtered_data)
                    labels = filtered_data['sentiment']
                    sizes = filtered_data['count'].tolist()
                    sentiments=filtered_data['sentiment'].tolist()
                    print("*******************SIZES******************************",sizes)
                    print("*******************sentiments******************************",sentiments)
                    C_pos=0
                    C_neu=0
                    C_neg=0
                    for sentiment, count in zip(sentiments, sizes):

                        if sentiment == 2:
                            C_pos=count
                        elif sentiment == 1:
                            C_neu=count
                        elif sentiment == 0:
                            C_neg=count
                    all_data = df.groupby(['makes', 'sentiment']).size().reset_index(name='count')
                    count_2_pos=0
                    count_1_neu=0
                    count_0_nag=0
                    count_2_pos = predictions_str.count('2')
                    count_1_neu = predictions_str.count('1')
                    count_0_nag = predictions_str.count('0')

                    pivot_table = all_data.pivot(index='makes', columns='sentiment', values='count')

                    # Plotting the bar chart
                    ax = pivot_table.plot(kind='bar', figsize=(11, 5))

                    # Adding labels and title
                    ax.set_xlabel('Make')
                    ax.set_ylabel('Count')
                    ax.set_title('Reviews by Make and Sentiment')

                    # Convert the chart to an interactive HTML format using mpld3
                    chart_htmll= mpld3.fig_to_html(ax.figure)
                ############################################################################################################
                    colors = ['#FF5733', '#33FF99', '#33A2FF']
                    plt.figure(figsize=(4, 4))

                    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    plt.axis('equal')
                    plt.title('Pie chart for reviews')

                    # Save the chart to a BytesIO object
                    chart = io.BytesIO()
                    plt.savefig(chart, format='png')
                    chart.seek(0)
                    chart_data = base64.b64encode(chart.getvalue()).decode('utf-8')

                    ################################################################################################
                    pivot_table = filtered_data.pivot(index='makes', columns='sentiment', values='count')

                    # Plotting the bar chart
                    ax = pivot_table.plot(kind='bar', figsize=(6, 3))

                    # Adding labels and title
                    ax.set_xlabel('Make')
                    ax.set_ylabel('Count')
                    ax.set_title('Bar chart for reviews')

                    # Convert the chart to an interactive HTML format using mpld3
                    chart_html = mpld3.fig_to_html(ax.figure)
                    # Return the results or render a template with the results
                    return render_template('index.html',chart_htmll=chart_htmll,chart_html=chart_html,chart_data=chart_data,make=user_make,model=user_modele,positive=C_pos,negative=C_neg,neutre=C_neu,posi=count_2_pos,neut=count_1_neu,negg=count_0_nag)
                else:
                    return'No data available for the specified make and model.'
            else:
                return 'Input make or model is not empty'
        else:
            return 'No method selected'
    else:
        return 'file not selected'
        


  
import random
if __name__ == '__main__':
    #app.run()

    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    app.run(use_reloader=False, debug=True, port=port)
    






