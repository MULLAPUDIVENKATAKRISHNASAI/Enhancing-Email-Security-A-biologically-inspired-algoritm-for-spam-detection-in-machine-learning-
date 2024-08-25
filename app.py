from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)
pickle_in = open('model.pickle','rb')
pac = pickle.load(pickle_in)
tfid = open('tranform.pickle','rb')
tfidf_vectorizer = pickle.load(tfid)

train = pd.read_csv('Email spam.csv')
train=train.dropna()
train['spam'].unique()
train[train['spam']=='its termination would not  have such a phenomenal impact on the power situation .  however '].shape
df_x=train['text']
df_y=train['spam']

x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.3, random_state=9)

tfidf_vectorizer= TfidfVectorizer(min_df=1,stop_words='english')
tfidf_train = tfidf_vectorizer.fit_transform(x_train) 

clf=MultinomialNB()
clf.fit(tfidf_train,y_train)
acc = clf.score(tfidf_train,y_train)
tfidf_test = tfidf_vectorizer.transform(x_test) 
y_pred = clf.predict(tfidf_test)

f1  = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
precsion = precision_score(y_test, y_pred, average='weighted')

# Harris Hawks Optimization Algorithm (not used in this version)
def hho_algorithm(objective_function, num_variables, num_hawks, max_iter, lb, ub):
    pass
    # Initialization
    positions = np.random.uniform(lb, ub, (num_hawks, num_variables))
    convergence_curve = []

    for iter in range(max_iter):
        # Calculate fitness values
        fitness_values = np.apply_along_axis(objective_function, 1, positions)

        # Sort positions based on fitness values
        sorted_indices = np.argsort(fitness_values)
        sorted_positions = positions[sorted_indices]

        # Update the top positions (based on the exploration and exploitation phase)
        for i in range(num_hawks):
            for j in range(num_variables):
                r1 = np.random.random() # Random number for evasion
                r2 = np.random.random() # Random number for attack

                # Evasion phase
                if r1 < 0.5:
                    positions[i, j] = sorted_positions[0, j] + np.random.uniform(-1, 1) * (sorted_positions[0, j] - sorted_positions[i, j])

                # Attack phase
                else:
                    positions[i, j] = sorted_positions[0, j] - r2 * (sorted_positions[0, j] - sorted_positions[i, j])

                # Boundary handling
                positions[i, j] = np.clip(positions[i, j], lb[j], ub[j])

        # Update convergence curve
        convergence_curve.append(np.min(fitness_values))
    return sorted_positions[0], convergence_curve

@app.route('/')
@app.route('/index') 
def index():
    return render_template('index.html')

@app.route('/login') 
def login():
    return render_template('login.html') 
   
@app.route('/home') 
def home():
    return render_template('home.html') 

@app.route('/abstract') 
def abstract():
    return render_template('abstract.html') 
 
@app.route('/future') 
def future():
    return render_template('future.html')    

@app.route('/user') 
def user():
    return render_template('user.html')     

@app.route('/upload') 
def upload():
    return render_template('upload.html') 

@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/chart')
def chart():    
    abc = request.args.get('news')    
    input_data = [abc.rstrip()]
    # Transforming input
    tfidf_test = tfidf_vectorizer.transform(input_data)
    # Predicting the input
    y_pred = pac.predict(tfidf_test)
    # For demonstration purposes, assuming y_test is your entire test dataset's labels
    y_test_pred = pac.predict(tfidf_vectorizer.transform(x_test))
     # Dummy call to HHO algorithm
    dummy_num_variables = 10
    dummy_num_hawks = 20
    dummy_max_iter = 100
    dummy_lb = np.zeros(dummy_num_variables)
    dummy_ub = np.ones(dummy_num_variables)
    accpred = accuracy_score(y_test, y_test_pred)
    if y_pred[0] == 1:         
        label = "Spam"
    elif y_pred[0] == 0:
        label = "No Spam"
    return render_template('prediction.html', prediction_text=label, val0=accpred, val1=acc, val2=f1, val3=recall, val4=precsion)

@app.route('/performance') 
def performance():
    return render_template('performance.html')    

def txtpred(text):  
    textn = [text.rstrip()]  
    # Transforming input
    tfidf_test = tfidf_vectorizer.transform(textn)
    # Predicting the input
    y_pred = pac.predict(tfidf_test)
    # For demonstration purposes, assuming y_test is your entire test dataset's labels
    y_test_pred = pac.predict(tfidf_vectorizer.transform(x_test))
     # Dummy call to HHO algorithm
    dummy_num_variables = 10
    dummy_num_hawks = 20
    dummy_max_iter = 100
    dummy_lb = np.zeros(dummy_num_variables)
    dummy_ub = np.ones(dummy_num_variables)
    accpred = accuracy_score(y_test, y_test_pred)
    if y_pred[0] == 1:         
        label = "Spam"
    elif y_pred[0] == 0:
        label = "No Spam"
    return label

@app.route('/read_file', methods=["POST"])
def read_file():
    if request.method == 'POST':
        file = request.files['datasetfile']  # Corrected the file name to match the HTML form
        text = file.read().decode("utf-8")
        print(text)
        # Perform processing on the text as needed
        label = txtpred(text)  # Assuming txtpred is a function defined elsewhere
        return render_template('upload.html', prediction_text=label)
    
if __name__ == '__main__':
    app.run(debug=True)
