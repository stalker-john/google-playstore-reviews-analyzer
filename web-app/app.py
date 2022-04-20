from flask import Flask,render_template,request
from model import *

app = Flask(__name__)
 
@app.route('/')
def form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def form_post():
    appid = request.form.get('appid')
    
    text = give_keytext(appid)
    mylist = []

    for tup in text:
        mylist.append(tup[0])
  
    return render_template('output.html', mylist=mylist)