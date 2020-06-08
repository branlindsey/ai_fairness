
import pandas as pd
import psycopg2 as pg2
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)




@app.route("/")
def show_table():
    
    return render_template('fraud.html', tables=table)


if __name__ == '__main__':
   
    
    app.run(host='0.0.0.0', port=8000, debug=True)
