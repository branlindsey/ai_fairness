
import pandas as pd
import psycopg2 as pg2
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


@app.route("/")
def show_table():
    conn = pg2.connect(dbname='transactions', user='postgres', password ='password', host='localhost')
    cur = conn.cursor()
    query = '''SELECT time_stamp, fraud, f_prob, payout_type, event_id, org_name, email_domain, country, total_sales
    FROM fraud_predictions
    ORDER BY time_stamp DESC
    LIMIT 15;
    '''
    cur.execute(query)
    df = pd.DataFrame(columns=['time_stamp', 'fraud', 'f_prob', 'payout_type', 'event_id', 'org_name', 'domain', 'country', 'total_sales'])
    for i,row in enumerate(cur.fetchall()):
        df.loc[i] = row
    table = df.to_html()
    return render_template('fraud.html', tables=table)


if __name__ == '__main__':
   
    
    app.run(host='0.0.0.0', port=8000, debug=True)
