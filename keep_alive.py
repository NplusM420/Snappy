from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "Hello. I'm a good bot! @NplusM made me do it!"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()
