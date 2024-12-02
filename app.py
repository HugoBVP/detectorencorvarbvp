from flask import Flask, render_template
import os

app = Flask(__name__)

# Ruta para la página principal que sirve el index.html
@app.route('/')
def index():
    return render_template('index.html')

# Ejecutar la aplicación en el host 0.0.0.0 y puerto 5000
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
