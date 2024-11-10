from flask import Flask, render_template, request, redirect, url_for
import os
from gradio_client import Client, handle_file
import shutil

app = Flask(__name__)
client = Client("yisol/IDM-VTON")

class VirtualTryOn:
    def try_on(self, background_image_path, garment_image_path):
        if not os.path.exists(background_image_path):
            raise FileNotFoundError(f"Background image not found at {background_image_path}.")
        if not os.path.exists(garment_image_path):
            raise FileNotFoundError(f"Garment image not found at {garment_image_path}.")

        result = client.predict(
            dict={
                "background": handle_file(background_image_path),
                "layers": [],
                "composite": None
            },
            garm_img=handle_file(garment_image_path),
            garment_des="Virtual try-on example",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        if not result or len(result) < 2:
            raise ValueError("Unexpected result from Gradio API.")

        output_background = result[0]
        output_garment = result[1]

        output_background_path = os.path.join('static', 'output_background.jpg')
        output_garment_path = os.path.join('static', 'output_garment.jpg')

        shutil.copy(output_background, output_background_path)
        shutil.copy(output_garment, output_garment_path)

        return f"/static/output_background.jpg", f"/static/output_garment.jpg"

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "admin":
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials, please try again.")

    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    background_image_url = None
    garment_image_url = None

    if request.method == 'POST':
        background_image = request.files.get('background_image')
        garment_image = request.files.get('garment_image')

        if background_image and garment_image:
            background_image_path = 'background.jpg'
            garment_image_path = 'garment.jpg'
            background_image.save(background_image_path)
            garment_image.save(garment_image_path)

            try:
                virtual_try_on = VirtualTryOn()
                background_image_url, garment_image_url = virtual_try_on.try_on(
                    background_image_path, garment_image_path
                )
            except Exception as e:
                return f"An error occurred: {str(e)}"

    return render_template('index.html', 
                           background_image_url=background_image_url, 
                           garment_image_url=garment_image_url)

# Favicon handler
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':  # Corrected line
    app.run(debug=True, port=10000)
