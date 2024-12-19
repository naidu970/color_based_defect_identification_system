
from flask import Flask, render_template, request
import io
import base64
from gradio_client import Client
import requests


app = Flask(__name__)


def upload_image(image_data):

    # Your ImgBB API key

    api_key = '2523d847e1c1653a9a342bda87acb27d'

    # Endpoint URL for image upload

    upload_url = 'https://api.imgbb.com/1/upload'

    # Prepare the data for the POST request

    payload = {'key': api_key, 'image': image_data.decode('utf-8')}

    # Make the POST request to upload the image

    response = requests.post(upload_url, payload)

    # Parse the JSON response

    json_response = response.json()

    if 'data' in json_response:
        image_url = json_response['data']['url']
        return image_url
    else:
        return json_response.get('error', {}).get('message',
                'Unknown error')


# routes

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')


@app.route('/about')
def about_page():
    return 'Naidu'


@app.route('/process_edit', methods=['POST'])
def process_edit():
    uploaded_image_url = request.form['uploaded_image_url']
    extracted_text = request.form['edited_text']
    text = extracted_text.replace("\n", "<br>")
    return render_template('index.html', extracted_text=extracted_text,
                           uploaded_image_url=uploaded_image_url,
                           prediction=text)


@app.route('/submit', methods=['GET', 'POST'])
def get_output():
    text = 'Upload image in any the following format : Png/Jpg/Jpeg'
    extracted_text = ' '
    if request.method == 'POST':

        try:

            extracted_text = request.files['my_image']
            extracted_text = extracted_text.stream.read()

            base64_image = base64.b64encode(extracted_text)
            uploaded_image_url = upload_image(base64_image)
            client = Client('manaidu20011/a')
            extracted_text = client.predict(image_path=None,
                    image_url=uploaded_image_url, api_name='/predict')
            text = extracted_text
            client = Client("manaidu20011/cloud")
            extracted_text = "predict"+extracted_text.replace("\n",":")[:-1]
            result = client.predict(command=extracted_text,api_name="/predict")
            extracted_text = result
            text= extracted_text
        except Exception as e:
            text = e
            extracted_text = \
                'Upload image in any the following format : Png/Jpg/Jpeg or Enter Text Here and click on Submit'
            uploaded_image_url = ' '

    return render_template('index.html', extracted_text=extracted_text,
                           uploaded_image_url=uploaded_image_url,
                           prediction=text)

if __name__ == '__main__':
    app.run(debug=True)
