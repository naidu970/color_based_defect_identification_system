from flask import Flask, render_template, request
import requests
from PIL import Image
import io
import base64
from gradio_client import Client
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors


import requests
app = Flask(__name__)

COLOR_NAMES={'aliceblue': [240.00600000000003, 247.9875, 255.0],
 'antiquewhite': [250.002, 235.00799999999998, 214.9905],
 'aqua': [0.0, 255.0, 255.0],
 'aquamarine': [126.99, 255.0, 212.007],
 'azure': [240.00600000000003, 255.0, 255.0],
 'beige': [245.004, 245.004, 219.9885],
 'bisque': [255.0, 227.9955, 195.993],
 'black': [0.0, 0.0, 0.0],
 'blanchedalmond': [255.0, 235.00799999999998, 204.99450000000002],
 'blue': [0.0, 0.0, 255.0],
 'blueviolet': [138.00599999999997, 42.993, 226.0065],
 'brown': [165.01049999999998, 41.99849999999999, 41.99849999999999],
 'burlywood': [222.003, 184.00799999999998, 134.99699999999999],
 'cadetblue': [94.9875, 157.99800000000002, 160.0125],
 'chartreuse': [126.99, 255.0, 0.0],
 'chocolate': [209.9925, 105.009, 29.987999999999996],
 'coral': [255.0, 126.99, 79.9935],
 'cornflowerblue': [100.01100000000001, 148.9965, 236.997],
 'cornsilk': [255.0, 247.9875, 219.9885],
 'crimson': [219.9885, 19.992, 60.00150000000001],
 'cyan': [0.0, 255.0, 255.0],
 'darkblue': [0.0, 0.0, 139.0005],
 'darkcyan': [0.0, 139.0005, 139.0005],
 'darkgoldenrod': [184.00799999999998, 134.0025, 10.990499999999999],
 'darkgray': [168.9885, 168.9885, 168.9885],
 'darkgreen': [0.0, 100.01100000000001, 0.0],
 'darkgrey': [168.9885, 168.9885, 168.9885],
 'darkkhaki': [189.00600000000003, 182.98800000000003, 106.998],
 'darkmagenta': [139.0005, 0.0, 139.0005],
 'darkolivegreen': [84.9915, 106.998, 46.9965],
 'darkorange': [255.0, 139.995, 0.0],
 'darkorchid': [153.0, 50.005500000000005, 204.0],
 'darkred': [139.0005, 0.0, 0.0],
 'darksalmon': [232.9935, 149.991, 121.992],
 'darkseagreen': [143.004, 188.0115, 143.004],
 'darkslateblue': [72.012, 60.996, 139.0005],
 'darkslategray': [46.9965, 78.99900000000001, 78.99900000000001],
 'darkslategrey': [46.9965, 78.99900000000001, 78.99900000000001],
 'darkturquoise': [0.0, 205.989, 208.998],
 'darkviolet': [148.00199999999998, 0.0, 211.0125],
 'deeppink': [255.0, 19.992, 147.0075],
 'deepskyblue': [0.0, 190.995, 255.0],
 'dimgray': [105.009, 105.009, 105.009],
 'dimgrey': [105.009, 105.009, 105.009],
 'dodgerblue': [29.987999999999996, 143.9985, 255.0],
 'firebrick': [177.99, 33.9915, 33.9915],
 'floralwhite': [255.0, 250.002, 240.00600000000003],
 'forestgreen': [33.9915, 139.0005, 33.9915],
 'fuchsia': [255.0, 0.0, 255.0],
 'gainsboro': [219.9885, 219.9885, 219.9885],
 'ghostwhite': [247.9875, 247.9875, 255.0],
 'gold': [255.0, 214.9905, 0.0],
 'goldenrod': [217.99949999999998, 165.01049999999998, 31.875],
 'gray': [127.5, 127.5, 127.5],
 'green': [0.0, 127.5, 0.0],
 'greenyellow': [172.99200000000002, 255.0, 46.9965],
 'grey': [127.5, 127.5, 127.5],
 'honeydew': [240.00600000000003, 255.0, 240.00600000000003],
 'hotpink': [255.0, 105.009, 180.0045],
 'indianred': [204.99450000000002, 92.00399999999999, 92.00399999999999],
 'indigo': [74.9955, 0.0, 129.999],
 'ivory': [255.0, 255.0, 240.00600000000003],
 'khaki': [240.00600000000003, 230.01, 139.995],
 'lavender': [230.01, 230.01, 250.002],
 'lavenderblush': [255.0, 240.00600000000003, 245.004],
 'lawngreen': [124.00650000000002, 251.99099999999999, 0.0],
 'lemonchiffon': [255.0, 250.002, 204.99450000000002],
 'lightblue': [172.99200000000002, 216.01049999999998, 230.01],
 'lightcoral': [240.00600000000003, 127.5, 127.5],
 'lightcyan': [223.99200000000002, 255.0, 255.0],
 'lightgoldenrodyellow': [250.002, 250.002, 209.9925],
 'lightgray': [211.0125, 211.0125, 211.0125],
 'lightgreen': [143.9985, 237.99149999999997, 143.9985],
 'lightgrey': [211.0125, 211.0125, 211.0125],
 'lightpink': [255.0, 181.9935, 193.0095],
 'lightsalmon': [255.0, 160.0125, 121.992],
 'lightseagreen': [31.875, 177.99, 170.00850000000003],
 'lightskyblue': [134.99699999999999, 205.989, 250.002],
 'lightslategray': [119.0085, 135.9915, 153.0],
 'lightslategrey': [119.0085, 135.9915, 153.0],
 'lightsteelblue': [176.00099999999998, 195.993, 222.003],
 'lightyellow': [255.0, 255.0, 223.99200000000002],
 'lime': [0.0, 255.0, 0.0],
 'limegreen': [50.005500000000005, 204.99450000000002, 50.005500000000005],
 'linen': [250.002, 240.00600000000003, 230.01],
 'magenta': [255.0, 0.0, 255.0],
 'maroon': [127.5, 0.0, 0.0],
 'mediumaquamarine': [102.0, 204.99450000000002, 170.00850000000003],
 'mediumblue': [0.0, 0.0, 204.99450000000002],
 'mediumorchid': [185.997, 84.9915, 211.0125],
 'mediumpurple': [147.0075, 111.99600000000001, 218.99399999999997],
 'mediumseagreen': [60.00150000000001, 179.01, 112.99050000000001],
 'mediumslateblue': [123.012, 103.98899999999999, 237.99149999999997],
 'mediumspringgreen': [0.0, 250.002, 153.99450000000002],
 'mediumturquoise': [72.012, 208.998, 204.0],
 'mediumvioletred': [199.002, 21.012000000000004, 133.00799999999998],
 'midnightblue': [24.99, 24.99, 111.99600000000001],
 'mintcream': [245.004, 255.0, 250.002],
 'mistyrose': [255.0, 227.9955, 225.01199999999997],
 'moccasin': [255.0, 227.9955, 180.99900000000002],
 'navajowhite': [255.0, 222.003, 172.99200000000002],
 'navy': [0.0, 0.0, 127.5],
 'oldlace': [253.011, 245.004, 230.01],
 'olive': [127.5, 127.5, 0.0],
 'olivedrab': [106.998, 142.0095, 35.0115],
 'orange': [255.0, 165.01049999999998, 0.0],
 'orangered': [255.0, 69.00299999999999, 0.0],
 'orchid': [217.99949999999998, 111.99600000000001, 213.996],
 'palegoldenrod': [237.99149999999997, 231.99900000000002, 170.00850000000003],
 'palegreen': [152.00549999999998, 250.99650000000003, 152.00549999999998],
 'paleturquoise': [175.0065, 237.99149999999997, 237.99149999999997],
 'palevioletred': [218.99399999999997, 111.99600000000001, 147.0075],
 'papayawhip': [255.0, 239.0115, 213.00150000000002],
 'peachpuff': [255.0, 217.99949999999998, 185.0025],
 'peru': [204.99450000000002, 133.00799999999998, 63.0105],
 'pink': [255.0, 191.98950000000002, 203.00549999999998],
 'plum': [221.00850000000003, 160.0125, 221.00850000000003],
 'powderblue': [176.00099999999998, 223.99200000000002, 230.01],
 'purple': [127.5, 0.0, 127.5],
 'rebeccapurple': [102.0, 51.0, 153.0],
 'red': [255.0, 0.0, 0.0],
 'rosybrown': [188.0115, 143.004, 143.004],
 'royalblue': [64.9995, 105.009, 225.01199999999997],
 'saddlebrown': [139.0005, 69.00299999999999, 18.9975],
 'salmon': [250.002, 127.5, 114.01050000000001],
 'sandybrown': [244.0095, 163.9905, 96.0075],
 'seagreen': [46.001999999999995, 139.0005, 87.00599999999999],
 'seashell': [255.0, 245.004, 237.99149999999997],
 'sienna': [160.0125, 82.008, 45.0075],
 'silver': [191.98950000000002, 191.98950000000002, 191.98950000000002],
 'skyblue': [134.99699999999999, 205.989, 235.00799999999998],
 'slateblue': [106.0035, 89.98949999999999, 204.99450000000002],
 'slategray': [111.99600000000001, 127.5, 143.9985],
 'slategrey': [111.99600000000001, 127.5, 143.9985],
 'snow': [255.0, 250.002, 250.002],
 'springgreen': [0.0, 255.0, 126.99],
 'steelblue': [69.9975, 129.999, 180.0045],
 'tan': [209.9925, 180.0045, 139.995],
 'teal': [0.0, 127.5, 127.5],
 'thistle': [216.01049999999998, 190.995, 216.01049999999998],
 'tomato': [255.0, 98.991, 70.992],
 'turquoise': [63.75, 223.99200000000002, 208.00349999999997],
 'violet': [237.99149999999997, 129.999, 237.99149999999997],
 'wheat': [245.004, 222.003, 179.01],
 'white': [255.0, 255.0, 255.0],
 'whitesmoke': [245.004, 245.004, 245.004],
 'yellow': [255.0, 255.0, 0.0],
 'yellowgreen': [153.99450000000002, 204.99450000000002, 50.005500000000005]}
def closest_color(requested_color):
    min_distance = float('inf')
    closest_name = ""
    for name, color in COLOR_NAMES.items():
        # Calculate the Euclidean distance between RGB values
        distance = np.sqrt(np.sum((np.array(requested_color) - np.array(color)) ** 2))
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def detect_all_colors(image_data, k=len(COLOR_NAMES.values())):
    # Load the image
    # image = cv2.imread(image_path)
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    output = ""
    # Check if the image was loaded successfully
    if image is None:
        output = f"Error: Unable to load image. Please check the file path."
        return output
    
    # Convert image from BGR to RGB (for better visualization)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)

    # Apply K-Means clustering to the pixels
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(pixels)

    # Get the cluster centers (the colors) and their respective labels (pixel assignments)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Calculate the percentage of each color
    label_counts = np.bincount(labels)
    total_pixels = len(labels)
    color_percentages = (label_counts / total_pixels) * 100

    # Get the number of actual clusters (less than k in some cases)
    unique_colors_count = len(np.unique(labels))

    # Print the colors, their percentages, and their closest color names
    colour_name_list ={}
    list_c=[]
    for i in range(unique_colors_count):
        rgb_color = colors[i]
        color_name = closest_color(rgb_color)
        if color_name not in colour_name_list.keys():
            colour_name_list[color_name]=color_percentages[i]
            list_c.append(colors[i])
        else:
            colour_name_list[color_name]+=color_percentages[i]
    total_percentage = sum(colour_name_list.values())
    for color_name, percentage in colour_name_list.items():
        output+=f"{percentage:.2f}%  Closest Color: {color_name}"+"\n"
    output+=f"Total: {total_percentage:.2f}%"
    return output
 



def upload_image(image_data):
    # Your ImgBB API key
    api_key = '2523d847e1c1653a9a342bda87acb27d'

    # Endpoint URL for image upload
    upload_url = "https://api.imgbb.com/1/upload"

    # Prepare the data for the POST request
    payload = {
        "key": api_key,
        "image": image_data.decode('utf-8')  
    }

    # Make the POST request to upload the image
    response = requests.post(upload_url, payload)

    # Parse the JSON response
    json_response = response.json()

    if 'data' in json_response:
        image_url = json_response['data']['url']
        return image_url
    else:
        return json_response.get('error', {}).get('message', 'Unknown error')


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Naidu"

@app.route("/process_edit", methods=['POST'])
def process_edit():
	uploaded_image_url = request.form['uploaded_image_url']
	text = extracted_text
	return render_template("index.html",  extracted_text =extracted_text ,uploaded_image_url=uploaded_image_url,prediction = text)


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	text = "Upload image in any the following format : Png/Jpg/Jpeg"
	extracted_text =" "
	if request.method == 'POST':
	
		try :
	
			extracted_text = request.files['my_image']
			extracted_text= extracted_text.stream.read()
		
			base64_image = base64.b64encode((extracted_text))
			uploaded_image_url = upload_image(base64_image)
            
			extracted_text = detect_all_colors(base64_image)
		except :
			text = "Invalid Format"
			extracted_text = "Upload image in any the following format : Png/Jpg/Jpeg or Enter Text Here and click on Submit"
			uploaded_image_url =" "
	
	return render_template("index.html", extracted_text =extracted_text ,uploaded_image_url=uploaded_image_url,prediction = text)



if __name__=="__main__":
    app.run(debug=True)
