from flask import Flask, render_template, url_for, request
import sqlite3
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import shutil
from markdown import markdown
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np  # dealing with arrays

from tensorflow.keras.preprocessing.image import load_img,img_to_array

from explainable_ai import get_response

APP_URL = 'http://127.0.0.1:5004'


# Load the trained model
model = load_model('ResNet50_model.h5')

# Load class names from the pickle file
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def predict_image(image):
    img =load_img(image, target_size=(150, 150))
    img_array =img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    print(predicted_class_index)
    predicted_class = class_names[predicted_class_index]
    print("predicted_class:",predicted_class)
    prediction1 = prediction.tolist()
    print(prediction1[0][predicted_class_index]*100)
    return predicted_class, prediction1[0][predicted_class_index]*100

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result)==0:
            return render_template('index.html',msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('home.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')



@app.route('/userlog.html', methods=['GET'])
def indexBt():
      return render_template('userlog.html')

@app.route('/graph.html', methods=['GET'])
def graph(): 
    images = [f'{APP_URL}/static/accuracy_plot.png',
              f'{APP_URL}/static/loss_plot.png',
              f'{APP_URL}/static/conf_mat.png']
    content=['Accuracy Graph',
             'Loss Graph(Error Message)',
            'Confusion Matrix']
    return render_template('graph.html',images=images,content=content)

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        try:
            # Clear previous images
            dirPath = os.path.join("static", "images")
            if os.path.exists(dirPath):
                for fileName in os.listdir(dirPath):
                    file_path = os.path.join(dirPath, fileName)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            else:
                os.makedirs(dirPath)
            
            # Handle both direct file upload and filename from test directory
            if 'file' in request.files and request.files['file'].filename != '':
                # Direct file upload
                file = request.files['file']
                # Secure the filename
                filename = secure_filename(file.filename)
                filepath = os.path.join(dirPath, filename)
                file.save(filepath)
            # elif 'filename' in request.form:
            #     # Using existing file from test directory
            #     filename = secure_filename(request.form['filename'])
            #     source = os.path.join("test", filename)
            #     filepath = os.path.join(dirPath, filename)
                
            #     if os.path.exists(source):
            #         shutil.copy(source, filepath)
            #     else:
            #         return render_template('userlog.html', error="File not found in test directory")
            else:
                return render_template('userlog.html', error="No file uploaded")

            # Read the image for processing
            image = cv2.imread(filepath)
            if image is None:
                return render_template('userlog.html', error="Invalid image file")
            
            # Process the image
            processed_images = process_image(image)
            
            # Get prediction
            predicted_class, accuracy = predict_image(filepath)
            
            # Map prediction to label using dictionary
            condition_map = {
                "Epidural_Hemorrhage": "Epidural_Hemorrhage",
                "Fracture_Yes_No": "Fracture_Yes_No",
                "hemorrhagic_stroke": "hemorrhagic_stroke",
                "Intraparenchymal_Hemorrhage": "Intraparenchymal_Hemorrhage",
                "Intraventricular_Hemorrhage": "Intraventricular_Hemorrhage",
                "ischemic_stroke": "ischemic_stroke",
                "No_Hemorrhage": "No_Hemorrhage",
                "normal": "normal",
                "Sinusitis_Negative": "Sinusitis_Negative",
                "Sinusitis_Positive": "Sinusitis_Positive",
                "Subarachnoid_Hemorrhage": "Subarachnoid_Hemorrhage",
                "Subdural_Hemorrhage": "Subdural_Hemorrhage"
            }
            
            str_label = condition_map.get(predicted_class, "Unknown Condition")
            explainable_ai_response = get_response(filepath,prompt=f"This scan is predicted to have an {str_label}.")
            
            return render_template('results.html', 
                                  status=str_label,
                                  status2=f'{accuracy}',
                                  explainable_ai_response=markdown(explainable_ai_response),
                                  ImageDisplay=f"{APP_URL}/static/images/{filename}",
                                  ImageDisplay1=f"{APP_URL}/static/{processed_images['gray']}",
                                  ImageDisplay2=f"{APP_URL}/static/{processed_images['edges']}",
                                  ImageDisplay3=f"{APP_URL}/static/{processed_images['threshold']}",
                                  ImageDisplay4=f"{APP_URL}/static/{processed_images['sharpened']}")
        
        except Exception as e:
            return render_template('userlog.html', error=f"Error processing image: {str(e)}")
            
    return render_template('userlog.html')

# Helper function to process images
def process_image(image):
    output_paths = {}
    
    # Color conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_paths['gray'] = 'gray.jpg'
    cv2.imwrite(os.path.join('static', output_paths['gray']), gray_image)
    
    # Apply the Canny edge detection
    edges = cv2.Canny(image, 250, 254)
    output_paths['edges'] = 'edges.jpg'
    cv2.imwrite(os.path.join('static', output_paths['edges']), edges)
    
    # Apply thresholding to segment the image
    retval2, threshold2 = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    output_paths['threshold'] = 'threshold.jpg'
    cv2.imwrite(os.path.join('static', output_paths['threshold']), threshold2)
    
    # Create the sharpening kernel
    kernel_sharpening = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]])
    
    # Apply the sharpening kernel to the image
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    output_paths['sharpened'] = 'sharpened.jpg'
    cv2.imwrite(os.path.join('static', output_paths['sharpened']), sharpened)
    
    return output_paths

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5004)
