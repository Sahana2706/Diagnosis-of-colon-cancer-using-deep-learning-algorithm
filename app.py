import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
import sqlite3
import shutil




app = Flask(__name__)

@app.route('/')
def index():
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

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


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

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

@app.route('/developer.html')
def developer():
    return render_template('developer.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
              'http://127.0.0.1:5000/static/loss_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png',
              'http://127.0.0.1:5000/static/f1-score.jpeg']
    content=['Accuracy Graph',
             "Loss Graph",
             'Confusion Matrix',
             'f1-Score']

            
    
        
    return render_template('graph.html',images=images,content=content)
    


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)# # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                    [-1,-1,-1]])

        # # apply the sharpening kernel to the image
        sharpened =cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)



       
        
        
        
        model=load_model('colonoscopy.h5')
        path='static/images/'+fileName


        # Load the class names
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        Tre=""
        Tre1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        if predicted_class =="dyed-lifted-polyps":
            str_label = "dyed-lifted-polyps(stage1)"
            Tre="Medical Treatmemnt"
            Tre1=["surgical polypectomy",
                  "colectomy"]
                         
           
        elif predicted_class == 'dyed-resection-margins':
            str_label = "dyed-resection-margins(stage2)"
            Tre="Medical Treatmemnt"
            Tre1=["re-excision",
                  "wider excision"]
            


        elif predicted_class == 'esophagitis':
            str_label = "esophagitis cancer"
            Tre="Medical Treatmemnt"
            Tre1=["immunotherapy",
                  "tarageted therapy"]
            

        elif predicted_class == 'normal-cecum':
            str_label = "normal-cecum(stage1)"
            Tre="Medical Treatmemnt"
            Tre1=["pyloromyotomy",
                  "gastrojejunostomy"]
            

        elif predicted_class == 'normal-pylorus':
            str_label = "normal-pylorus(stage2)"
            Tre="Medical Treatmemnt"
            Tre1=["Adjuvant therapy",
                  "NeoAdjuvant therapy"]

        elif predicted_class == 'normal-z-line':
            str_label = "normal-z-line(stage3)"
            Tre="Medical Treatmemnt"
            Tre1=["Gastrectomy",
                  "ivor lewis esophagectomy"]

        elif predicted_class == 'polyps':
            str_label = "polyps cancer"
            Tre="Medical Treatmemnt"
            Tre1=["polypectomy",
                  "endoscopic mucosal resection",
                  "endoscopic submucosal dissection"]

        elif predicted_class == 'ulcerative-colitis':
            str_label = "ulcerative-colitis cancer"
            Tre="Medical Treatmemnt"
            Tre1=["Aminosalicylates",
                  "immunomodulators"]

        A=predicted_class =="dyed-lifted-polyps"
        B=predicted_class == 'dyed-resection-margins'
        C=predicted_class == 'esophagitis'
        D= predicted_class == 'normal-cecum'
        E= predicted_class == 'normal-pylorus'

        F=predicted_class == 'normal-z-line'
        G=predicted_class == 'polyps'
        H=predicted_class == 'ulcerative-colitis'


        dic={'0':A,'1':B,'2':C,'3':D,'4':E,'5':F,'6':G,'7':H}
        algm = list(dic.keys()) 
        accu = list(dic.values()) 
        fig = plt.figure(figsize = (5, 5))  
        plt.bar(algm, accu, color ='maroon', width = 0.3)  
        plt.xlabel("Comparision") 
        plt.ylabel("Accuracy Level") 
        plt.title("Accuracy Comparision between \n Colonoscopy cancer Detection")
        plt.savefig('static/matrix.png')    
            

            

       
            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
       
            

       


        return render_template('results.html', status=str_label,accuracy=accuracy,Treatment=Tre,Treatment1=Tre1, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg",ImageDisplay5="http://127.0.0.1:5000/static/matrix.png")
        
    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
