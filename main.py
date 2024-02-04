import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import cv2 as cv
import numpy as np
import urllib.request
from keras.models import load_model
from PIL import Image, ImageOps
import requests
import time

kivy.require('2.0.0')

class YaView(BoxLayout):
    def __init__(self):
        super(YaView, self).__init__()

    def debut(self):    
        # Remplacez l'URL par l'adresse IP correcte de votre ESP32-CAM
        url = 'http://192.168.150.209/cam-lo.jpg'
winName = 'CAM'

cv.namedWindow(winName)

# Charger le modèle Keras
model = load_model("keras_Model.h5", compile=False)

# Charger les étiquettes
class_names = open("labels.txt", "r").readlines()

# ... (code précédent)

while True:
    # Capturer l'image depuis la caméra ESP32-CAM
    imgResponse = urllib.request.urlopen('http://192.168.150.209/cam-lo.jpg')
    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
    img = cv.imdecode(imgNp, -1)

    # Prétraiter l'image pour l'entrée du modèle Keras
    image = Image.fromarray(img).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prédire avec le modèle
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Afficher les résultats
    print("Class:", class_name[2:], "Confidence Score:", confidence_score)

    # Envoyer les données à l'adresse IP
    payload = {'confidence_score': confidence_score}
    endpoint = 'http://192.168.150.209/send-data'  # Remplacez par le chemin de votre endpoint
    response = requests.post(endpoint, data=payload)

    # Déclencher la capture si la classe est "Manque Potassium" et la confiance est élevée
    if class_name[2:] == "Manque Potassium" and confidence_score > 0.80:
        # Capturer et enregistrer l'image localement si nécessaire
        cv.imwrite("carence_potassium.jpg", img)

    # Afficher l'image capturée
    cv.imshow(winName, img)

    # Gérer la fermeture de la fenêtre
    tecla = cv.waitKey(5) & 0xFF
    if tecla == 27:
        break

    # Pause de 30 secondes avant la prochaine itération
    time.sleep(10)


    
class YaApp(App):
    def build(self):
        return YaView()
    
yaApp = YaApp()
yaApp.run()