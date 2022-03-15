from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import boto3
from urllib.request import urlopen
import os

image_width = 128
image_height = 128

BUCKET_NAME = 'dogbreedcl'
MODEL_FILE_NAME = 'dog_breed_cl_mobilenet_v2_js2.h5'

class Classifier():

    def __init__(self):
    
        self.model = load_model(MODEL_FILE_NAME)
        self.class_names = ['Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Tzu', 'Blenheim spaniel', 'Papillon', 'Toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'Tan coonhound', 'Walker hound', 'English foxhound', 'Redbone', 'Borzoi', 'Irish wolfhound', 'Italian greyhound', 'Whippet', 'Ibizan hound', 'Norwegian elkhound', 'Otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'Haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'Cairn', 'Australian terrier', 'Dandie dinmont', 'Boston bull', 'Miniature schnauzer', 'Giant schnauzer', 'Standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'Silky terrier', 'Coated wheaten terrier', 'West highland white terrier', 'Lhasa', 'Coated retriever', 'Coated retriever', 'Golden retriever', 'Labrador retriever', 'Chesapeake bay retriever', 'Haired pointer', 'Vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'Clumber', 'English springer', 'Welsh springer spaniel', 'Cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'Kuvasz', 'Schipperke', 'Groenendael', 'Malinois', 'Briard', 'Kelpie', 'Komondor', 'Old english sheepdog', 'Shetland sheepdog', 'Collie', 'Border collie', 'Bouvier des flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'Miniature pinscher', 'Greater swiss mountain dog', 'Bernese mountain dog', 'Appenzeller', 'Entlebucher', 'Boxer', 'Bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great dane', 'Saint bernard', 'Eskimo dog', 'Malamute', 'Siberian husky', 'Affenpinscher', 'Basenji', 'Pug', 'Leonberg', 'Newfoundland', 'Great pyrenees', 'Samoyed', 'Pomeranian', 'Chow', 'Keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'Toy poodle', 'Miniature poodle', 'Standard poodle', 'Mexican hairless', 'Dingo', 'Dhole', 'African hunting dog']
        #os.remove(MODEL_FILE_NAME)

    def classify(self, imageURL, local):
        if local == 1:
            x = load_img(imageURL, target_size=(image_width, image_height))
            x = img_to_array(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
        else:
            image_url = get_file('Dog', origin=imageURL )
            x = load_img(image_url, target_size=( image_width, image_height ) )
            x = img_to_array(x)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            os.remove(image_url)
        predictions = self.model.predict(x)
        pred_list = predictions[0].tolist()
        preds = []
        top_indices = [pred_list.index(x) for x in sorted(pred_list, reverse = True)[:3]]
        for i in top_indices:
            preds.append((self.class_names[i], pred_list[i]))
        return(preds, None)

