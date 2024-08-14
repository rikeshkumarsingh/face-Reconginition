import os
import datetime
import pickle
from flask import Flask, request, jsonify
import cv2
from PIL import Image
import numpy as np
import face_recognition
import util
from test import test  

app = Flask(__name__)
db_dir = './db'
if not os.path.exists(db_dir):
    os.mkdir(db_dir)
log_path = './log.txt'

def save_log(name, status):
    with open(log_path, 'a') as f:
        f.write('{},{},{}\n'.format(name, datetime.datetime.now(), status))
        f.close()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server is running'}), 200

# @app.route('/register', methods=['POST'])
# def register_new_user():
#     if 'image' not in request.files or 'username' not in request.form:
#         return jsonify({'error': 'Image and username are required'}), 400

#     image_file = request.files['image']
#     username = request.form['username']
    
#     # Load the image
#     image = Image.open(image_file)
#     image_np = np.array(image)
    
#     embeddings = face_recognition.face_encodings(image_np)[0]
#     with open(os.path.join(db_dir, '{}.pickle'.format(username)), 'wb') as file:
#         pickle.dump(embeddings, file)
    
#     return jsonify({'message': 'User registered successfully'}), 200
@app.route('/register', methods=['POST'])
def register_new_user():
    if 'image' not in request.files or 'username' not in request.form or 'name' not in request.form or 'empid' not in request.form:
        return jsonify({'error': 'Image, username, name, and empid are required'}), 400

    image_file = request.files['image']
    username = request.form['username']
    name = request.form['name']
    empid = request.form['empid']
    
    # Load the image
    image = Image.open(image_file)
    image_np = np.array(image)
    
    embeddings = face_recognition.face_encodings(image_np)
    if not embeddings:
        return jsonify({'error': 'No face detected in the image'}), 400
    
    user_data = {
        'username': username,
        'name': name,
        'empid': empid,
        'embeddings': embeddings[0]
    }
    
    with open(os.path.join(db_dir, '{}.pickle'.format(username)), 'wb') as file:
        pickle.dump(user_data, file)
    
    return jsonify({'message': 'User registered successfully'}), 200

# @app.route('/login', methods=['POST'])
# def login():
#     if 'image' not in request.files:
#         return jsonify({'error': 'Image is required'}), 400

#     image_file = request.files['image']
    
#     # Load the image
#     image = Image.open(image_file)
#     image_np = np.array(image)

#     label = test(
#         image=image_np,
#         model_dir='C:/Rikesh/H/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',
#         device_id=0
#     )

#     if label == 1:
#         name = util.recognize(image_np, db_dir)
#         if name in ['unknown_person', 'no_persons_found']:
#             return jsonify({'error': 'Unknown user. Please register new user or try again.'}), 401
#         else:
#             save_log(name, 'in')
#             return jsonify({'message': 'Welcome, {}.'.format(name)}), 200
#     else:
#         return jsonify({'error': 'Spoof detected. You are fake!'}), 403

@app.route('/login', methods=['POST'])
def login():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    image_file = request.files['image']
    
    # Load the image
    image = Image.open(image_file)
    image_np = np.array(image)

    label = test(
        image=image_np,
        model_dir='C:/Rikesh/H/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',
        device_id=0
    )

    if label == 1:
        user_data = util.recognize(image_np, db_dir)
        if user_data is None:
            return jsonify({'error': 'Unknown user. Please register new user or try again.'}), 401
        else:
            name = user_data['name']
            empid = user_data['empid']
            username = user_data['username']
            save_log(username, 'in')
            return jsonify({
                'message': 'Welcome!',
                'username': username,
                'name': name,
                'empid': empid
            }), 200
    else:
        return jsonify({'error': 'Spoof detected. You are fake!'}), 403


@app.route('/logout', methods=['POST'])
def logout():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    image_file = request.files['image']
    
    # Load the image
    image = Image.open(image_file)
    image_np = np.array(image)

    label = test(
        image=image_np,
        model_dir='C:/Rikesh/H/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',
        device_id=0
    )

    if label == 1:
        user_data = util.recognize(image_np, db_dir)
        if user_data is None:
            return jsonify({'error': 'Unknown user. Please register new user or try again.'}), 401
        else:
            name = user_data['name']
            empid = user_data['empid']
            username = user_data['username']
            save_log(username, 'out')
            return jsonify({
                'message': 'Goodbye!',
                'username': username,
                'name': name,
                'empid': empid
            }), 200
    else:
        return jsonify({'error': 'Spoof detected. You are fake!'}), 403


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
