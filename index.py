from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image, UnidentifiedImageError
import base64
import numpy as np
from io import BytesIO
import os
from pymongo import MongoClient
import datetime
import gridfs

app = Flask(__name__)

# Conectar ao MongoDB
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['ocr_database']
fs = gridfs.GridFS(db)

# Inicializando o PaddleOCR com modelos locais
ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='en', 
    use_gpu=False, 
    det_model_dir='models/ch_PP-OCRv4_det_infer',
    rec_model_dir='models/ch_PP-OCRv4_rec_infer'
)

def save_image_to_folder(image, folder_path, filename):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    image_path = os.path.join(folder_path, filename)
    image.save(image_path)
    return image_path

def process_image(img: Image.Image, confidence):
    img2np = np.array(img)
    result = ocr.ocr(img2np, cls=True)[0]
    
    # Convertendo a imagem para RGB
    image = img.convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    
    final_result = [dict(boxes=box, txt=txt, score=score) for box, txt, score in zip(boxes, txts, scores)]
    final_result = [item for item in final_result if item['score'] > confidence]

    return final_result

@app.route('/process_image', methods=['POST'])
def process_base64_image():
    try:
        data = request.get_json()
        image_b64 = data['image']
        confidence = data.get('confidence', 0.5)
        
        # Ajuste no padding para evitar erros de decodificação
        missing_padding = len(image_b64) % 4
        if missing_padding:
            image_b64 += '=' * (4 - missing_padding)
        
        # Removendo o prefixo "data:image/png;base64,"
        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]
        
        # Decodificando a imagem base64
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
        except (UnidentifiedImageError, base64.binascii.Error):
            return jsonify({'error': 'Cannot identify image file. Please check the input format.'})
        
        # Salvando a imagem original na pasta "camera_img"
        original_image_path = save_image_to_folder(image, 'camera_img', 'original_image.png')
        
        # Processando a imagem com OCR
        ocr_data = process_image(image, confidence)
        
        # Salvando a imagem processada na pasta "ocr_img"
        processed_image_path = save_image_to_folder(image, 'ocr_img', 'ocr_image.png')
        
        # Salvando a imagem processada no MongoDB com GridFS
        with open(processed_image_path, 'rb') as f:
            processed_image_id = fs.put(f, filename="processed_image.png")
        
        # Salvando os dados do OCR no MongoDB
        ocr_entry = {
            'processed_image_id': processed_image_id,
            'ocr_data': ocr_data,
            'timestamp': datetime.datetime.now()
        }
        db.ocr_entries.insert_one(ocr_entry)
        
        return jsonify({'message': 'Image and OCR data saved successfully', 'image_id': str(processed_image_id)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
