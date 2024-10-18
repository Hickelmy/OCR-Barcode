import base64
import os
import cv2
import numpy as np
import pika
import json
from paddleocr import PaddleOCR
from io import BytesIO

ocr = PaddleOCR()

# RabbitMQ configuration
QUEUE_NAME = 'image_processing'
RESPONSE_QUEUE = 'image_response'
RABBITMQ_HOST = 'localhost'

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
channel = connection.channel()

channel.queue_declare(queue=QUEUE_NAME)
channel.queue_declare(queue=RESPONSE_QUEUE)

def salvar_imagem_base64(base64_string, caminho_pasta, nome_arquivo):
    # Decodifica o string Base64 para bytes de imagem
    imagem_bytes = base64.b64decode(base64_string)
    
    # Converte bytes de imagem para um array NumPy
    np_arr = np.frombuffer(imagem_bytes, np.uint8)
    imagem = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Cria a pasta caso não exista
    if not os.path.exists(caminho_pasta):
        os.makedirs(caminho_pasta)
    
    # Cria o caminho completo do arquivo
    caminho_arquivo = os.path.join(caminho_pasta, nome_arquivo)
    
    # Salva a imagem usando OpenCV
    cv2.imwrite(caminho_arquivo, imagem)

    print(f'Imagem salva em: {caminho_arquivo}')
    return imagem

def realizar_ocr(imagem, lang='en', confidence=0.5):
    # Realiza OCR usando PaddleOCR
    resultados = ocr.ocr(imagem, cls=True)
    if resultados is None:
        return ""
    
    resultado_texto = []
    boxes = []
    for linha in resultados:
        if linha is None:
            continue
        for res in linha:
            texto, conf = res[1]
            if conf >= confidence:
                resultado_texto.append(texto)
                boxes.append(res[0])
    
    return ' '.join(resultado_texto), boxes

def draw_ocr_bbox(image, boxes, colors):
    box_num = len(boxes)
    for i in range(box_num):
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, colors[i], 2)
    return image

def imagem_para_base64(imagem):
    _, buffer = cv2.imencode('.png', imagem)
    imagem_base64 = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/png;base64,{imagem_base64}'

def callback(ch, method, properties, body):
    data = json.loads(body)
    base64_string = data.get('image')
    if base64_string and base64_string.startswith('data:image/'):
        base64_string = base64_string.split(',', 1)[1]
    
    caminho_pasta = data.get('caminho_pasta', 'imagens')
    nome_arquivo = data.get('nome_arquivo', 'imagem.png')
    lang = data.get('lang', 'en')
    confidence = data.get('confidence', 0.5)

    if not base64_string:
        response = {'error': 'Base64 string não fornecida'}
        channel.basic_publish(exchange='', routing_key=RESPONSE_QUEUE, body=json.dumps(response))
        return

    imagem = salvar_imagem_base64(base64_string, caminho_pasta, nome_arquivo)
    texto_extraido, boxes = realizar_ocr(imagem, lang, confidence)
    
    # Desenhar as caixas delimitadoras na imagem
    colors = [(0, 255, 0)] * len(boxes)  # Usar a cor verde para todas as caixas
    imagem_com_caixas = draw_ocr_bbox(imagem, boxes, colors)
    
    # Converter a imagem com as caixas para Base64
    imagem_com_caixas_base64 = imagem_para_base64(imagem_com_caixas)
    
    response = {
        'texto_extraido': texto_extraido,
        'imagem_com_caixas_base64': imagem_com_caixas_base64
    }

    # Send the response back to the response queue
    channel.basic_publish(exchange='', routing_key=RESPONSE_QUEUE, body=json.dumps(response))
    print('[x] Processed and sent response back to response queue')

channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
