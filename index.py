from paddleocr import PaddleOCR 
import json
from PIL import Image
import gradio as gr
import numpy as np
import cv2

def get_random_color():
    c = tuple(np.random.randint(0, 256, 3).tolist())
    return c

def draw_ocr_bbox(image, boxes, colors):
    box_num = len(boxes)
    for i in range(box_num):
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, colors[i], 2)
    return image

def inference(img: Image.Image, lang, confidence):
    # Carregando o OCR sem download dinâmico
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang=lang,
        use_gpu=False,
        det_model_dir='./models/en_PP-OCRv3_det_infer',  # Caminho local para o modelo de detecção
        rec_model_dir='./models/en_PP-OCRv3_rec_infer',  # Caminho local para o modelo de reconhecimento
        cls_model_dir='./models/en_PP-OCRv3_cls_infer'   # Caminho local para o modelo de classificação de ângulo
    )

    img2np = np.array(img)
    result = ocr.ocr(img2np, cls=True)[0]

    # Converter imagem para RGB
    image = img.convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    final_result = [
        dict(boxes=box, txt=txt, score=score, _c=get_random_color())
        for box, txt, score in zip(boxes, txts, scores)
    ]
    final_result = [item for item in final_result if item['score'] > confidence]

    im_show = draw_ocr_bbox(image, [item['boxes'] for item in final_result], [item['_c'] for item in final_result])
    im_show = Image.fromarray(im_show)
    data = [[json.dumps(item['boxes']), round(item['score'], 3), item['txt']] for item in final_result]
    return im_show, data

def webcam_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar espelhamento da imagem
        frame = cv2.flip(frame, 0)

        # Exibir o feed da webcam
        cv2.imshow('Webcam Feed', frame)
        
        # Encerrar ao pressionar a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    demo = gr.Interface(
        inference,
        [
            gr.Image(type='pil', label='Input'),
            gr.Dropdown(choices=['ch', 'en', 'fr', 'german', 'korean', 'japan'], value='ch', label='language'),
            gr.Slider(0.1, 1, 0.5, step=0.1, label='confidence_threshold')
        ],
        [
            gr.Image(type='pil', label='Output'),
            gr.Dataframe(headers=['bbox', 'score', 'text'], label='Result')
        ],
        title='Projeto Label',
        description='Teste de OCR',
        examples=[
            ['example_imgs/img1.webp', 'en', 0.5],
            ['example_imgs/img2.webp', 'en', 0.7],
            ['example_imgs/img3.jpg', 'en', 0.7],
        ],
        css=".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
    )
    demo.queue(max_size=10)
    demo.launch(debug=True, server_name="127.0.0.1")

    # Iniciar feed da webcam
    webcam_feed()
