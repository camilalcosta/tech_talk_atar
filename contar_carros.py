import numpy as np
import cv2
import sys

VIDEO = 'Ponte.mp4'

algorithm_types = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']


# -------------------------------------------------------------------------------------------------------------------------

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Criação do kernel de dilatação
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((3, 3), np.uint8)  # Criação do kernel de abertura
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((3, 3), np.uint8)  # Criação do kernel de fechamento
    return kernel


def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'),
                                iterations=2)  # Aplicação do filtro de fechamento na imagem
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'),
                                iterations=2)  # Aplicação do filtro de abertura na imagem
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)  # Aplicação da dilatação na imagem
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'),
                                   iterations=2)  # Aplicação do filtro de fechamento na imagem
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'),
                                   iterations=2)  # Aplicação do filtro de abertura na imagem fechada
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)  # Aplicação da dilatação na imagem aberta
        return dilation


def subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()  # Criação de um objeto subtrator de fundo do tipo GMG
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()  # Criação de um objeto subtrator de fundo do tipo MOG
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()  # Criação de um objeto subtrator de fundo do tipo MOG2
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()  # Criação de um objeto subtrator de fundo do tipo KNN
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()  # Criação de um objeto subtrator de fundo do tipo CNT
    print('Detector inválido')
    sys.exit(1)


# -------------------------------------------------------------------------------------------------------------------------

w_min = 30  # largura minima do retangulo
h_min = 30  # altura minima do retangulo
offset = 2  # Erro permitido entre pixel
linha_ROI = 550  # Posição da linha de contagem
carros = 0


def centroide(x, y, w, h):
    """
    :param x: x do objeto
    :param y: y do objeto
    :param w: largura do objeto
    :param h: altura do objeto
    :return: tupla que contém as coordenadas do centro de um objeto
    """
    x1 = w // 2
    y1 = h // 2
    cx = x + x1  # Coordenada x do centroide
    cy = y + y1  # Coordenada y do centroide
    return cx, cy


detec = []


def set_info(detec):
    global carros
    for (x, y) in detec:
        if (linha_ROI + offset) > y > (linha_ROI - offset):  # Verificação se o objeto atravessou a linha de contagem
            carros += 1  # Incremento do número de carros detectados
            cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (0, 127, 255),
                     3)  # Desenho da linha de contagem na imagem
            detec.remove((x, y))  # Remoção do objeto da lista de detecção
            print("Carros detectados até o momento: " + str(carros))  # Impressão do número de carros detectados


def show_info(frame, mask):
    text = f'Carros: {carros}'
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                5)  # Desenho do texto com o número de carros na imagem
    cv2.imshow("Video Original", frame)  # Exibição da imagem original
    cv2.imshow("Detectar", mask)  # Exibição da máscara


cap = cv2.VideoCapture(VIDEO)  # Leitura do vídeo

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Definição do codec do vídeo

algorithm_type = algorithm_types[0]  # Seleção do tipo de algoritmo de subtração de fundo
background_subtractor = subtractor(algorithm_type)  # Criação do objeto subtrator de fundo

while True:

    ok, frame = cap.read()  # Leitura de cada frame do vídeo

    if not ok:
        break

    mask = background_subtractor.apply(frame)  # Aplicação do algoritmo de subtração de fundo ao frame
    mask = Filter(mask, 'combine')  # Aplicação de filtro à máscara

    contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Encontrando contornos na máscara
    cv2.line(frame, (25, linha_ROI), (1200, linha_ROI), (255, 127, 0), 3)  # Desenho da linha de contagem na imagem

    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)  # Cálculo do retângulo delimitador de cada contorno
        validar_contorno = (w >= w_min) and (h >= h_min)  # Verificação se o retângulo atende às dimensões mínimas

        if not validar_contorno:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenho do retângulo delimitador na imagem
        centro = centroide(x, y, w, h)  # Cálculo do centroide do objeto
        detec.append(centro)  # Adição do centroide à lista de detecção
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)  # Desenho do círculo indicando o centroide na imagem

    set_info(detec)  # Atualização das informações sobre os carros detectados
    show_info(frame, mask)  # Exibição das informações na imagem

    if cv2.waitKey(1) == 27:  # Verificação se a tecla ESC foi pressionada
        break

cv2.destroyAllWindows()
cap.release()
