"""

O Tesseract é uma é uma API que possui tecnologia capaz de reconhecer caracteres a partir de
um arquivo de imagem com suporte a mais de 100 idiomas.
Então será que só o uso do Tesseract aqui seria suficiente para podermos obter a leitura dos caracteres desta imagem?
Vamos ver

"""

"""
SOBRE PSM no pytessract 
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
"""
import pytesseract
from pytesseract import image_to_string
from PIL import Image
import cv2
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


arquivos_brutos = ["img/captcha_0.png", "img/captcha_1.png"]

#
# for arquivo in arquivos_brutos:
#     for psm in range(3,13):
#         captcha = Image.open(arquivo)
#         caracteres_permitidos = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#         layout_de_analise = psm
#         texto = image_to_string(captcha, config=f"--psm {layout_de_analise} -c tessedit_char_whitelist={caracteres_permitidos}")
#         print(f'Captcha: {arquivo}, psm: {psm}, resultado :{texto}')

def tratar(path):

    imagem = cv2.imread(path)
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    arquivo, extensao = os.path.splitext(path)
    cv2.imwrite(f"{arquivo}_cinza{extensao}", imagem_cinza)

    # primeiro tratamento -> kernels
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilatada = cv2.dilate(imagem_cinza, kernel)


    # segundo tratamento - imagem dilatada com binarização
    _, imagem_dilatada = cv2.threshold(dilatada, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f"{arquivo}_imagem_dilatada{extensao}", imagem_dilatada)

    # terceiro tratamento com pillow - Limpeza para deixarmos apenas os Pixeis mais escuros

    imagem = Image.open(f"{arquivo}_imagem_dilatada.png")
    imagem.convert("P")
    imagem2 = Image.new("P", imagem.size, (255, 255, 255))

    for x in range(imagem.size[1]):
        for y in range(imagem.size[0]):
            cor_pixel = imagem.getpixel((y, x))
            if cor_pixel < 115:
                imagem2.putpixel((y, x), (0, 0, 0))

    imagem2.save(f"{arquivo}_processado.png")

# Depois de muito teste vamos usar tratamentos e psms diferentes para cada tipo de captcha

def ler_captcha_zero():
    tratar('img/captcha_0.png')
    captcha = Image.open("img/captcha_0_processado.png")
    caracteres_permitidos = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    layout_de_analise = 7
    texto = image_to_string(captcha,
                            config=f"--psm {layout_de_analise} -c tessedit_char_whitelist={caracteres_permitidos}")
    print(f'Captcha: {"img/captcha_0_processado.png"}, psm: {layout_de_analise}, resultado :{texto}')


def ler_captcha_um():
    tratar('img/captcha_1.png')
    captcha = Image.open("img/captcha_1_processado.png")
    caracteres_permitidos = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    layout_de_analise = 10
    texto = image_to_string(captcha,
                            config=f"--psm {layout_de_analise} -c tessedit_char_whitelist={caracteres_permitidos}")
    print(f'Captcha: {"img/captcha_1_processado_processado.png"}, psm: {layout_de_analise}, resultado :{texto}')

if __name__ == "__main__":
    ler_captcha_zero()
    ler_captcha_um()