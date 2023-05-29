import cv2
import os
from PIL import Image


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


if __name__ == "__main__":
    tratar(r"img/captcha_0.png")
    # tratar(r"img/pronta_para_o_tesseract.png")








