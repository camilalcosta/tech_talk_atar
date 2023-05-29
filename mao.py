import cv2
import mediapipe as mp

"""
Para melhorar a precisão na contagem dos dedos, você pode ajustar o código da seguinte maneira:

Ajustar as coordenadas dos dedos e do polegar de acordo com a orientação correta das mãos detectadas no Mediapipe. 
Certifique-se de que as coordenadas estejam corretamente alinhadas com os dedos e o polegar da mão.

Aumentar o tamanho do círculo desenhado nos marcos das mãos para melhorar a detecção e reduzir possíveis falsos negativos.

Utilizar uma abordagem mais robusta para a verificação dos dedos levantados, levando em consideração a posição e a 
orientação dos marcos das mãos. Você pode considerar utilizar técnicas como ângulos entre os marcos dos dedos ou 
verificar a distância entre os marcos para determinar se um dedo está levantado ou não.

Realizar um pré-processamento da imagem, como a aplicação de filtros ou aprimoramento de contraste, para melhorar 
a detecção dos marcos das mãos.

Experimentar diferentes parâmetros e ajustes no mediapipe e nas configurações da câmera para obter melhores
resultados de detecção.

Lembrando que a precisão da contagem de dedos pode depender de diversos fatores, como a qualidade da imagem, o
posicionamento da mão e a iluminação do ambiente. Portanto, é importante realizar testes e ajustes para obter os
melhores resultados possíveis.
"""

cap = cv2.VideoCapture(0)  # Inicializa a captura de vídeo a partir da webcam (índice 0)
mp_hands = mp.solutions.hands  # Carrega o módulo de detecção de mãos do Mediapipe
hands = mp_hands.Hands()  # Cria uma instância do detector de mãos
mp_draw = mp.solutions.drawing_utils  # Utilitários para desenhar marcos e conexões nas mãos detectadas
finger_coords = [(8, 6), (12, 10), (16, 14), (20, 18)]  # Coordenadas dos dedos a serem verificados
thumb_coords = (4, 2)  # Coordenadas do polegar a serem verificadas

while True:
    success, image = cap.read()  # Lê um frame do vídeo capturado
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte o frame para o espaço de cores RGB
    results = hands.process(rgb_image)  # Processa o frame para detectar as mãos

    multi_landmarks = results.multi_hand_landmarks  # Obtém as informações dos marcos das mãos detectadas
    if multi_landmarks:
        hand_list = []
        for hand_landmarks in multi_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Desenha os marcos e conexões das mãos no frame
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Calcula as coordenadas dos marcos em pixels
                hand_list.append((cx, cy))  # Armazena as coordenadas dos marcos na lista hand_list
                for point in hand_list:
                    cv2.circle(image, point, 10, (255, 0, 0), cv2.FILLED)  # Desenha um círculo nos marcos das mãos
                    up_count = 0  # Inicializa o contador de dedos levantados

        for coordinate in finger_coords:
            if hand_list[coordinate[0]] < hand_list[coordinate[1]]:
                up_count += 1  # Verifica se os dedos estão levantados com base nas coordenadas da lista hand_list
        if hand_list[thumb_coords[0]] > hand_list[thumb_coords[1]]:
            up_count += 1  # Verifica se o polegar está levantado com base nas coordenadas da lista hand_list

        cv2.putText(image, str(up_count), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 0), 12)  # Adiciona o número de dedos levantados na imagem

    cv2.imshow("Movimento da mão", image)  # Exibe o frame com as marcações
    if cv2.waitKey(1) == ord('q'):  # Aguarda pressionar a tecla 'q' para sair do loop
        break

cap.release()  # Libera os recursos da captura de vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas abertas
