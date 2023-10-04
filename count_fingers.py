import cv2
#importa la biblioteca mediapipe
import mediapipe as mp

cap = cv2.VideoCapture(0)

#crea dos variables para poder incluir los metodos hands y drawing
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
#crea una variable donde guardaremos el algoritmo de valor de confianza
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

#funcion para dibujar las conexiones
def drawHandsLandmarks(image,hand_landmarks):
    # Dibujar las conexiones entre los puntos de referencia
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image,landmarks,mp_hands.HAND_CONNECTIONS)

while True:
    success, image = cap.read()
    #cambiar de posicion la mano con .flip()
    image = cv2.flip(image,1)
    #detectar los puntos de referencia de las manos con .process()
    results = hands.process(image)

    # Obtener la posición de los puntos de referencia del resultado procesado
    hand_landmarks = results.multi_hand_landmarks

    # Dibujar puntos de referencia con drawHandLanmarks
    drawHandsLandmarks(image,hand_landmarks)

    cv2.imshow("Controlador de medios", image)

    key = cv2.waitKey(1)
    if key == 32:
        break 

cv2.destroyAllWindows()
