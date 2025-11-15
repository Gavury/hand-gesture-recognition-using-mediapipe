import cv2
import mediapipe as mp
import pyautogui
import time
import math

# --- Configurações ---
COOLDOWN_HANDS = 0.5
COOLDOWN_NAV = 1.5   # Cooldown de 1.5s para setas, espaço e faixas
COOLDOWN_VOL = 0.2   # Cooldown de 0.5s para volume
COOLDOWN_PAUSE = 1.0 # Cooldown de 1.0s para o gesto de pause/play
last_press_time_nav = 0
last_press_time_vol = 0
last_press_time_pause = 0

# --- NOVO: Estado de Pausar/Despausar ---
is_paused = False # Variável para controlar o estado da pausa
# --- FIM NOVO ---

# --- Inicialização do MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,  # Detecta até 2 mãos
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
 )  

# IDs dos pontos de referência
tip_ids = [4, 8, 12, 16, 20] 

# --- Calibração ---
OK_DISTANCE_THRESHOLD = 0.06 
TOUCH_DISTANCE_THRESHOLD = 0.05 

# --- Inicialização da Webcam ---
cap = cv2.VideoCapture(0)

print("Iniciando... Navegacao agora com Gesto 'Numero 2' (Paz).")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_time = time.time()
    action_performed = False 

    if is_paused:
        cv2.putText(image, "PAUSADO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.line(image, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 100), 2)
        cv2.line(image, (image.shape[1], 0), (0, image.shape[0]), (0, 0, 100), 2)

    if results.multi_hand_landmarks:
        
        # --- Lógica de Pausa (Gesto de 2 Mãos) ---
        is_fingers_touching = False
        if len(results.multi_hand_landmarks) == 2:
            left_hand_lm = None
            right_hand_lm = None
            for idx, hand_info in enumerate(results.multi_handedness):
                hand_label = hand_info.classification[0].label
                if hand_label == "Left":
                    left_hand_lm = results.multi_hand_landmarks[idx].landmark
                elif hand_label == "Right":
                    right_hand_lm = results.multi_hand_landmarks[idx].landmark
            
            if left_hand_lm and right_hand_lm:
                left_tip = left_hand_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                right_tip = right_hand_lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distancia_toque = math.hypot(left_tip.x - right_tip.x, left_tip.y - right_tip.y)
                if distancia_toque < TOUCH_DISTANCE_THRESHOLD:
                    is_fingers_touching = True

        if is_fingers_touching and (current_time - last_press_time_pause > COOLDOWN_PAUSE):
            if is_paused:
                cv2.putText(image, "DESPAUSANDO...", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print("Toque detectado -> DESPAUSANDO")
            else:
                cv2.putText(image, "PAUSANDO...", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Toque detectado -> PAUSANDO")
            
            is_paused = not is_paused 
            action_performed = True
            last_press_time_pause = current_time

        # --- MODO ATIVO ---
        if not is_paused and not action_performed:
        
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                if action_performed:
                    break 
                
                lm = hand_landmarks.landmark 
                
                # Cálculo do tamanho da palma (para referência)
                wrist = lm[0]
                middle_mcp = lm[9]
                palm_size = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)

                # --- Lógica 1: Gesto "Número 2" (Paz e Amor) ---
                # Substitui a antiga "Mão Aberta"
                buffer_open = 0.01 

                # Indicador deve estar RETO (Regra da Escada)
                index_straight = (lm[8].y < lm[7].y - buffer_open) and \
                                 (lm[7].y < lm[6].y - buffer_open) and \
                                 (lm[6].y < lm[5].y - buffer_open)

                # Médio deve estar RETO (Regra da Escada)
                middle_straight = (lm[12].y < lm[11].y - buffer_open) and \
                                  (lm[11].y < lm[10].y - buffer_open) and \
                                  (lm[10].y < lm[9].y - buffer_open)

                # Anelar deve estar FECHADO (Ponta abaixo da junta PIP)
                ring_closed = lm[16].y > lm[14].y 

                # Mindinho deve estar FECHADO (Ponta abaixo da junta PIP)
                pinky_closed = lm[20].y > lm[18].y

                # Combinação: Indicador e Médio ABERTOS + Anelar e Mindinho FECHADOS
                is_peace_sign = index_straight and middle_straight and ring_closed and pinky_closed


                # --- Lógica 2: Gesto "L" (Anti-Garra + Pontas Afastadas) ---
                buffer_l = 0.02
                index_straight_l = (lm[8].y < lm[7].y - buffer_l) and \
                                   (lm[7].y < lm[6].y - buffer_l) and \
                                   (lm[6].y < lm[5].y - buffer_l)

                middle_closed_l = lm[tip_ids[2]].y > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_closed_l = lm[tip_ids[3]].y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
                pinky_closed_l = lm[tip_ids[4]].y > lm[mp_hands.HandLandmark.PINKY_PIP].y
                
                thumb_tip = lm[4]
                index_mcp = lm[5]
                thumb_index_dist = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
                is_thumb_extended = thumb_index_dist > (palm_size * 0.5)

                tips_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
                is_tips_far_apart = tips_dist > palm_size 
                
                is_l_gesture = is_thumb_extended and index_straight_l and middle_closed_l and \
                               ring_closed_l and pinky_closed_l and is_tips_far_apart

                # --- Lógica 3: Gesto "OK" ---
                distancia_polegar_indicador = math.hypot(
                    lm[mp_hands.HandLandmark.THUMB_TIP].x - lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    lm[mp_hands.HandLandmark.THUMB_TIP].y - lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                )
                is_thumb_index_close = distancia_polegar_indicador < OK_DISTANCE_THRESHOLD
                middle_open = lm[tip_ids[2]].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_open = lm[tip_ids[3]].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
                pinky_open = lm[tip_ids[4]].y < lm[mp_hands.HandLandmark.PINKY_PIP].y
                is_ok_gesture = is_thumb_index_close and middle_open and ring_open and pinky_open
                
                # --- Lógica 4: Mindinho RIGOROSAMENTE Esticado ---
                buffer = 0.02
                pinky_straight_f = (lm[20].y < lm[19].y - buffer) and \
                                   (lm[19].y < lm[18].y - buffer) and \
                                   (lm[18].y < lm[17].y - buffer)
                
                index_closed_only = lm[tip_ids[1]].y > lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_closed_only = lm[tip_ids[2]].y > lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_closed_only = lm[tip_ids[3]].y > lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
                
                is_pinky_up_gesture = pinky_straight_f and index_closed_only and middle_closed_only and ring_closed_only
                
                
                # --- Bloco de Decisão ---
                
                if is_peace_sign: # MUDANÇA AQUI: "is_hand_open" virou "is_peace_sign"
                    if (current_time - last_press_time_nav > COOLDOWN_HANDS):
                        hand_info = results.multi_handedness[idx]
                        hand_label = hand_info.classification[0].label 
                        if hand_label == 'Right':
                            cv2.putText(image, "NUMERO 2 DIREITA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print("Gesto '2' (Direita) -> Pressionando CTRL + SETA DIREITA")
                            pyautogui.press('right')#pyautogui.hotkey('ctrl', 'right')
                            action_performed = True
                            last_press_time_nav = current_time 
                        elif hand_label == 'Left':
                            cv2.putText(image, "NUMERO 2 ESQUERDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            print("Gesto '2' (Esquerda) -> Pressionando CTRL + SETA ESQUERDA")
                            pyautogui.press('left')#pyautogui.hotkey('ctrl', 'left')
                            action_performed = True
                            last_press_time_nav = current_time
                
                elif is_l_gesture:
                    if (current_time - last_press_time_vol > COOLDOWN_VOL):
                        hand_info = results.multi_handedness[idx]
                        hand_label = hand_info.classification[0].label 
                        if hand_label == 'Right':
                            cv2.putText(image, "L DIREITA - VOL +", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            print("Gesto 'L' (Direita) -> Pressionando VOLUME UP")
                            pyautogui.press('volumeup')
                            action_performed = True
                            last_press_time_vol = current_time 
                        elif hand_label == 'Left':
                            cv2.putText(image, "L ESQUERDA - VOL -", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
                            print("Gesto 'L' (Esquerda) -> Pressionando VOLUME DOWN")
                            pyautogui.press('volumedown')
                            action_performed = True
                            last_press_time_vol = current_time
                
                elif is_ok_gesture:
                    if (current_time - last_press_time_nav > COOLDOWN_NAV):
                        hand_info = results.multi_handedness[idx]
                        hand_label = hand_info.classification[0].label
                        if hand_label == 'Right': 
                            cv2.putText(image, "OK DIREITA - ESPACO", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            print("Gesto 'OK' (Direita) -> Pressionando ESPACO")
                            pyautogui.press('space') 
                            action_performed = True
                            last_press_time_nav = current_time
                
                elif is_pinky_up_gesture:
                    if (current_time - last_press_time_nav > COOLDOWN_NAV):
                        cv2.putText(image, "MINDINHO - Tecla F", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                        print("Gesto 'Mindinho' detectado -> Pressionando F")
                        pyautogui.press('f')
                        action_performed = True
                        last_press_time_nav = current_time

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
             mp_drawing.draw_landmarks(
                image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow('Controle por Gestos - MediaPipe', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()