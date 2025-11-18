import cv2
import mediapipe as mp
import pyautogui
import time
import math

# --- Configurações ---
COOLDOWN_HANDS = 0.3
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
OK_DISTANCE_THRESHOLD = 0.055
TOUCH_DISTANCE_THRESHOLD = 0.05 
Z_TOUCH_THRESHOLD = 0.08

# --- Inicialização da Webcam ---
cap = cv2.VideoCapture(0)

print("Iniciando... Feedback visual imediato ativado.")

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
                left_tip = left_hand_lm[8]
                right_tip = right_hand_lm[8]
                distancia_toque_xy = math.hypot(left_tip.x - right_tip.x, left_tip.y - right_tip.y)
                is_2d_close = distancia_toque_xy < TOUCH_DISTANCE_THRESHOLD
                distancia_toque_z = abs(left_tip.z - right_tip.z)
                is_z_aligned = distancia_toque_z < Z_TOUCH_THRESHOLD
                
                # Só ativa se estiver perto na tela E na profundidade
                if is_2d_close and is_z_aligned:
                    is_fingers_touching = True

        if is_fingers_touching:
            # Mostra o feedback de "pausa" imediatamente
            if is_paused:
                cv2.putText(image, "GESTO: DESPAUSAR", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(image, "GESTO: PAUSAR", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # E agora checa o cooldown para a AÇÃO
            if (current_time - last_press_time_pause > COOLDOWN_PAUSE):
                if is_paused:
                    print("Toque detectado -> DESPAUSANDO")
                else:
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
                
                # --- DADOS GERAIS ---
                wrist = lm[0]
                middle_mcp = lm[9]
                palm_size = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
                
                # --- Lógica 1: Gesto "Número 2" (Polegar Cruzado) ---
                buffer_open = 0.01 
                index_straight = (lm[8].y < lm[7].y - buffer_open) and (lm[7].y < lm[6].y - buffer_open) and (lm[6].y < lm[5].y - buffer_open)
                middle_straight = (lm[12].y < lm[11].y - buffer_open) and (lm[11].y < lm[10].y - buffer_open) and (lm[10].y < lm[9].y - buffer_open)
                ring_closed = lm[16].y > lm[13].y 
                pinky_closed = lm[20].y > lm[17].y
                thumb_tip = lm[4]
                pinky_mcp = lm[17]
                thumb_to_pinky_dist = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y)
                is_thumb_tightly_closed = thumb_to_pinky_dist < (palm_size * 0.6)
                is_peace_sign = index_straight and middle_straight and ring_closed and pinky_closed and is_thumb_tightly_closed

                # --- Lógica 2: Gesto "L" (COM DISTÂNCIA DAS PONTAS) ---
                buffer_l = 0.02
                index_straight_l = (lm[8].y < lm[7].y - buffer_l) and (lm[7].y < lm[6].y - buffer_l) and (lm[6].y < lm[5].y - buffer_l)
                middle_closed_l = lm[12].y > lm[9].y
                ring_closed_l = lm[16].y > lm[13].y
                pinky_closed_l = lm[20].y > lm[17].y
                thumb_tip = lm[4]
                index_mcp = lm[5]
                thumb_index_dist = math.hypot(thumb_tip.x - index_mcp.x, thumb_tip.y - index_mcp.y)
                is_thumb_extended = thumb_index_dist > (palm_size * 0.5)
                tips_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
                is_tips_far_apart = tips_dist > (palm_size * 1.3) 
                is_l_gesture = is_thumb_extended and index_straight_l and middle_closed_l and \
                               ring_closed_l and pinky_closed_l and is_tips_far_apart

                # --- Lógica 3: Gesto "OK" ---
                distancia_polegar_indicador = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
                is_thumb_index_close = distancia_polegar_indicador < OK_DISTANCE_THRESHOLD
                middle_open = lm[12].y < lm[11].y and lm[11].y < lm[10].y
                ring_open = lm[16].y < lm[15].y and lm[15].y < lm[14].y
                pinky_open = lm[20].y < lm[19].y and lm[19].y < lm[18].y
                is_ok_gesture = is_thumb_index_close and middle_open and ring_open and pinky_open
                
                # --- Lógica 4: Mindinho RIGOROSAMENTE Esticado ---
                buffer = 0.02
                distancia_polegar_indicador = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
                is_thumb_index_close = distancia_polegar_indicador < (OK_DISTANCE_THRESHOLD*2.6)
                pinky_straight_f = (lm[20].y < lm[19].y - buffer) and (lm[19].y < lm[18].y - buffer) and (lm[18].y < lm[17].y - buffer)
                index_closed_only = lm[8].y > lm[5].y
                middle_closed_only = lm[12].y > lm[9].y
                ring_closed_only = lm[16].y > lm[13].y
                is_pinky_up_gesture = pinky_straight_f and index_closed_only and middle_closed_only and ring_closed_only and is_thumb_index_close
                
                
                # --- Bloco de Decisão (COM FEEDBACK VISUAL) ---
                
                if is_peace_sign:
                    hand_info = results.multi_handedness[idx]
                    hand_label = hand_info.classification[0].label 
                    
                    # Feedback visual (Azul, sempre que o gesto for detectado)
                    if hand_label == 'Right':
                        cv2.putText(image, "GESTO: Paz DIREITA (Setas)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    elif hand_label == 'Left':
                        cv2.putText(image, "GESTO: Paz ESQUERDA (Setas)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Ação (só acontece se o cooldown tiver passado)
                    if (current_time - last_press_time_nav > COOLDOWN_NAV):
                        if hand_label == 'Right':
                            print("Gesto '2' (Direita) -> Pressionando SETA DIREITA")
                            pyautogui.press('right')#pyautogui.hotkey('ctrl', 'right')
                        elif hand_label == 'Left':
                            print("Gesto '2' (Esquerda) -> Pressionando SETA ESQUERDA")
                            pyautogui.press('left')#pyautogui.hotkey('ctrl', 'left')
                        action_performed = True
                        last_press_time_nav = current_time
                
                elif is_l_gesture:
                    hand_info = results.multi_handedness[idx]
                    hand_label = hand_info.classification[0].label 
                    
                    # Feedback visual
                    if hand_label == 'Right':
                        cv2.putText(image, "GESTO: L DIREITA (Vol+)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    elif hand_label == 'Left':
                        cv2.putText(image, "GESTO: L ESQUERDA (Vol-)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Ação
                    if (current_time - last_press_time_vol > COOLDOWN_VOL):
                        if hand_label == 'Right':
                            print("Gesto 'L' (Direita) -> Pressionando VOLUME UP")
                            pyautogui.press('volumeup')
                        elif hand_label == 'Left':
                            print("Gesto 'L' (Esquerda) -> Pressionando VOLUME DOWN")
                            pyautogui.press('volumedown')
                        action_performed = True
                        last_press_time_vol = current_time
                
                elif is_ok_gesture:
                    hand_info = results.multi_handedness[idx]
                    hand_label = hand_info.classification[0].label
                    
                    # Feedback visual
                    if hand_label == 'Right': 
                        cv2.putText(image, "GESTO: OK DIREITA (Espaco)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    elif hand_label == 'Left':
                        cv2.putText(image, "GESTO: OK ESQUERDA (Espaco)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Ação
                    if (current_time - last_press_time_nav > COOLDOWN_NAV):
                        if hand_label == 'Right': 
                            print("Gesto 'OK' (Direita) -> Pressionando ESPACO")
                            pyautogui.press('space') 
                            action_performed = True
                            last_press_time_nav = current_time
                        elif hand_label == 'Left':
                            print("Gesto 'OK' (Esquerda) -> Pressionando ESPACO")
                            pyautogui.press('space') 
                            action_performed = True
                            last_press_time_nav = current_time    
                
                elif is_pinky_up_gesture:
                    # Feedback visual
                    cv2.putText(image, "GESTO: MINDINHO (Tecla F)", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Ação
                    if (current_time - last_press_time_nav > COOLDOWN_NAV):
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
