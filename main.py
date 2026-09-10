import cv2
import numpy as np
import onnxruntime as ort
import time
import random

# --- CONFIGURATION ---
MODEL_PATH = "rps_model.onnx"
CLASSES = ['paper', 'rock', 'scissors'] 

print(f"Loading model from {MODEL_PATH}...")
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure 'rps_model.onnx' is in the same folder as this script!")
    exit()

def get_winner(player, computer):
    player = player.lower()
    computer = computer.lower()
    if player == computer: return "Tie"
    if (player == "rock" and computer == "scissors") or \
       (player == "scissors" and computer == "paper") or \
       (player == "paper" and computer == "rock"):
        return "Player Wins!"
    return "Computer Wins!"

cap = cv2.VideoCapture(0)

# Game State
state = "IDLE" # States: IDLE, COUNTDOWN, CAPTURE, RESULT
countdown_start = 0
result_start = 0
final_message = ""
p_move = ""
c_move = ""

# Box Coordinates (Region of Interest)
x1, y1, x2, y2 = 100, 100, 400, 400

print("\n--- CONTROLS ---")
print("Press 's' to START")
print("Press 'r' to RESTART (when game ends)")
print("Press 'q' to QUIT")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame for mirror effect (easier for humans)
    frame = cv2.flip(frame, 1)
    
    # Box Color Logic
    box_color = (0, 255, 0) # Green (Default)
    if state == "COUNTDOWN":
        box_color = (0, 0, 255) # Red (Get Ready!)
    elif state == "CAPTURE":
        box_color = (0, 255, 255) # Yellow (Processing...)
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    
    # --- GAME LOGIC ---
    if state == "IDLE":
        cv2.putText(frame, "Press 'S' to Start", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Put hand in box", (x1, y1-20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    elif state == "COUNTDOWN":
        elapsed = time.time() - countdown_start
        if elapsed < 1:
            text = "3"
        elif elapsed < 2:
            text = "2"
        elif elapsed < 3:
            text = "1"
        else:
            state = "CAPTURE"
            countdown_start = time.time() # Reset timer for the "Hold" phase
            text = "SHOW HAND!"
        
        # Draw countdown text
        if state == "COUNTDOWN":
            cv2.putText(frame, text, (220, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)

    elif state == "CAPTURE":
        # Give user 0.5 seconds to freeze their hand AFTER '1' disappears
        elapsed = time.time() - countdown_start
        cv2.putText(frame, "HOLD IT!", (150, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        if elapsed > 0.5: # Wait half a second before snapping photo
            # 1. Capture the Region of Interest (ROI)
            roi = frame[y1:y2, x1:x2]
            
            # 2. Resize to what the model expects (150x150)
            img = cv2.resize(roi, (150, 150))
            
            # --- DEBUG WINDOW: See what the AI sees ---
            cv2.imshow("What AI Sees", img) 
            # ------------------------------------------

            # 3. Preprocess for AI
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0) # Add batch dimension
            img = img / 255.0 # Normalize pixel values to 0-1
            
            # 4. Predict
            predictions = session.run(None, {input_name: img})[0]
            idx = np.argmax(predictions)
            p_move = CLASSES[idx]
            
            # 5. Computer Move & Winner
            c_move = random.choice(CLASSES)
            final_message = get_winner(p_move, c_move)
            
            state = "RESULT"

    elif state == "RESULT":
        # Display Results
        cv2.putText(frame, f"You: {p_move}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"CPU: {c_move}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Color the winner text
        res_color = (255, 255, 255)
        if "Player" in final_message: res_color = (0, 255, 0)
        elif "Computer" in final_message: res_color = (0, 0, 255)
            
        cv2.putText(frame, final_message, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, res_color, 3)
        cv2.putText(frame, "Press 'r' to restart", (50, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    # Show the main game window
    cv2.imshow("Rock Paper Scissors", frame)
    
    key = cv2.waitKey(1)
    
    # Control Logic
    if key == ord('q'): 
        break
    if key == ord('s') and state == "IDLE":
        state = "COUNTDOWN"
        countdown_start = time.time()
    if key == ord('r') and state == "RESULT":
        state = "IDLE"
        cv2.destroyWindow("What AI Sees") # Close the debug window between rounds

cap.release()
cv2.destroyAllWindows()