import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

PATH = '/Users/niccolo/Downloads/Unbalanced.h5'
MIN_CONFIDENCE = 0.8
def main():
    model = tf.keras.models.load_model(PATH)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'space'}

    cap = cv2.VideoCapture(0)

    draw_landmarks = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks and draw_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z

                    x_.append(x)
                    y_.append(y)

                    # Append normalized x, y, and z coordinates for each landmark
                    data_aux.extend([x - min(x_), y - min(y_), z])

                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10

                x2 = int(max(x_) * frame.shape[1]) + 10
                y2 = int(max(y_) * frame.shape[0]) + 10

                prediction = model.predict([np.asarray(data_aux).reshape(1, -1)])
                print(prediction.argmax(), prediction[0][prediction.argmax()])
                if prediction[0][prediction.argmax()] > MIN_CONFIDENCE:
                    predicted_character = labels_dict[prediction.argmax()]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_character} - {'%.2f' % prediction[0][prediction.argmax()]}" , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            draw_landmarks = not draw_landmarks

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
