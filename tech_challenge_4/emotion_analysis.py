import cv2
from deepface import DeepFace
import os
from tqdm import tqdm
import operator as op
import mediapipe as mp

def detect_emotions(video_path, output_video_path, output_summary_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    identified_emotions = []
    arms_up_counter = 0
    hands_besides_head = 0
    anomalous_pose_counter = 0
    def is_arm_up(landmarks):
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        left_arm_up = left_elbow.y < left_eye.y
        right_arm_up = right_elbow.y < right_eye.y

        return left_arm_up or right_arm_up
    
    def is_hands_besides_head(landmarks):
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        return left_wrist.y > left_ear.y and right_wrist.y < right_ear.y

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poses = pose.process(rgb_frame)
        emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for face in emotions:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            dominant_emotion = face['dominant_emotion']
            identified_emotions.append(dominant_emotion)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        if poses.pose_landmarks:
            arm_up_flag = is_arm_up(poses.pose_landmarks.landmark)
            if arm_up_flag:
                arms_up_counter += 1

            hands_besides_head_flag = is_hands_besides_head(poses.pose_landmarks.landmark)
            if hands_besides_head_flag:
                hands_besides_head += 1
            
            if not arm_up_flag and not hands_besides_head_flag:
                anomalous_pose_counter += 1
        
        cv2.putText(frame, f'Bracos levantados: {arms_up_counter} | Maos ao lado da cabeca: {hands_besides_head} | Pose anomala: {anomalous_pose_counter}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        out.write(frame)

    grouped_emotions = []
    x = list(set(identified_emotions))
    for i in x:
        a = op.countOf(identified_emotions, i)
        b = "".join(i)
        grouped_emotions.append((a, b))

    full_summary = []
    for key, value in grouped_emotions:
        full_summary.append(f"A emoção {value} apareceu {key} vezes")

    full_summary.append(f"Braços levantados: {arms_up_counter}")
    full_summary.append(f"Mãos ao lado da cabeça: {hands_besides_head}")
    full_summary.append(f"Poses anomalas: {anomalous_pose_counter}")
    full_summary.append(f"Total de frames analisados: {total_frames}")

    with open(output_summary_path, "w") as file:
        file.write("\n".join(full_summary))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'input_video.mp4')
output_video_path = os.path.join(script_dir, 'output_video.mp4')
output_summary_path = os.path.join(script_dir, 'summary.txt')

detect_emotions(input_video_path, output_video_path, output_summary_path)