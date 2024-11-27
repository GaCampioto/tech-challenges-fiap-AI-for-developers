import cv2
from deepface import DeepFace
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def detect_emotions_and_plot_anomalies(video_path, output_video_path, output_summary_path, output_graph_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obtendo informações do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps  # Tempo total do vídeo em segundos

    # Configuração do vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    identified_emotions = []
    frame_timestamps = []
    anomalies = []
    previous_emotion = None

    frame_count = 0

    # Processando o vídeo
    for frame_number in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Analisar emoções a cada 5 frames
        emotions = []
        if frame_count % 5 == 0:
            try:
                emotions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            except Exception as e:
                print(f"Erro ao processar emoções no frame {frame_count}: {e}")
                continue

        # Processar resultados das emoções
        if emotions:
            for face in emotions:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                dominant_emotion = face['dominant_emotion']
                timestamp = frame_count / fps  # Tempo atual do frame em segundos
                identified_emotions.append((dominant_emotion, timestamp))
                frame_timestamps.append(timestamp)

                # Detectando anomalias (mudanças bruscas)
                if previous_emotion and dominant_emotion != previous_emotion:
                    anomalies.append((timestamp, previous_emotion, dominant_emotion))
                
                previous_emotion = dominant_emotion

                # Adicionando detecção no vídeo
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x+10, y+h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        out.write(frame)

    # Agrupar e calcular estatísticas
    emotion_durations = {}
    for emotion, timestamp in identified_emotions:
        emotion_durations[emotion] = emotion_durations.get(emotion, 0) + (1 / fps) * 5  # Duração acumulada por emoção

    # Criar resumo com base no tempo total
    full_summary = [f"Duração total do vídeo: {video_duration:.2f} segundos"]
    for emotion, duration in emotion_durations.items():
        percentage = (duration / video_duration) * 100
        full_summary.append(f"A emoção '{emotion}' apareceu por {duration:.2f} segundos | {percentage:.2f}% do vídeo")
    full_summary.append(f"Total de anomalias detectadas: {len(anomalies)}")

    # Salvar resumo
    with open(output_summary_path, "w") as file:
        file.write("\n".join(full_summary))

    # Plotando o gráfico
    unique_emotions = list(set(emotion for emotion, _ in identified_emotions))
    emotion_map = {emotion: i for i, emotion in enumerate(unique_emotions)}
    mapped_emotions = [emotion_map[emotion] for emotion, _ in identified_emotions]

    plt.figure(figsize=(15, 7))
    plt.plot(frame_timestamps, mapped_emotions, label="Emoções detectadas", color="blue", marker='o', linestyle='-')

    # Destaque para anomalias
    for timestamp, prev_emotion, curr_emotion in anomalies:
        plt.axvline(x=timestamp, color="red", linestyle="--", alpha=0.99)
       # plt.text(timestamp, -0.5, f"{prev_emotion} → {curr_emotion}", rotation=90, color="red", fontsize=8)

    # Configurações do gráfico
    plt.yticks(range(len(unique_emotions)), unique_emotions)
    plt.xlabel("Tempo (segundos)")
    plt.ylabel("Emoções")
    plt.title("Detecção de Emoções e Mudanças Bruscas (Anomalias)")
    plt.legend(["Emoções", "Mudanças bruscas"])
    plt.grid(True)
    plt.tight_layout()

    # Salvando o gráfico
    plt.savefig(output_graph_path)
    plt.show()

    # Finalizar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processamento concluído, resumo e gráfico gerados com sucesso!")

# Caminhos dos arquivos
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'parts/part_1.mp4')  # 
output_video_path = os.path.join(script_dir, 'output_video_vidal.mp4')
output_summary_path = os.path.join(script_dir, 'summary.txt')
output_graph_path = os.path.join(script_dir, 'emotion_anomalies_graph.png')

# Executar função
detect_emotions_and_plot_anomalies(input_video_path, output_video_path, output_summary_path, output_graph_path)
