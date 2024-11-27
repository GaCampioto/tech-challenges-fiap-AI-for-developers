import cv2
import os

def split_video(video_path, output_dir, num_parts):
    # Certifique-se de que o diretório de saída existe
    os.makedirs(output_dir, exist_ok=True)

    # Abre o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obtém as propriedades do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calcula o número de frames por parte
    frames_per_part = total_frames // num_parts

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Processa cada parte
    for part in range(num_parts):
        start_frame = part * frames_per_part
        end_frame = start_frame + frames_per_part if part < num_parts - 1 else total_frames
        
        output_file = os.path.join(output_dir, f"part_{part + 1}.mp4")
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"Processando parte {part + 1}: frames {start_frame} a {end_frame}")
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()

    cap.release()
    print("Divisão do vídeo concluída!")

# Configurações
input_video_path = "input_video.mp4"
output_dir = "parts"
num_parts = 10  # Número de partes que deseja dividir

split_video(input_video_path, output_dir, num_parts)
