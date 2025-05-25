import sys
import os
from pathlib import Path
import keyboard  # Adicione esta biblioteca

def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path = nvidia_base_path / 'cuda_runtime' / 'bin'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add = [str(cuda_path), str(cublas_path), str(cudnn_path)]
    env_vars = ['CUDA_PATH', 'CUDA_PATH_V12_4', 'PATH']
    
    for env_var in env_vars:
        current_value = os.environ.get(env_var, '')
        new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
        os.environ[env_var] = new_value

set_cuda_paths()



import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel  # Importando o faster-whisper 
import queue
import threading
import pyautogui
import pyperclip
import time

# Carregar o modelo do Faster Whisper
# Primeiro parâmetro é o tamanho do modelo: "tiny", "base", "small", "medium", "large"
# compute_type pode ser "int8", "float16", ou "float32"
model = WhisperModel("large", device="cuda", compute_type="float32", cpu_threads=4, num_workers=3)

# Parâmetros de gravação
samplerate = 16000  # Taxa de amostragem compatível com Whisper
block_duration = 5  # duração de cada bloco em segundos
block_size = samplerate * block_duration

# fila para comunicação entre a captura de áudio e a transcrição
q = queue.Queue()

# Controlador de pausa
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
    # Armazena o áudio na fila
    q.put(indata.copy())

def transcribe_stream():
    buffer_audio = np.array([], dtype=np.float32)
    print("Começando a transcrição. Fale algo...")
    print("O texto será digitado automaticamente onde seu cursor estiver.")
    print("Você tem 5 segundos para posicionar o cursor onde deseja que o texto apareça.")
    print("Pressione Ctrl+K para parar a execução.")  # Atualiza a mensagem
    time.sleep(5)  # Dar tempo ao usuário para posicionar o cursor

    try:
        while not stop_event.is_set():
            # Obtém o próximo bloco de áudio
            audio_block = q.get()
            # Adiciona ao buffer
            buffer_audio = np.concatenate((buffer_audio, audio_block[:,0]))

            # Quando o buffer alcançar o tamanho desejado, transcreve
            if len(buffer_audio) >= block_size:
                # Redimensiona para o tamanho exato
                to_transcribe = buffer_audio[:block_size]
                buffer_audio = buffer_audio[block_size:]

                # Normaliza o áudio para o formato esperado pelo Whisper
                audio_for_model = to_transcribe.astype(np.float32)

                # Transcreve com faster-whisper
                # O método transcribe do faster-whisper retorna um gerador + info
                segments, info = model.transcribe(
                    audio_for_model, 
                    language="pt", 
                    beam_size=5,
                    vad_filter=True
                )
                
                # Coleta todo o texto transcrito dos segmentos
                transcribed_text = ""
                for segment in segments:
                    transcribed_text += segment.text + " "
                
                # print("Transcrição:", transcribed_text) # muito chato isso aqui, fica no terminal printando
                
                # Digita o texto onde o cursor está posicionado usando clipboard para preservar acentos
                if transcribed_text.strip():  # Verifica se há texto para digitar
                    # Guarda o conteúdo atual do clipboard
                    previous_clipboard = pyperclip.paste()
                    
                    # Copia o texto transcrito para o clipboard
                    pyperclip.copy(transcribed_text + " ")  # Adiciona espaço ao final
                    
                    # Simula Ctrl+V para colar
                    pyautogui.hotkey('ctrl', 'v')
                    
                    # Restaura o clipboard anterior (opcional)
                    pyperclip.copy(previous_clipboard)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nTranscrição interrompida pelo usuário.")

# Função para detectar Ctrl+K
def on_ctrl_k():
    stop_event.set()
    print("\nTranscrição interrompida pelo usuário (Ctrl+K).")

# Registra o manipulador de Ctrl+K
keyboard.add_hotkey('ctrl+k', on_ctrl_k)

# Configuração do stream de áudio
stream = sd.InputStream(
    samplerate=samplerate,
    channels=1,
    callback=audio_callback
)

# Executando a captura e transcrição em threads separadas
transcription_thread = threading.Thread(target=transcribe_stream)

with stream:
    transcription_thread.start()
    # Mantém a execução até o usuário parar
    try:
        while not stop_event.is_set():
            sd.sleep(1000)
    except KeyboardInterrupt:
        stop_event.set()
        print("\nTranscrição interrompida pelo usuário (Ctrl+K).")

transcription_thread.join()
print("Programa encerrado.")
