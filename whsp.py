import whisper
import os
from deep_translator import GoogleTranslator

model_size = "large-v3"

# Run on GPU with FP16
model = whisper.load_model("base")
answ_list = []
folder_path = r'C:\Users\levit\PycharmProjects\testwhisper\voices'

if os.path.exists(folder_path):
    files = os.listdir(folder_path)

    for audio in files:
        audio_path = os.path.join(folder_path, audio)  # Создаем полный путь к аудиофайлу
        info = model.transcribe(audio_path)
        print(info["text"])
        answ_list.append(info["text"])

for info in answ_list:
    print("Transcribed audio", info)
    print("Transcribed and translated audio", GoogleTranslator(source='auto', target='ru').translate(info))