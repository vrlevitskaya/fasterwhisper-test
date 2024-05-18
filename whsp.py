from faster_whisper import WhisperModel
import os

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")
answ_list = []
folder_path = r'C:\Users\levit\PycharmProjects\testwhisper\voices'

if os.path.exists(folder_path):
    files = os.listdir(folder_path)

    for audio in files:
        _, info = model.transcribe(audio, beam_size=5)
        answ_list.append(info)

for info in answ_list:
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

