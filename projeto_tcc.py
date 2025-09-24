import sys
import queue
import json
import threading
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from PyQt5.QtCore import pyqtSignal, QObject
from vosk import Model, KaldiRecognizer
from llama_cpp import Llama

# === CONFIGURAÇÕES ===
VOSK_MODEL_PATH = r"C:\modelos_voskpy\vosk-model-small-pt-0.3"
LLM_MODEL_PATH = r"C:\modelosllm\tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
SAMPLERATE = 16000
MAX_TOKENS = 100

# === Inicialização do áudio e modelo ===
q = queue.Queue()
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, SAMPLERATE)

llm = Llama(model_path=r"C:/modelosllm/tinyllama-1.1b-chat-v1.0.Q2_K.gguf")

# === Classe para sinalizar atualizações para a GUI ===
class Communicate(QObject):
    nova_entrada = pyqtSignal(str)
    nova_resposta = pyqtSignal(str)

comms = Communicate()

# === Função de geração da resposta ===
def responder_com_llm(texto):
    try:
        prompt = (
            "Você é Aurelius, um assistente virtual educado, claro e objetivo. "
            "Responda sempre em português, de forma breve, direta e relevante. "
            "Nunca inclua exemplos, exercícios ou texto não relacionado. "
            "Se não souber a resposta, diga 'Desculpe, não sei responder isso.'\n\n"
            f"Usuário: {texto}\nAurelius:"
        )
        resposta_obj = llm(prompt, max_tokens=MAX_TOKENS, temperature=0.3, stop=["Usuário:"])
        resposta = resposta_obj["choices"][0]["text"].strip()
        comms.nova_resposta.emit(resposta)
    except Exception as e:
        comms.nova_resposta.emit(f"❌ Erro: {e}")

# === Callback do áudio ===
def audio_callback(indata, frames, time_info, status):
    q.put(bytes(indata))

# === Thread para processar o áudio continuamente ===
def processar_audio():
    global recognizer
    with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=8000, dtype='int16', channels=1, callback=audio_callback):
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                texto = result.get("text", "")
                if texto:
                    comms.nova_entrada.emit(texto)
                    threading.Thread(target=responder_com_llm, args=(texto,)).start()
                    recognizer = KaldiRecognizer(vosk_model, SAMPLERATE)

# === GUI ===
class AssistenteGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Assistente Virtual - Aurelius")
        self.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout()

        self.label_voz = QLabel("🗣️ Você disse:")
        self.txt_voz = QTextEdit()
        self.txt_voz.setReadOnly(True)

        self.label_resposta = QLabel("🤖 Aurelius respondeu:")
        self.txt_resposta = QTextEdit()
        self.txt_resposta.setReadOnly(True)

        self.btn_limpar = QPushButton("🧹 Limpar")
        self.btn_limpar.clicked.connect(self.limpar_chat)

        layout.addWidget(self.label_voz)
        layout.addWidget(self.txt_voz)
        layout.addWidget(self.label_resposta)
        layout.addWidget(self.txt_resposta)
        layout.addWidget(self.btn_limpar)

        self.setLayout(layout)

        # Conectar sinais para atualizar GUI
        comms.nova_entrada.connect(self.atualizar_entrada)
        comms.nova_resposta.connect(self.atualizar_resposta)

    def atualizar_entrada(self, texto):
        self.txt_voz.append(texto)

    def atualizar_resposta(self, texto):
        self.txt_resposta.append(texto)

    def limpar_chat(self):
        self.txt_voz.clear()
        self.txt_resposta.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AssistenteGUI()
    gui.show()

    # Inicia thread de áudio
    threading.Thread(target=processar_audio, daemon=True).start()

    sys.exit(app.exec_())
