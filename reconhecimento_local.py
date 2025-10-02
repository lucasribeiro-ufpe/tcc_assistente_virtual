import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
from llama_cpp import Llama
import paho.mqtt.client as mqtt

# === CONFIGURAÇÕES ===

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPICO_ENTRADA = "assistente/voz"
TOPICO_RESPOSTA = "assistente/resposta"

VOSK_MODEL_PATH = r"C:\vosk-model-small-pt-0.3"  # 
SAMPLERATE = 16000

LLM_MODEL_PATH = r"C:\modelosllm\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  #

# Inicializa MQTT
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Inicializa fila e modelos Vosk
q = queue.Queue()
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, SAMPLERATE)

# Inicializa LLM
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=512,
    n_threads=2,
    use_mlock=False,
    verbose=False
)

# Histórico da conversa
historico = [
    "Usuário (Lucas): Qual é o seu nome?",
    "Aurelius: Meu nome é Aurelius. Estou aqui para te ajudar."
]

def resposta_valida(texto_resposta):
    termos_invalidos = ["def ", "exercise", "print(", "import ", "class ", "```", "http", "<html>", "error"]
    texto_lower = texto_resposta.lower()

    for termo in termos_invalidos:
        if termo in texto_lower:
            return False

    if len(texto_resposta) < 3 or len(texto_resposta) > 1500:
        return False

    return True

def responder_com_llm(texto):
    global historico
    try:
        texto_lower = texto.lower()

        palavras_ativacao = [
            "aurelius", "aurélio", "aurélio?", "aurelius?",
            "ô aurelius", "ô aurélio",
            "ei aurelius", "fala aurelius"
        ]

        if not any(p in texto_lower for p in palavras_ativacao):
            print("⏸️ Ignorado (sem palavra-chave de ativação).")
            return

        for palavra in palavras_ativacao:
            texto_lower = texto_lower.replace(palavra, "")
        texto_limpo = texto_lower.strip()


        prompt_base = (
            "Você é Aurelius, um assistente virtual educado, claro e objetivo. "
            "Responda sempre em português, de forma breve, direta e relevante. "
        )

        prompt = prompt_base

        resposta_obj = llm(prompt, max_tokens=120, temperature=0.6)
        resposta = resposta_obj["choices"][0]["text"].strip()

        if resposta_valida(resposta):
            print("🤖 Resposta válida:", resposta)
            client.publish(TOPICO_RESPOSTA, resposta)
        else:
            print("⚠️ Resposta inválida detectada. Enviando mensagem padrão.")
            client.publish(TOPICO_RESPOSTA, "Desculpe, não sei responder isso.")

    except Exception as e:
        print("❌ Erro ao gerar resposta:", e)


def audio_callback(indata, frames, time_info, status):
    q.put(bytes(indata))


print("🎤 Diga algo... (CTRL+C para sair)")

with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=8000, dtype='int16', channels=1, callback=audio_callback):
    while True:
        data = q.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            texto = result.get("text", "")
            if texto:
                print("🗣️ Você disse:", texto)
                client.publish(TOPICO_ENTRADA, texto)
                responder_com_llm(texto)
