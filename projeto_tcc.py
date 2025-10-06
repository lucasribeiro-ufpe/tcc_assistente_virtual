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

VOSK_MODEL_PATH = r"C:\vosk-model-small-pt-0.3"  
SAMPLERATE = 16000

LLM_MODEL_PATH = r"C:\modelosllm\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"  

# Inicializa MQTT
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# # Inicializa fila e modelos Vosk
# q = queue.Queue()
# vosk_model = Model(VOSK_MODEL_PATH)
# recognizer = KaldiRecognizer(vosk_model, SAMPLERATE)

#   Inicializando LLM
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=1024,       # Mantém razoável (aumenta coerência sem pesar memória)
    n_threads=8,      # Ideal se estiver no Raspberry Pi 4; pode ser 8 no PC
    use_mlock=False,
    verbose=False
)


# def resposta_valida(texto_resposta):
#     termos_invalidos = ["def ", "exercise", "print(", "import ", "class ", "```", "http", "<html>", "error"]
#     texto_lower = texto_resposta.lower()

#     for termo in termos_invalidos:
#         if termo in texto_lower:
#             return False

#     if len(texto_resposta) < 3 or len(texto_resposta) > 1500:
#         return False

#     return True

def responder_com_llm(texto):
    try:
        # texto_lower = texto.lower()

        # palavras_ativacao = [
        #     "aurelius", "aurélio", "aurélio?", "aurelius?",
        #     "ô aurelius", "ô aurélio",
        #     "ei aurelius", "fala aurelius"
        # ]

        # if not any(p in texto_lower for p in palavras_ativacao):
        #     print("Ignorado (sem palavra-chave de ativação).")
        #     return

        # for palavra in palavras_ativacao:
        #     texto_lower = texto_lower.replace(palavra, "")
        # texto_limpo = texto_lower.strip()

        prompt_base = """
        Você é uma assistente virtual chamada Aurelius.
        Seu objetivo é responder em português de forma breve, clara e educada.
        Se o usuário der um comando (como "ligar luz 1", "abrir portão", "desligar TV", "ligar relé 1"), responda SOMENTE com uma resposta curta no formato: 
        "LIGAR_LUZ_1","DESLIGAR_TV","ABRIR_PORTAO","LIGAR_RELE_1".
        Se o usuário fizer uma pergunta simples (como "qual é a capital do Brasil"),
        responda diretamente, sem enrolação.
        Se não souber a resposta, diga apenas: "Desculpe, não entendi o comando."
        """


        while True:
            user_input = input("Digite sua pergunta para LLM (CTRL+C para sair): ")

            if not user_input.strip():
                print("Aurelius: Não entendi. Tente novamente.")
                continue

            # Cria o prompt contextualizado (instrução + pergunta do usuário)
            prompt = f"{prompt_base}\nUsuário: {user_input}\nAurelius:"

            resposta_obj = llm(
                prompt,
                max_tokens=60,
                temperature=0.1,
                top_p=0.8,
                repeat_penalty=1.2,
                stop=["Usuário:", "Aurelius:"]
            )

            resposta = resposta_obj["choices"][0]["text"].strip()

            print("Aurelius:", resposta)
                # client.publish(TOPICO_RESPOSTA, resposta)

    except Exception as e:
        print("Erro ao gerar resposta:", e)


# def audio_callback(indata, frames, time_info, status):
#     q.put(bytes(indata))


# print(" Diga algo... (CTRL+C para sair)")
while True:
    texto = input("Digite sua pergunta para LLM(CTRL+C para sair): ")

    responder_com_llm(texto)

# with sd.RawInputStream(samplerate=SAMPLERATE, blocksize=8000, dtype='int16', channels=1, callback=audio_callback):
#     while True:
#         data = q.get()
#         if recognizer.AcceptWaveform(data):
#             result = json.loads(recognizer.Result())
#             texto = result.get("text", "")
#             if texto:
#                 print("Você disse:", texto)
#                 client.publish(TOPICO_ENTRADA, texto)
#                 responder_com_llm(texto)
