import inspect
from TTS.api import TTS

# Carregando o modelo TTS
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"  # Substitua pelo seu modelo
tts = TTS(model_name)

# Obtendo a função tts_to_file
tts_to_file = tts.tts_to_file

# Listando os parâmetros do método tts_to_file
params = inspect.signature(tts_to_file).parameters

# Exibindo os parâmetros
print("Parâmetros do método tts_to_file:")
for param in params:
    print(f"- {param}")

# Exibir informações adicionais
print("\nInformações dos parâmetros:")
for param in params.values():
    print(f"{param.name}: {param.annotation}, default={param.default}")
