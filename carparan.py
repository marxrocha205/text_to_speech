import csv
import os

# Diretório onde os arquivos .wav estão localizados
audio_dir = 'data/'

# Dados dos áudios e suas transcrições
audio_data = [
    ("girafali1.wav", "Olá, milha-amigos! Hoje vamos explorar algo muito especial e importante, nossas emoções! Sabiam que cada um de nós tem um jardim de emoções dentro de si? Vamos aprender juntos como reconhecer e cuidar das nossas emoções? Esta é a Flor da Alegria. Ela é brilhante e vermelha. Quando estamos felizes, é como se essa flor estivesse florescendo dentro de nós. E esta é a Flor Azul, que representa a tristeza."),
    ("girafali2.wav", "Todos nós ficamos tristes às vezes, mas está tudo bem. É como uma chuva que rega o nosso jardim. Vamos ver o que mais temos aqui? Ah! A Flor Amarela, de surpresa! Surpresas podem ser divertidas e emocionantes, não é mesmo? Essa flor verde é a calma. Quando estamos calmos, podemos pensar com clareza e desfrutar de momentos de paz, quando estamos com soninho, por exemplo. E não podemos nos esquecer da Flor Roxa, de frustração.")
]

# Criação dos arquivos CSV
def create_csv(csv_file, data):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'text'])
        for audio_file, transcription in data:
            writer.writerow([os.path.join(audio_dir, audio_file), transcription])

# Divida os dados em treinamento e validação (80/20)
split_index = int(len(audio_data) * 0.8)
train_data = audio_data[:split_index]
val_data = audio_data[split_index:]

# Crie os arquivos CSV
create_csv('data/train.csv', train_data)
create_csv('data/val.csv', val_data)

print("Arquivos CSV criados com sucesso!")
