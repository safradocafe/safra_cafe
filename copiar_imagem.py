import shutil
import os

# 1. Define os caminhos
caminho_original = r"D:\Arquivos_R\sr\sr_cafe\Santa_Vera\teste app\indices_capa.jpeg"
pasta_destino = "safra_cafe/imagens"
caminho_destino = os.path.join(pasta_destino, "indices_capa.jpeg")

# 2. Cria a pasta se não existir
os.makedirs(pasta_destino, exist_ok=True)

# 3. Copia a imagem
shutil.copy2(caminho_original, caminho_destino)

print(f"Imagem copiada para: {caminho_destino}")
