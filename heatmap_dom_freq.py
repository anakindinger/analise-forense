import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# --- FUNÇÕES DE CÁLCULO E DETECÇÃO ---

def calculate_bispectrum_magnitude_simplified(block):
    """
    Função CONCEITUAL: Calcula uma métrica de correlação de terceira ordem (Biespectro)
    simplificada para um bloco de imagem 2D. 
    
    Em uma aplicação real, a função Biespectro 4D completa seria usada.
    Aqui, o foco é na magnitude média como indicador de não-linearidade.
    """
    
    # 1. Transformada de Fourier 2D
    Y = np.fft.fft2(block)
    Y_shift = np.fft.fftshift(Y)
    rows, cols = Y_shift.shape
    
    # Seleção de frequências de referência fixas (ex: baixa frequência próxima ao centro, mas não DC)
    w_ref_x, w_ref_y = rows // 2 + 1, cols // 2 + 1 
    Y_ref = Y_shift[w_ref_x, w_ref_y]
    
    B_2D = np.zeros_like(Y_shift, dtype=complex)
    
    for wx1 in range(rows):
        for wy1 in range(cols):
            wx_sum = wx1 + (w_ref_x - rows // 2) 
            wy_sum = wy1 + (w_ref_y - cols // 2)
            
            if (0 <= wx_sum < rows) and (0 <= wy_sum < cols):
                Y_w1 = Y_shift[wx1, wy1]
                Y_conj_sum = np.conj(Y_shift[wx_sum, wy_sum])
                
                # B(w1, w2) = Y(w1) * Y(w2) * Y*(w1 + w2)
                B_2D[wx1, wy1] = Y_w1 * Y_ref * Y_conj_sum
    
    # Retornamos a magnitude média do Biespectro no bloco.
    return np.mean(np.abs(B_2D))


def detect_forgery_bispectrum(img, block_size=32, step=16):
    """
    Aplica a análise biespectral em blocos e gera um mapa de anomalias (heatmap).
    """
    
    # Converte a imagem para escala de cinza e float32
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        img_gray = img.astype(np.float32)
        
    rows, cols = img_gray.shape
    
    # Listas para armazenar as coordenadas centrais e o valor da métrica
    bispectrum_values = []
    block_centers = []
    
    # 1. Segmentação da Imagem (Blocos Sobrepostos)
    for r in range(0, rows - block_size + 1, step):
        for c in range(0, cols - block_size + 1, step):
            # Extrai o bloco
            block = img_gray[r:r + block_size, c:c + block_size]
            
            # Subtrai a média para remover a componente DC e estabilizar a FFT
            block = block - np.mean(block)
            
            # 2. Cálculo da Métrica (Biespectro)
            bispec_val = calculate_bispectrum_magnitude_simplified(block)
            
            # Armazena o valor e a coordenada central do bloco
            bispectrum_values.append(bispec_val)
            block_centers.append((c + block_size // 2, r + block_size // 2))

    # Converte para arrays numpy
    bispectrum_values = np.array(bispectrum_values)
    block_centers = np.array(block_centers)
    
    # 3. Modelagem Estatística e Normalização
    
    # Encontra o limite (threshold) para normalizar a visualização.
    # Usamos o 90º percentil como um limite para destacar as anomalias mais significativas.
    # Em um caso real, o limite seria calibrado estatisticamente (ex: 3 desvios padrão acima da média).
    limit = np.percentile(bispectrum_values, 90)
    
    # Cria o Mapa de Anomalias normalizado entre 0 e 1 (para o heatmap)
    anomaly_map_raw = np.clip(bispectrum_values / limit, 0, 1)

    # 4. Interpolação para Visualização (Heatmap Suave)
    
    # Cria um grid para a imagem original
    grid_x, grid_y = np.mgrid[0:cols:1, 0:rows:1]
    
    # Interpola os valores do centro dos blocos para o tamanho total da imagem
    anomaly_heatmap = griddata(block_centers, anomaly_map_raw, 
                               (grid_x, grid_y), method='cubic', fill_value=0).T
                               
    # Normaliza o heatmap final para 0-255
    anomaly_heatmap = cv2.normalize(anomaly_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    return anomaly_heatmap.astype(np.uint8)

# --- DEMONSTRAÇÃO COM IMAGEM SIMULADA OU REAL ---

# Escolha: usar imagem simulada ou carregar uma real
USE_REAL_IMAGE = True  # Mude para False para usar imagem simulada
IMAGE_PATH = 'suri_alterado.jpg'  # Defina o caminho da imagem aqui, ex: "CNH_Aberta/00000000_gt_ocr.txt"

if USE_REAL_IMAGE and IMAGE_PATH is None:
    print("Imagens disponíveis na pasta CNH_Aberta:")
    import os
    cnh_files = [f for f in os.listdir('CNH_Aberta') if f.endswith('.txt')]
    for i, f in enumerate(cnh_files[:10]):
        print(f"  {i+1}. {f}")
    
    # Você pode escolher qual imagem usar modificando IMAGE_PATH
    # Exemplo: IMAGE_PATH = "CNH_Aberta/00000000_gt_ocr.txt"

if USE_REAL_IMAGE and IMAGE_PATH:
    # Carrega uma imagem real do arquivo de texto (OCR)
    # Se o arquivo for imagem (jpg, png, etc), ajuste o código abaixo:
    simulated_forgery = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if simulated_forgery is None:
        print(f"Erro ao carregar a imagem: {IMAGE_PATH}")
        print("Usando imagem simulada como fallback...")
        USE_REAL_IMAGE = False

if not USE_REAL_IMAGE or IMAGE_PATH is None:
    # 1. Criação de uma Imagem de Teste (simulando falsificação)
    image_size = 256
    # Imagem Base (natural: ruído branco com baixa correlação)
    natural_area = np.random.rand(image_size, image_size) * 150 + 50

    # Área Adulterada (introduzindo uma não-linearidade)
    # Falsificação: aplicar uma não-linearidade quadrática (y = x + 0.3*x^2), que aumenta a bicoerência
    forged_area = natural_area[100:200, 50:150]
    alpha = 0.3 # Coeficiente de não-linearidade
    forged_area_nonlinear = forged_area + alpha * (forged_area**2 / 255)
    forged_area_nonlinear = np.clip(forged_area_nonlinear, 0, 255)

    # Emendar a área adulterada de volta na imagem base
    simulated_forgery = natural_area.copy()
    simulated_forgery[100:200, 50:150] = forged_area_nonlinear

# 2. Executa a Detecção
heatmap_anomaly = detect_forgery_bispectrum(simulated_forgery, block_size=32, step=8)

# 3. Visualização dos Resultados
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# Original
axes[0].imshow(simulated_forgery, cmap='gray')
axes[0].set_title('1. Imagem de Teste (Falsificação Sim.)')
axes[0].axis('off')

# Heatmap puro
im1 = axes[1].imshow(heatmap_anomaly, cmap='hot', alpha=0.9)
axes[1].set_title('2. Mapa de Anomalias (Biespectro)')
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], shrink=0.6)

# Sobreposição
axes[2].imshow(simulated_forgery, cmap='gray', alpha=0.7)
im2 = axes[2].imshow(heatmap_anomaly, cmap='jet', alpha=0.5)
axes[2].set_title('3. Sobreposição (Detecção da Falsificação)')
axes[2].axis('off')

plt.suptitle('Detecção de Falsificação via Análise Biespectral (Conceitual)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Salva a figura se o backend atual for não-interativo (Agg), caso contrário mostra a janela
if plt.get_backend().lower().startswith('agg'):
    out_fname = 'heatmap.png'
    plt.savefig(out_fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figura salva em: {out_fname}')
else:
    plt.show()