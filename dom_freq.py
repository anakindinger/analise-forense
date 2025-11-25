import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

def calculate_bispectrum_2d(block):
    """
    Calcula e retorna a magnitude do Biespectro 4D simplificado (visualizado em 2D)
    para um bloco de imagem.
    
    A complexidade do Biespectro 4D (B(w1, w2, w3, w4)) é simplificada aqui
    para fins conceituais e de visualização, focando no princípio de correlação.
    """
    
    # 1. Transformada de Fourier 2D (Y)
    Y = np.fft.fft2(block)
    
    # Mover o componente de frequência zero (DC) para o centro para simplificar a indexação
    Y_shift = np.fft.fftshift(Y)
    rows, cols = Y_shift.shape
    
    # A Bicoerência é uma matriz 4D. Para visualização, fixamos as frequências w2.
    # Vamos criar uma matriz 2D que representa o biespectro em B(wx1, wy1)
    
    # Definição do Biespectro 4D: 
    # B(wx1, wy1, wx2, wy2) = Y(wx1, wy1) * Y(wx2, wy2) * Y*(wx1 + wx2, wy1 + wy2)
    
    # Seleção de uma frequência de referência fixa (ex: a baixa frequência central, excluindo DC)
    # Na prática, isso seria varrido, mas aqui é fixado para simplificar para 2D.
    # Exemplo: um ponto ligeiramente fora do centro (baixa frequência próxima)
    w_ref_x, w_ref_y = rows // 2 + 1, cols // 2 + 1 
    
    # Frequências de referência fixas (Y_ref = Y(wx2, wy2))
    Y_ref = Y_shift[w_ref_x, w_ref_y]
    
    # Biespectro Simplificado 2D (para visualização no plano w1):
    # B_2D(wx1, wy1) = Y(wx1, wy1) * Y_ref * Y*(wx1 + w_ref_x - center, wy1 + w_ref_y - center)
    
    # Esta é a parte mais complexa: o terceiro termo Y*(w1+w2) exige indexação cuidadosa.
    # Para fins de demonstração, criaremos uma matriz de Biespectro 2D:
    
    B_2D = np.zeros_like(Y_shift, dtype=complex)
    
    # Iterar sobre as frequências (wx1, wy1)
    for wx1 in range(rows):
        for wy1 in range(cols):
            
            # Calcular a posição do termo de soma (wx1+wx2, wy1+wy2) no array Y_shift
            # Deve-se levar em conta a mudança de índice (shift)
            wx_sum = wx1 + (w_ref_x - rows // 2) 
            wy_sum = wy1 + (w_ref_y - cols // 2)
            
            # Tratar o wraparound (índices fora do limite)
            if (0 <= wx_sum < rows) and (0 <= wy_sum < cols):
                
                Y_w1 = Y_shift[wx1, wy1]
                Y_conj_sum = np.conj(Y_shift[wx_sum, wy_sum])
                
                # Biespectro (sem normalização para Bicoerência)
                # B(w1, w2) = Y(w1) * Y(w2) * Y*(w1 + w2)
                B_2D[wx1, wy1] = Y_w1 * Y_ref * Y_conj_sum
    
    # Retorna a Magnitude do Biespectro 2D (Simplificado)
    return np.abs(B_2D)

# --- Demonstração ---

# 1. Carregar a imagem 
img = Image.open('suri.jpg').convert('L')  # Converter para escala de cinza
natural_block = np.array(img)

# 2. Usar uma imagem adulterada 
altered_img = Image.open('suri_alterado.jpg').convert('L')
forged_block = np.array(altered_img).astype(float)

# Garantir que os valores permaneçam no intervalo 0-255
forged_block = np.clip(forged_block, 0, 255)

# 3. Cálculo do Biespectro
B_natural = calculate_bispectrum_2d(natural_block)
B_forged = calculate_bispectrum_2d(forged_block)

# 4. Visualização dos Resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Imagens
axes[0, 0].imshow(natural_block, cmap='gray')
axes[0, 0].set_title('Bloco Natural')
axes[0, 0].axis('off')

axes[0, 1].imshow(forged_block, cmap='gray')
axes[0, 1].set_title('Bloco Adulterado (Não-Linearidade)')
axes[0, 1].axis('off')

# Biespectro (Magnitude)
# Usar log para visualização, pois os valores variam muito
B_natural_log = np.log(B_natural + 1e-6) # Adiciona epsilon para evitar log(0)
B_forged_log = np.log(B_forged + 1e-6)

axes[1, 0].imshow(B_natural_log, cmap='jet')
axes[1, 0].set_title(f'Magnitude do Biespectro (Natural)\n Média: {np.mean(B_natural):.2f}')
axes[1, 0].axis('off')

axes[1, 1].imshow(B_forged_log, cmap='jet')
axes[1, 1].set_title(f'Magnitude do Biespectro (Adulterado)\n Média: {np.mean(B_forged):.2f}')
axes[1, 1].axis('off')

plt.suptitle('Detecção Conceitual de Não-Linearidades (Falsificações) usando Biespectro')
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Salva a figura se o backend atual for não-interativo (Agg), caso contrário mostra a janela
if matplotlib.get_backend().lower().startswith('agg'):
    out_fname = 'biespectro_demo.png'
    plt.savefig(out_fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figura salva em: {out_fname}')
else:
    plt.show()