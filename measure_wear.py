import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Inicializar a câmera e o pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
# Adicionando o filtro de decimação
decimation = rs.decimation_filter()

# Obter quadros da câmera
frames = pipeline.wait_for_frames()
time.sleep(5)  # pausa para dar tempo do sensor pegar todos os pontos
depth_frame = frames.get_depth_frame()

# Aplicar o filtro de decimação
depth_frame = decimation.process(depth_frame)

# Carregar a máscara
mask = cv2.imread('D:/estudos/mestrado/3codigo_MSC/measurement_wear/codes/images/bloco_azul_otsu.png', cv2.IMREAD_GRAYSCALE)
# Redimensionar a máscara para corresponder à resolução do quadro de profundidade após decimação
depth_image_temp = np.asanyarray(depth_frame.get_data())
#mask = cv2.resize(mask, (depth_image_temp.shape[1], depth_image_temp.shape[0]), interpolation = cv2.INTER_NEAREST)
cv2.imshow('mask', mask)
cv2.waitKey(0) 


# Obter imagem de profundidade
depth_image = np.asanyarray(depth_frame.get_data())
cv2.imwrite('teste.png', depth_image)
depth_image_opencv = cv2.imread('teste.png', cv2.IMREAD_GRAYSCALE)
depth_image_opencv = cv2.resize(depth_image_opencv, (640,480),  interpolation = cv2.INTER_NEAREST)
print("depth_image "  + str(depth_image.shape))
print('depth_image_opencv ' + str(depth_image_opencv.shape))
cv2.imshow('depth_image_opencv',depth_image_opencv )
depth_image_smoothed = cv2.GaussianBlur(depth_image_opencv, (5, 5), 0)
cv2.imshow('depth_image_smoothed', depth_image_smoothed)
cv2.waitKey(0) 

# Mascarar a imagem de profundidade
masked_depth = np.where((mask > 0.5) & (depth_image_smoothed <= 355), depth_image_smoothed, 0)  

Y, X = np.meshgrid(np.linspace(0, 1, masked_depth.shape[1]), np.linspace(0, 1, masked_depth.shape[0]))

# Extraia os pontos válidos da imagem de profundidade e da máscara
valid_points = masked_depth[mask > 0]
valid_Y = Y[mask > 0]
valid_X = X[mask > 0]

# Visualizando a nuvem de pontos
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(valid_X, valid_Y, valid_points, c=valid_points, cmap='viridis', s=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Depth (mm)')
ax.set_title('Nuvem de pontos 3D da câmera de profundidade RealSense')
plt.show()

pipeline.stop()
cv2.destroyAllWindows() 
# Função para gerar relatório
def generate_report(mask, masked_depth):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    valid_distances = masked_depth[mask > 0.5]
    mean_distance = 325  # Distância da superfície do objeto à câmera
    std_distance = np.std(valid_distances)
    threshold = mean_distance + 1
    anomaly_mask = np.zeros_like(masked_depth)
    anomaly_mask[masked_depth > threshold] = 1
    
    axs[0].imshow(mask, cmap='gray')
    axs[0].imshow(anomaly_mask, cmap='Reds', alpha=0.6)
    axs[0].set_title('Máscara Binária com Desníveis em Destaque')
    axs[0].axis('off')
    
    axs[1].hist(valid_distances, bins=50, color='blue', edgecolor='black')
    axs[1].axvline(threshold, color='red', linestyle='dashed', linewidth=2, label='Threshold de Desníveis')
    axs[1].set_title('Distribuição de distâncias medidas')
    axs[1].set_xlabel('Distância (mm)')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    
    min_distance = np.min(valid_distances)
    max_distance = np.max(valid_distances)
    
    stats_text = f"""
    Distância Mínima: {min_distance} mm
    Distância Máxima: {max_distance} mm
    Distância Média: {mean_distance:.2f} mm
    Desvio padrão: {std_distance:.2f} mm
    """
    
    plt.figtext(0.65, 0.3, stats_text, fontsize=10, ha='left')
    plt.tight_layout()
    plt.show()

# Gerando o relatório no final
generate_report(mask, masked_depth)