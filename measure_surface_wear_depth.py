import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from skimage.filters import threshold_otsu
from scipy import ndimage

# Inicializar a câmera e o pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)
time.sleep(5)  # pausa para dar tempo do sensor pegar todos os pontos
# Adicionando o filtro de decimação
# decimation = rs.decimation_filter()
hole_filling = rs.hole_filling_filter()
# spatial_filter = rs.spatial_filter()
# temporal_filter = rs.temporal_filter()


# Alinhamento dos frames de cor e profundidade
align_to = rs.stream.color
align = rs.align(align_to)

# Obter quadros da câmera
frames = pipeline.wait_for_frames()
frames = align.process(frames)
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

img = np.asanyarray(color_frame.get_data())
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = threshold_otsu(img_gray)
img_otsu  = img_gray < thresh

# cv2.imwrite('bloco4.png', img_otsu.astype(np.uint8) * 255) #descomente esta linha para realizar a segmentação do objeto e depois comente novamente para não criar de novo

# Aplicar o filtros
# depth_frame = spatial_filter.process(depth_frame)
# depth_frame = temporal_filter.process(depth_frame)
depth_frame = hole_filling.process(depth_frame)

# Carregar a máscara
mask = cv2.imread('bloco4.png', cv2.IMREAD_GRAYSCALE)# Redimensionar a máscara para corresponder à resolução do quadro de profundidade após decimação
# depth_image_temp = np.asanyarray(depth_frame.get_data())
# mask = cv2.resize(mask, (depth_image_temp.shape[1], depth_image_temp.shape[0]))
mask = ndimage.binary_erosion(mask, structure=np.ones((35, 35))).astype(mask.dtype)

# Obter imagem de profundidade

depth_image = np.asanyarray(depth_frame.get_data())
depth_image_smoothed = depth_image
depth_image_smoothed = cv2.GaussianBlur(depth_image, (5, 5), 0)

# Mascarar a imagem de profundidade
masked_depth = np.where((mask > 0.5) & (depth_image_smoothed <= 420), depth_image_smoothed, 0) #420 para os blocos maiores esse número é a medição da câmera até o fundo do objeto

Y, X = np.meshgrid(np.linspace(0, 1, masked_depth.shape[1]), np.linspace(0, 1, masked_depth.shape[0]))

# Extraia os pontos válidos da imagem de profundidade e da máscara
valid_points = masked_depth[masked_depth > 0]
valid_Y = Y[masked_depth > 0]
valid_X = X[masked_depth > 0]

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

# Função para gerar relatório
def generate_report(mask, masked_depth):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    valid_distances = masked_depth[mask > 0.5]
    mean_distance = 366  # 370 Medida de distância da câmera até a superfície frontal do objeto (onde começa o objeto)
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
    Distância Máxima dentro da região de desgaste bloco 4: {max_distance} mm
    """
    
    plt.figtext(0.65, 0.3, stats_text, fontsize=10, ha='left')
    plt.tight_layout()
    plt.show()

# Gerando o relatório no final
generate_report(mask, masked_depth)