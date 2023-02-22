#importando as libs que eu preciso
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from PIL import Image as im
from numpy import asarray
import sys
np.set_printoptions(threshold=sys.maxsize)

import pickle

#função pra pegar a disntância entre a câmera realsense e o objeto de interesse
def get_distance_array(depth_frame):
    mask1 = pickle.load(open('D:/estudos/mestrado/code_msc/realsense/masks5.sav','rb'))
    seconds = time.time()
    depth_image = np.asanyarray(depth_frame.get_data())
    depth = np.zeros( (depth_image.shape[0], depth_image.shape[1]))
    depth255 = np.zeros( (depth_image.shape[0], depth_image.shape[1]))
    x = tuple(range(0, depth_image.shape[1], 1))
    y = tuple(range(0, depth_image.shape[0], 1))
    for i in range(len(y)):
        for j in range(len(x)):
            if mask1 [0][i][j] > 0:
                depth[i, j] = depth_frame.get_distance(j, i)
                if(depth[i, j] > 2):
                    depth[i, j] = 2
                if(depth[i, j] < 0):
                    depth[i, j] = 0
                depth[i, j] = (((depth[i, j]))*255)
            else:
                depth[i, j] = -1
            
                
    seconds = time.time()- seconds
    return (depth, seconds)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 15)
#config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 15)
config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 15)

# Start streaming
profile = pipeline.start(config)
#Set distance configs
sensor_dep = profile.get_device().query_sensors()[0]
sensor_dep.set_option(rs.option.laser_power, 20)

colorizer = rs.colorizer()

colorizer.set_option(rs.option.min_distance, 0.4)
print("min_distance = %d" % colorizer.get_option(rs.option.min_distance) )
colorizer.set_option(rs.option.max_distance, 1.1)
print("max_distance = %d" % colorizer.get_option(rs.option.max_distance) )

try:
    ok_flag = True
    while ok_flag==True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame() 
        #color_frame = frames.get_color_frame()
        infra1_frame = frames.get_infrared_frame(1)
        
        #Calcule distance array from camera
        array_distance, tempo = get_distance_array(depth_frame)
        for distance in array_distance:
            print(distance)
        
        print("Tempo processamento frame: ", tempo)
        image_distance = im.fromarray(array_distance)
        image_distance = image_distance.convert("RGB")

        #Colorizer
        depth_frame = colorizer.colorize(depth_frame)
    
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Convert depth_color_image [R,G,B, 3 channel] to depth_gray_image [1 channel]
        depth_gray_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)

        infrared_image = np.asanyarray(infra1_frame.get_data())
        #infrared_image = im.fromarray(infrared_image)
        
        infrared_depth = np.dstack([infrared_image, infrared_image, depth_gray_image]).astype(np.uint8)
        infrared_depth = im.fromarray(infrared_depth)
        
        depth_gray_image = im.fromarray(depth_gray_image)
        infrared_image = im.fromarray(infrared_image)
        infrared_depth = infrared_depth.convert("RGB")
        infrared_image = infrared_image.convert("RGB")
        depth_gray_imageRGB = depth_gray_image.convert("RGB")
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)
        
        images = np.vstack((np.hstack((infrared_depth, image_distance)), 
                           (np.hstack((infrared_image, depth_gray_imageRGB)))))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
    
        
        if cv2.waitKey(1) == 27:
            ok_flag = False
            
    cv2.destroyAllWindows()
    save_ply(frames)
finally:

    # Stop streaming
    pipeline.stop()