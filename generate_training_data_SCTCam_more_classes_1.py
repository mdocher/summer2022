'''
GENERATE VERITAS EVENTS: LASER, MONO, STEREO, MUON
5000 EVENTS IN EACH QUADRANT
'''

import time
import numpy as np
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
import matplotlib.pyplot as plt
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.image import toymodel
import astropy.units as u

from tqdm import tqdm
from numpy.random import randn
import random, os

from numpy import savetxt
import h5py

number_of_data_points = 2500


#camera_list = ["VERITAS", "MAGICCam","HESS-I","HESS-II","FACT","CHEC","ASTRICam","DigiCam","FlashCam","NectarCam","LSTCam","SCTCam"]
camera_type = "SCTCam"
training_data_dir = "/home/kdocher/code/" + camera_type + "/"


def generate_noise():
    os.makedirs(training_data_dir + "Noise", exist_ok=True)
    os.makedirs(training_data_dir + "Noise_param", exist_ok=True)
    for i in tqdm(range(number_of_data_points)):
        poisson_mean = random.randint(5,10)
        data = np.random.poisson(poisson_mean, size = (11328)) #Number of VERITAS pixel
        
        camgeom = CameraGeometry.from_name(camera_type)
        
        # Save data as text file
        text_file_name = training_data_dir + "Noise" + "/_M1_%s_M2_%s_raw.txt" %(i, i)
        savetxt(text_file_name , data, delimiter = ',')
        
        # Save basic parameters as text file
        text_file_name = training_data_dir + "Noise_param" + "/param_M1_%s_M2_%s_raw.txt" %(i, i)
        header_text = "poisson_mean"
        param = [poisson_mean]
        savetxt(text_file_name , param, delimiter = ',', header=header_text)
        
        if i < 10:
            number_of_pixels = np.shape(data)[0]
            #print(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            disp = CameraDisplay(camgeom, show_frame=False, cmap = "afmhot")
            disp.image = data
            disp.add_colorbar()
            disp.set_limits_minmax(0.,30.)
            plt.gca().set_title(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            plt.xlim([-0.55, 0.55])
            plt.ylim([-0.55, 0.55])
            plt.savefig(training_data_dir+"Noise/_%s.png" %(i))
            plt.close()

def generate_laser():
    os.makedirs(training_data_dir + "Laser", exist_ok=True)
    os.makedirs(training_data_dir + "Laser_param", exist_ok=True)
    for i in tqdm(range(number_of_data_points)):
                
        x_1 = random.randint(-15,15)/100.
        y_1 = -random.randint(-15,15)/100.
        r = random.randint(20,25)/4000.
        s = random.randint(50,80)/10000.

        model_1 = toymodel.RingGaussian(
            x=(x_1) * u.m,
            y=(y_1) * u.m,
            radius=(r) * u.m,
            sigma=(s) * u.m,)

        intensity_1	 = random.randint(10000,12500)
        nsb_level = 2
        
        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        data = image_1 + nsb_level
        #print(np.shape(data))

        # Save data as text file
        text_file_name = training_data_dir + "Laser" + "/_M1_%s_M2_%s_raw.txt" %(i, i)
        savetxt(text_file_name , data, delimiter = ',')
        
        # Save basic parameters as text file
        text_file_name = training_data_dir + "Laser_param" + "/param_M1_%s_M2_%s_raw.txt" %(i, i)
        header_text = "x_1,y_1,r,s,intensity_1,nsb_level"
        param = [x_1,y_1,r,s,intensity_1,nsb_level]
        savetxt(text_file_name , param, delimiter = ',', header=header_text)
        
        if i < 10:
            number_of_pixels = np.shape(data)[0]
            #print(str(camgeom) + " " + str(number_of_pixels) + " pixels")

            disp = CameraDisplay(camgeom, show_frame=False, cmap = "afmhot")
            disp.image = data
            disp.add_colorbar()
            disp.set_limits_minmax(0.,30.)
            plt.gca().set_title(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            plt.xlim([-0.55, 0.55])
            plt.ylim([-0.55, 0.55])
            plt.savefig(training_data_dir+"Laser/_%s.png" %(i))
            plt.close()
            
def generate_muon():
    os.makedirs(training_data_dir + "Muon", exist_ok=True)
    os.makedirs(training_data_dir + "Muon_param", exist_ok=True)
    
    for i in tqdm(range(number_of_data_points)):

        x_1 = random.randint(-15,15)/100.
        y_1 = -random.randint(-15,15)/100.
        r = random.randint(10,30)/100.
        s = random.randint(5,20)/700.

        
        model_1 = toymodel.RingGaussian(
            x=(x_1) * u.m,
            y=(y_1) * u.m,
            radius=(r) * u.m,
            sigma=(s) * u.m,)

        intensity_1	 = random.randint(6000,8500)
        nsb_level = 2
        
        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        data = image_1 + nsb_level

        # Save data as text file
        text_file_name = training_data_dir + "Muon" + "/_M1_%s_M2_%s_raw.txt" %(i, i)
        savetxt(text_file_name , data, delimiter = ',')
        
        # Save basic parameters as text file
        text_file_name = training_data_dir + "Muon_param" + "/param_M1_%s_M2_%s_raw.txt" %(i, i)
        header_text = "x_1,y_1,r,s,intensity_1,nsb_level"
        param = [x_1,y_1,r,s,intensity_1,nsb_level]
        savetxt(text_file_name , param, delimiter = ',', header=header_text)

        if i < 10:
            number_of_pixels = np.shape(data)[0]
            #print(str(camgeom) + " " + str(number_of_pixels) + " pixels")

            disp = CameraDisplay(camgeom, show_frame=False, cmap = "afmhot")
            disp.image = data
            disp.add_colorbar()
            disp.set_limits_minmax(0.,30.)
            plt.gca().set_title(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            plt.xlim([-0.55, 0.55])
            plt.ylim([-0.55, 0.55])
            plt.savefig(training_data_dir+"Muon/_%s.png" %(i))
            plt.close()

def generate_mono_shower_1():
    os.makedirs(training_data_dir + "Mono_shower_1", exist_ok=True)
    os.makedirs(training_data_dir + "Mono_shower_param_1", exist_ok=True)
    
    for i in tqdm(range(number_of_data_points)):
    
        x_1 = random.randint(5,15)/100.
        y_1 = random.randint(5,15)/100.
        width_1 = random.randint(15,30)/1000.
        length_1 = random.randint(5,10)/100.
        psi_start = random.randint(5,85)
        skew_1 = np.random.normal(0.75,0.05,1)[0]
        
        intensity_1	 = random.randint(1000,3500)
        
        nsb_level = 2
        
        model_1 = toymodel.SkewedGaussian(
            x=(x_1) * u.m,
            y=(y_1) * u.m,
            width=(width_1) * u.m,
            length=(length_1) * u.m,
            psi= (psi_start )* u.deg,
            skewness=skew_1)

        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        data = image_1 + nsb_level
        

        # Save data as text file
        text_file_name = training_data_dir + "Mono_shower_1" + "/_M1_%s_M2_%s_raw.txt" %(i, i)
        savetxt(text_file_name , data, delimiter = ',')

        # Save basic parameters as text file
        text_file_name = training_data_dir + "Mono_shower_param_1" + "/param_M1_%s_M2_%s_raw.txt" %(i, i)
        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]
        savetxt(text_file_name , param, delimiter = ',', header=header_text)

        if i < 10:
            number_of_pixels = np.shape(data)[0]
            #print(str(camgeom) + " " + str(number_of_pixels) + " pixels")

            disp = CameraDisplay(camgeom, show_frame=False, cmap = "afmhot")
            disp.image = data
            disp.add_colorbar()
            disp.set_limits_minmax(0.,30.)
            plt.gca().set_title(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            plt.xlim([-0.55, 0.55])
            plt.ylim([-0.55, 0.55])
            plt.savefig(training_data_dir+"Mono_shower_1/_%s.png" %(i))
            plt.close()

def generate_mono_shower_2():
    os.makedirs(training_data_dir + "Mono_shower_2", exist_ok=True)
    os.makedirs(training_data_dir + "Mono_shower_param_2", exist_ok=True)
    
    for i in tqdm(range(number_of_data_points)):
    
        x_1 = -random.randint(5,15)/100.
        y_1 = random.randint(5,15)/100.
        width_1 = random.randint(15,30)/1000.
        length_1 = random.randint(5,10)/100.
        psi_start = random.randint(95,175)
        skew_1 = np.random.normal(0.75,0.05,1)[0]
        
        intensity_1	 = random.randint(1000,3500)
        
        nsb_level = 2
        
        model_1 = toymodel.SkewedGaussian(
            x=(x_1) * u.m,
            y=(y_1) * u.m,
            width=(width_1) * u.m,
            length=(length_1) * u.m,
            psi= (psi_start )* u.deg,
            skewness=skew_1)

    
        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        data = image_1 + nsb_level
        

        # Save data as text file
        text_file_name = training_data_dir + "Mono_shower_2" + "/_M1_%s_M2_%s_raw.txt" %(i, i)
        savetxt(text_file_name , data, delimiter = ',')

        # Save basic parameters as text file
        text_file_name = training_data_dir + "Mono_shower_param_2" + "/param_M1_%s_M2_%s_raw.txt" %(i, i)
        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]
        savetxt(text_file_name , param, delimiter = ',', header=header_text)

        if i < 10:
            number_of_pixels = np.shape(data)[0]
            #print(str(camgeom) + " " + str(number_of_pixels) + " pixels")

            disp = CameraDisplay(camgeom, show_frame=False, cmap = "afmhot")
            disp.image = data
            disp.add_colorbar()
            disp.set_limits_minmax(0.,30.)
            plt.gca().set_title(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            plt.xlim([-0.55, 0.55])
            plt.ylim([-0.55, 0.55])
            plt.savefig(training_data_dir+"Mono_shower_2/_%s.png" %(i))
            plt.close()


def generate_mono_shower_3():
    os.makedirs(training_data_dir + "Mono_shower_3", exist_ok=True)
    os.makedirs(training_data_dir + "Mono_shower_param_3", exist_ok=True)
    
    for i in tqdm(range(number_of_data_points)):
    
        x_1 = -random.randint(5,15)/100.
        y_1 = -random.randint(5,15)/100.
        width_1 = random.randint(15,30)/1000.
        length_1 = random.randint(5,10)/100.
        psi_start = random.randint(185,265)
        skew_1 = np.random.normal(0.75,0.05,1)[0]
        
        intensity_1	 = random.randint(1000,3500)
        
        nsb_level = 2
        
        model_1 = toymodel.SkewedGaussian(
            x=(x_1) * u.m,
            y=(y_1) * u.m,
            width=(width_1) * u.m,
            length=(length_1) * u.m,
            psi= (psi_start )* u.deg,
            skewness=skew_1)

    
        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        data = image_1 + nsb_level
        

        # Save data as text file
        text_file_name = training_data_dir + "Mono_shower_3" + "/_M1_%s_M2_%s_raw.txt" %(i, i)
        savetxt(text_file_name , data, delimiter = ',')

        # Save basic parameters as text file
        text_file_name = training_data_dir + "Mono_shower_param_3" + "/param_M1_%s_M2_%s_raw.txt" %(i, i)
        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]
        savetxt(text_file_name , param, delimiter = ',', header=header_text)

        if i < 10:
            number_of_pixels = np.shape(data)[0]
            #print(str(camgeom) + " " + str(number_of_pixels) + " pixels")

            disp = CameraDisplay(camgeom, show_frame=False, cmap = "afmhot")
            disp.image = data
            disp.add_colorbar()
            disp.set_limits_minmax(0.,30.)
            plt.gca().set_title(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            plt.xlim([-0.55, 0.55])
            plt.ylim([-0.55, 0.55])
            plt.savefig(training_data_dir+"Mono_shower_3/_%s.png" %(i))
            plt.close()

def generate_mono_shower_4():
    os.makedirs(training_data_dir + "Mono_shower_4", exist_ok=True)
    os.makedirs(training_data_dir + "Mono_shower_param_4", exist_ok=True)
    
    for i in tqdm(range(number_of_data_points)):
    
        x_1 = random.randint(5,15)/100.
        y_1 = -random.randint(5,15)/100.
        width_1 = random.randint(15,30)/1000.
        length_1 = random.randint(5,10)/100.
        psi_start = random.randint(275,355)
        skew_1 = np.random.normal(0.75,0.05,1)[0]
        
        intensity_1	 = random.randint(1000,3500)
        
        nsb_level = 2
        
        model_1 = toymodel.SkewedGaussian(
            x=(x_1) * u.m,
            y=(y_1) * u.m,
            width=(width_1) * u.m,
            length=(length_1) * u.m,
            psi= (psi_start )* u.deg,
            skewness=skew_1)

    
        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        data = image_1 + nsb_level

        # Save data as text file
        text_file_name = training_data_dir + "Mono_shower_4" + "/_M1_%s_M2_%s_raw.txt" %(i, i)
        savetxt(text_file_name , data, delimiter = ',')

        # Save basic parameters as text file
        text_file_name = training_data_dir + "Mono_shower_param_4" + "/param_M1_%s_M2_%s_raw.txt" %(i, i)
        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]
        savetxt(text_file_name , param, delimiter = ',', header=header_text)

        if i < 10:
            number_of_pixels = np.shape(data)[0]
            #print(str(camgeom) + " " + str(number_of_pixels) + " pixels")

            disp = CameraDisplay(camgeom, show_frame=False, cmap = "afmhot")
            disp.image = data
            disp.add_colorbar()
            disp.set_limits_minmax(0.,30.)
            plt.gca().set_title(str(camgeom) + " " + str(number_of_pixels) + " pixels")
            plt.xlim([-0.55, 0.55])
            plt.ylim([-0.55, 0.55])
            plt.savefig(training_data_dir+"Mono_shower_4/_%s.png" %(i))
            plt.close()

def main():
    time_start = time.time()
    generate_noise()
    generate_laser()
    generate_muon()
    generate_mono_shower_1()
    generate_mono_shower_2()
    generate_mono_shower_3()
    generate_mono_shower_4()
    
    time_end = time.time()
    print(time_end - time_start)

if __name__ == "__main__":
    main()

