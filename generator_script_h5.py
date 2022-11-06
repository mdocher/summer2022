'''
GENERATE VERITAS EVENTS: NOISE, LASER, MUON, MONO SHOWERS IN EACH OF THE FOUR GRAPHICAL QUADRANTS
2500 EVENTS IN EACH QUADRANT
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
#could run this for all 12 cameras 
camera_type = "SCTCam"
training_data_dir = "/datax/scratch/kdocher/" 

#label = ["noise","laser","muon","mono_shower_1","mono_shower_2","mono_shower_3","mono_shower4"]
label = [0,1,2,3,4,5,6]

def generate_noise():
    os.makedirs(training_data_dir + "Noise", exist_ok=True)
    #os.makedirs(training_data_dir + "Noise_param", exist_ok=True)
    h5_file_name = training_data_dir + "noise" + "_raw.hdf5" #%(i, i)

    f1 = h5py.File(h5_file_name, "w")

    h5_param_file_name = training_data_dir + "noise" + "_param_raw.hdf5" #%(i, i)

    f2 = h5py.File(h5_param_file_name, "w")
    header_text = "poisson_mean"

    signal_type = label[0]


    for i in tqdm(range(number_of_data_points)):
        poisson_mean = random.randint(5,10)
        param = [poisson_mean]
        noise_data = np.random.poisson(poisson_mean, size = (11328)) #Number of SCT cam pixels

        camgeom = CameraGeometry.from_name(camera_type)

        group_name_1 = 'noise_'+str(i+1) #image name
        grp1 = f1.create_group(group_name_1)

        group_name_2 = 'noise_param'+str(i+1) 
        grp2 = f2.create_group(group_name_2)        

        noise = np.array(noise_data, dtype=np.float32)
        noise_array = np.expand_dims(noise, axis=0)
        noise_label = np.array(signal_type, dtype=np.int32)
        noise_label_array = np.expand_dims(noise_label, axis=0)

        grp1.create_dataset('data', data=noise_array)
        grp1.create_dataset('label', data=noise_label_array)   #signal_type = label[0]

        grp2.create_dataset('data', data=np.array(param, dtype=np.float32))
        grp2.create_dataset('label', data=np.array(signal_type, dtype=np.int32))
        
        # Save data as h5 file
        #dset1 = f1.create_dataset("dataset"+str(i), dtype='i', data=data)

        #dset2 = f2.create_dataset(header_text+str(i), dtype='i', data=param)
                

    f1.close()
    f2.close()



def generate_laser():
    os.makedirs(training_data_dir + "Laser", exist_ok=True)
    #os.makedirs(training_data_dir + "Laser_param", exist_ok=True)
    h5_file_name = training_data_dir + "laser" + "_raw.hdf5" #%(i, i)
    f3 = h5py.File(h5_file_name, "w")
    h5_param_file_name = training_data_dir + "laser" + "_param_raw.hdf5" #%(i, i)
    f4 = h5py.File(h5_param_file_name, "w")

    signal_type = label[1]

    for i in tqdm(range(number_of_data_points)):
         
        x_1 = random.randint(-15,15)/100.
        y_1 = random.randint(-15,15)/100.
        r = random.randint(20,25)/4000.
        s = random.randint(50,80)/10000.

        model_1 = toymodel.RingGaussian(
            x=(x_1) * u.m,
            y=(y_1) * u.m,
            radius=(r) * u.m,
            sigma=(s) * u.m,)
        
        intensity_1	 = random.randint(10000,12500)
        nsb_level = 2

        header_text = "x_1,y_1,r,s,intensity_1,nsb_level"
        param = [x_1,y_1,r,s,intensity_1,nsb_level]

        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        laser_data = image_1 + nsb_level

        group_name_1 = 'laser_'+str(i+1) #image name
        grp1 = f3.create_group(group_name_1)

        group_name_2 = 'laser_param'+str(i+1) 
        grp2 = f4.create_group(group_name_2)     

        laser = np.array(laser_data, dtype=np.float32)
        laser_array = np.expand_dims(laser, axis=0)
        laser_label = np.array(signal_type, dtype=np.int32)
        laser_label_array = np.expand_dims(laser_label, axis=0)   

        grp1.create_dataset('data', data=laser_array)
        grp1.create_dataset('label', data=laser_label_array)   #signal_type = label[0]

        grp2.create_dataset('data', data=np.array(param, dtype=np.float32))
        grp2.create_dataset('label', data=np.array(signal_type, dtype=np.int32))

        #dset1 = f3.create_dataset("dataset"+ "_" + str(i), dtype='i', data=data)
        #dset2 = f4.create_dataset(header_text+ "_" + str(i), dtype='i', data=param)


    f3.close()
    f4.close()


def generate_muon():
    os.makedirs(training_data_dir + "muon", exist_ok=True)
    #os.makedirs(training_data_dir + "muon_param", exist_ok=True)
    h5_file_name = training_data_dir + "muon" + "_raw.hdf5" #%(i, i)
    f5 = h5py.File(h5_file_name, "w")
    h5_param_file_name = training_data_dir + "muon" + "_param_raw.hdf5" #%(i, i)
    f6 = h5py.File(h5_param_file_name, "w")
    
    signal_type = label[2]

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

        header_text = "x_1,y_1,r,s,intensity_1,nsb_level,"
        param = [x_1,y_1,r,s,intensity_1,nsb_level]
        
        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        muon_data = image_1 + nsb_level

        group_name_1 = 'muon_'+str(i+1) #image name
        grp1 = f5.create_group(group_name_1)

        group_name_2 = 'muon_param'+str(i+1) 
        grp2 = f6.create_group(group_name_2)  

        muon = np.array(muon_data, dtype=np.float32)
        muon_array = np.expand_dims(muon, axis=0)
        muon_label = np.array(signal_type, dtype=np.int32)
        muon_label_array = np.expand_dims(muon_label, axis=0)      

        grp1.create_dataset('data', data=muon_array)
        grp1.create_dataset('label', data=muon_label_array)   #signal_type = label[0]

        grp2.create_dataset('data', data=np.array(param, dtype=np.float32))
        grp2.create_dataset('label', data=np.array(signal_type, dtype=np.int32))

        #save data as h5 file
        #dset1 = f5.create_dataset("dataset"+ "_" + str(i), dtype='i', data=data)
        #dset2 = f6.create_dataset(header_text+ "_" + str(i), dtype='i', data=param)

    f5.close()
    f6.close()

def generate_mono_shower_1():
    os.makedirs(training_data_dir + "mono_shower_1", exist_ok=True)
    #os.makedirs(training_data_dir + "Mono_shower_1_param", exist_ok=True)
    h5_file_name = training_data_dir + "mono_shower_1" + "_raw.hdf5" #%(i, i)
    f7 = h5py.File(h5_file_name, "w")
    h5_param_file_name = training_data_dir + "mono_shower_1" + "_param_raw.hdf5" #%(i, i)
    f8 = h5py.File(h5_param_file_name, "w")

    signal_type = label[3]
    
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

        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]

        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        mono_1_data = image_1 + nsb_level

        mono_1 = np.array(mono_1_data, dtype=np.float32)
        mono_1_array = np.expand_dims(mono_1, axis=0)
        mono_1_label = np.array(signal_type, dtype=np.int32)
        mono_1_label_array = np.expand_dims(mono_1_label, axis=0)

        group_name_1 = 'mono_shower_1'+str(i+1) #image name
        grp1 = f7.create_group(group_name_1)

        group_name_2 = 'mono_shower_1_param'+str(i+1) 
        grp2 = f8.create_group(group_name_2)        

        grp1.create_dataset('data', data=mono_1_array)
        grp1.create_dataset('label', data=mono_1_label_array)   #signal_type = label[0]

        grp2.create_dataset('data', data=np.array(param, dtype=np.float32))
        grp2.create_dataset('label', data=np.array(signal_type, dtype=np.int32))
        
        #save data as h5 file
        #dset1 = f7.create_dataset("dataset"+ "_" + str(i), dtype='i', data=data)
        #dset2 = f8.create_dataset(header_text+ "_" + str(i), dtype='i', data=param)

    f7.close()
    f8.close()

def generate_mono_shower_2():
    os.makedirs(training_data_dir + "Mono_shower_2", exist_ok=True)
    #os.makedirs(training_data_dir + "Mono_shower_2_param", exist_ok=True)
    h5_file_name = training_data_dir + "mono_shower_2" + "_raw.hdf5" #%(i, i)
    f9 = h5py.File(h5_file_name, "w")
    h5_param_file_name = training_data_dir + "mono_shower_2" + "_param_raw.hdf5" #%(i, i)
    f10 = h5py.File(h5_param_file_name, "w")
    
    signal_type = label[4]

    for i in tqdm(range(number_of_data_points)):
    
        x_1 = random.randint(5,15)/100.
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

        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]

        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        mono_2_data = image_1 + nsb_level

        mono_2 = np.array(mono_2_data, dtype=np.float32)
        mono_2_array = np.expand_dims(mono_2, axis=0)
        mono_2_label = np.array(signal_type, dtype=np.int32)
        mono_2_label_array = np.expand_dims(mono_2_label, axis=0)
        
        group_name_1 = 'mono_shower_2_'+str(i+1) #image name
        grp1 = f9.create_group(group_name_1)

        group_name_2 = 'mono_shower_2_param'+str(i+1) 
        grp2 = f10.create_group(group_name_2)        


        grp1.create_dataset('data', data=mono_2_array)
        grp1.create_dataset('label', data=mono_2_label_array)   #signal_type = label[0]

        grp2.create_dataset('data', data=np.array(param, dtype=np.float32))
        grp2.create_dataset('label', data=np.array(signal_type, dtype=np.int32))

        #save data as h5 file
        #dset1 = f9.create_dataset("dataset"+ "_" + str(i), dtype='i', data=data)
        #dset2 = f10.create_dataset(header_text+ "_" + str(i), dtype='i', data=param)

    f9.close()
    f10.close()

def generate_mono_shower_3():
    os.makedirs(training_data_dir + "Mono_shower_3", exist_ok=True)
    #os.makedirs(training_data_dir + "Mono_shower_3_param", exist_ok=True)
    h5_file_name = training_data_dir + "mono_shower_3" + "_raw.hdf5" #%(i, i)
    f11 = h5py.File(h5_file_name, "w")
    h5_param_file_name = training_data_dir + "mono_shower_3" + "_param_raw.hdf5" #%(i, i)
    f12 = h5py.File(h5_param_file_name, "w")

    signal_type = label[5]
    
    for i in tqdm(range(number_of_data_points)):
    
        x_1 = random.randint(5,15)/100.
        y_1 = random.randint(5,15)/100.
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

        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]

        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        mono_3_data = image_1 + nsb_level

        mono_3 = np.array(mono_3_data, dtype=np.float32)
        mono_3_array = np.expand_dims(mono_3, axis=0)
        mono_3_label = np.array(signal_type, dtype=np.int32)
        mono_3_label_array = np.expand_dims(mono_3_label, axis=0)

        group_name_1 = 'mono_shower_3'+str(i+1) #image name
        grp1 = f11.create_group(group_name_1)

        group_name_2 = 'mono_shower_3_param'+str(i+1) 
        grp2 = f12.create_group(group_name_2)        


        grp1.create_dataset('data', data=mono_3_array)
        grp1.create_dataset('label', data=mono_3_label_array)   #signal_type = label[0]

        grp2.create_dataset('data', data=np.array(param, dtype=np.float32))
        grp2.create_dataset('label', data=np.array(signal_type, dtype=np.int32))
        
        #save data as h5 file
        #dset1 = f11.create_dataset("dataset"+ "_" + str(i), dtype='i', data=data)
        #dset2 = f12.create_dataset(header_text+ "_" + str(i), dtype='i', data=param)

    f11.close()
    f12.close()       

def generate_mono_shower_4():
    os.makedirs(training_data_dir + "Mono_shower_4", exist_ok=True)
    #os.makedirs(training_data_dir + "Mono_shower_4_param", exist_ok=True)
    h5_file_name = training_data_dir + "mono_shower_4" + "_raw.hdf5" #%(i, i)
    f13 = h5py.File(h5_file_name, "w")
    h5_param_file_name = training_data_dir + "mono_shower_4" + "_param_raw.hdf5" #%(i, i)
    f14 = h5py.File(h5_param_file_name, "w")

    signal_type = label[6]
    
    for i in tqdm(range(number_of_data_points)):
    
        x_1 = random.randint(5,15)/100.
        y_1 = random.randint(5,15)/100.
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

        header_text = "x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level"
        param = [x_1,y_1,width_1,length_1,psi_start,skew_1,intensity_1,nsb_level]

        camgeom = CameraGeometry.from_name(camera_type)
        image_1, *_ = model_1.generate_image(camgeom, intensity=intensity_1, nsb_level_pe=nsb_level,)

        mono_4_data = image_1 + nsb_level

        mono_4 = np.array(mono_4_data, dtype=np.float32)
        mono_4_array = np.expand_dims(mono_4, axis=0)
        mono_4_label = np.array(signal_type, dtype=np.int32)
        mono_4_label_array = np.expand_dims(mono_4_label, axis=0)
        
        group_name_1 = 'mono_shower_4_'+str(i+1) #image name
        grp1 = f13.create_group(group_name_1)

        group_name_2 = 'mono_shower_4_param'+str(i+1) 
        grp2 = f14.create_group(group_name_2)        


        grp1.create_dataset('data', data=mono_4_array)
        grp1.create_dataset('label', data=mono_4_label_array)   #signal_type = label[0]

        grp2.create_dataset('data', data=np.array(param, dtype=np.float32))
        grp2.create_dataset('label', data=np.array(signal_type, dtype=np.int32))

        #save data as h5 file
        #dset1 = f13.create_dataset("dataset"+ "_" + str(i), dtype='i', data=data)
        #dset2 = f14.create_dataset(header_text+ "_" + str(i), dtype='i', data=param)

    f13.close()
    f14.close()   

def main():
    time_start = time.time()
    #generate_noise()
    #generate_laser()
    #generate_muon()
    #generate_mono_shower_1()
    generate_mono_shower_2()
    generate_mono_shower_3()
    generate_mono_shower_4()
    
    time_end = time.time()
    print(time_end - time_start)

if __name__ == "__main__":
    main()
