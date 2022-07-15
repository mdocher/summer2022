from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import numpy as np
from numpy import reshape
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import time


import os, glob
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy import savetxt
from numpy.random import randn
from numpy.random import randint

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
sns.set_palette('colorblind')
plt.rcParams.update({'font.size': 24})
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["figure.figsize"] = (15,15)

n_signals_per_batch = 2500
number_of_steps = 5
number_of_classes = 7
n_timesteps = 11328
latent_dim = 100
loss_type = 'mean_squared_error' # loss: can be binary_crossentropy


Quad_0_Dir = "/home/kdocher/code/SCTCam//Noise"
Quad_1_Dir = "/home/kdocher/code/SCTCam/Mono_shower_1"
Quad_2_Dir = "/home/kdocher/code/SCTCam/Mono_shower_2"
Quad_3_Dir = "/home/kdocher/code/SCTCam/Mono_shower_3"
Quad_4_Dir = "/home/kdocher/code/SCTCam/Mono_shower_4"
Quad_5_Dir = "/home/kdocher/code/SCTCam/Quad_laser"
Quad_6_Dir = "/home/kdocher/code/SCTCam/Muon"

############
## Models ##
############
def define_generator(epoch, latent_dim, n_classes=number_of_classes):
    gen_model_dir = "models/generator_model%04d.h5" %(epoch)
    model = load_model(gen_model_dir)
    return model
    
def define_discriminator(epoch, in_shape=n_timesteps,n_classes=number_of_classes):
    dis_model_dir = "models/discriminator_model%04d.h5" %(epoch)
    model = load_model(dis_model_dir)
    return model

def define_gan(epoch, g_model, d_model):
    gan_model_dir = "models/gan_model%04d.h5" %(epoch)
    model = load_model(gan_model_dir)
    return model



def plot_TSNE_epoch(epoch):
    ##########################
    # Quad_0: Poisson noise. #
    ##########################
    print("Loading quad_0")
    quad_0 = [[] for i in range(n_signals_per_batch)]
    list_of_txt_files_clean = glob.glob(Quad_0_Dir + "/*.txt")
    for i in tqdm(range(n_signals_per_batch)):
        pixel_charge = np.loadtxt(list_of_txt_files_clean[i], delimiter = "\n")
        quad_0[i] = pixel_charge
    quad_0 = np.asarray(quad_0)
    quad_0_real = quad_0
    label_quad_0_real =  np.ones((n_signals_per_batch))*0

    #########################
    # Quad_1: 1st Quadrant. #
    #########################
    print("Loading quad_1")
    quad_1 = [[] for i in range(n_signals_per_batch)]
    list_of_txt_files_clean = glob.glob(Quad_1_Dir + "/*.txt")
    for i in tqdm(range(n_signals_per_batch)):
        pixel_charge = np.loadtxt(list_of_txt_files_clean[i], delimiter = "\n")
        quad_1[i] = pixel_charge
    quad_1 = np.asarray(quad_1)
    quad_1_real = quad_1
    label_quad_1_real =  np.ones((n_signals_per_batch))*1
        
    #########################
    # Quad_2: 2nd Quadrant. #
    #########################
    print("Loading quad_2")
    quad_2 = [[] for i in range(n_signals_per_batch)]
    list_of_txt_files_clean = glob.glob(Quad_2_Dir + "/*.txt")
    for i in tqdm(range(n_signals_per_batch)):
        pixel_charge = np.loadtxt(list_of_txt_files_clean[i], delimiter = "\n")
        quad_2[i] = pixel_charge
    quad_2 = np.asarray(quad_2)
    quad_2_real = quad_2
    label_quad_2_real =  np.ones((n_signals_per_batch))*2
        
    #########################
    # Quad_3: 3rd Quadrant. #
    #########################
    print("Loading quad_3")
    quad_3 = [[] for i in range(n_signals_per_batch)]
    list_of_txt_files_clean = glob.glob(Quad_3_Dir + "/*.txt")
    for i in tqdm(range(n_signals_per_batch)):
        pixel_charge = np.loadtxt(list_of_txt_files_clean[i], delimiter = "\n")
        quad_3[i] = pixel_charge
    quad_3 = np.asarray(quad_3)
    quad_3_real = quad_3
    label_quad_3_real =  np.ones((n_signals_per_batch))*3
    
    #########################
    # Quad_4: 4th Quadrant. #
    #########################
    print("Loading quad_4")
    quad_4 = [[] for i in range(n_signals_per_batch)]
    list_of_txt_files_clean = glob.glob(Quad_4_Dir + "/*.txt")
    for i in tqdm(range(n_signals_per_batch)):
        pixel_charge = np.loadtxt(list_of_txt_files_clean[i], delimiter = "\n")
        quad_4[i] = pixel_charge
    quad_4 = np.asarray(quad_4)
    quad_4_real = quad_4
    label_quad_4_real =  np.ones((n_signals_per_batch))*4
     

    #########################
    # Quad_5: Laser.        #
    #########################
    print("Loading laser")
    quad_5 = [[] for i in range(n_signals_per_batch)]
    list_of_txt_files_clean = glob.glob(Quad_5_Dir + "/*.txt")
    for i in tqdm(range(n_signals_per_batch)):
        pixel_charge = np.loadtxt(list_of_txt_files_clean[i], delimiter = "\n")
        quad_5[i] = pixel_charge
    quad_5 = np.asarray(quad_5)
    quad_5_real = quad_5
    label_quad_5_real =  np.ones((n_signals_per_batch))*5

    #########################
    # Quad_6: Muon.         #
    #########################
    print("Loading muon")
    quad_6 = [[] for i in range(n_signals_per_batch)]
    list_of_txt_files_clean = glob.glob(Quad_6_Dir + "/*.txt")
    for i in tqdm(range(n_signals_per_batch)):
        pixel_charge = np.loadtxt(list_of_txt_files_clean[i], delimiter = "\n")
        quad_6[i] = pixel_charge
    quad_6 = np.asarray(quad_6)
    quad_6_real = quad_6
    label_quad_6_real =  np.ones((n_signals_per_batch))*6

     
    
    net_data = np.concatenate((quad_0_real, quad_1_real,quad_2_real,quad_3_real,quad_4_real,quad_5_real,quad_6_real), axis = 0)
    net_data_label = np.concatenate((label_quad_0_real, label_quad_1_real,label_quad_2_real,label_quad_3_real,label_quad_4_real,label_quad_5_real,label_quad_6_real), axis = 0)

    #print(np.shape(net_data))
    
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(net_data)
    
    #########################
    # PLOTTING              #
    #########################
    
    df = pd.DataFrame()
    
    #Data process: 0 is MC, 1 is cGAN, only used for plotting
    number_of_classes = 7
    net_label_MC = np.repeat("MC", n_signals_per_batch * number_of_classes)
    #net_label_GAN = np.repeat("ML", n_signals_per_batch * number_of_classes)
    net_label_process = np.concatenate((net_label_MC), axis = None)

    data_label = ["Noise","Quad_1","Quad_2","Quad_3","Quad_4","Laser","Muon"]
    net_data_label = np.empty(n_signals_per_batch * number_of_classes, dtype=np.dtype('U100'))
    for i in range(number_of_classes):
        net_data_label[i*n_signals_per_batch:(i+1)*n_signals_per_batch] = data_label[i]
    #net_data_label=np.tile(net_data_label,2)
    
    df["Process"] = net_label_process
    df["Label"] = net_data_label
    df["tSNE comp-1"] = z[:,0]
    df["tSNE comp-2"] = z[:,1]

    sns.scatterplot(x="tSNE comp-1", y="tSNE comp-2", hue=df.Label, style=df.Process,  palette=sns.color_palette("hls", 7), data=df, legend="full")
    plt.title("SCTCam Cherenkov data T-SNE projection")
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    #plt.show()
    plt.savefig('SCTCam_comparison_tsne_projection_raw_temp_4_%04d.png' %(epoch))
    plt.close()
        

def main():
    time_start = time.time()
    plot_TSNE_epoch(2500)
    time_end = time.time()
    print("Runtime: ", time_end - time_start)
		
if __name__ == "__main__":
	main()

