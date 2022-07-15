# summer2022
scripts for biGAN project at the BL REU

Project Description:

Short laser bursts are a form of emission that we generally don't see as a result of natural cosmic phenomena. If we were to observe a short laser burst coming from space, we can assume that it emanated from some form of extraterrestrial technology. Short laser burst technosignatures are thus a useful candidate to look for when conducting SETI investigations. 

Imaging Atmospheric Cherenkov Telescopes (IACTs) are very useful when detecting short laser bursts. Though IACTs are designed as high-energy astrophysical instruments, they also pick up optical laser signals and cosmic rays. These laser signals tend to be overwhelmed with noise, so some analysis is necessary in order to extract signals. There are a few different algorithms and techniques that can be used to process these signals, but unfortunately these methods are computationally expensive and time consuming. 

One solution to this issue comes in the form of GANs (Generative Adversarial Networks). GANs are a machine learning technique that take a real dataset and create realistic synthetic data, then use a discriminator algorithm to figure out which signals are real and which are fake. Since we don't have enough data on short laser burst techosignatures to train a machine learning algorithm to find them, it makes sense to synthesize some for algorithm training purposes. 

The goal of this project is to train a discriminator algorithm to successfully find short laser burst technosignatures among noise and fake signals. The scripts in this repository are mostly from John Hoang, to which I have made various changes. 
