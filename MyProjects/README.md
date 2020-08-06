# My Projects List

Hello there! I have listed here the projects I have made and maintaining. This project work is a part of many courses and specializations I have completed on Coursera and Udacity. The list of the MOOC courses/specailizations I have completed can be found [here](MOOCs). Here, I present a short description of every project I have worked on.

### Facial Composites

Facial composites are widely used in forensics to generate images of suspects. Since victim or witness usually isn’t good at drawing, computer-aided generation is applied to reconstruct the face attacker. One of the most commonly used techniques is evolutionary systems that compose the final face from many predefined parts.

In this project, I implement an app for creating a facial composite that will be able to construct desired faces without explicitly providing databases of templates. I apply Variational Autoencoders, Gaussian processes, and Bayesian Optimization for this task.

[See Project](Facial%20Composites.ipynb)

### Gaussian Processes using GPy and GPyOpt

I have created Gaussian Process Regressor and a Sparse GP model using GPy and GPyOpt API.

[See Project](Gaussian%20Processes%20using%20GPy%20and%20GpyOpt.ipynb)

### Image Captioning

I have used TensorFlow's pretrained InceptionNet as a encoder network and dynamic stacked LSTM as a decoder network. All the images provided as an input are converted to vectors of a fixed length and this vectors are a input to the one-to-many LSTM that outputs a english caption for the image. I have used finetuning using Keras for the task for about 30 epochs. It produces reasonable captions while testing on internet images. You can also test an image by uploading it to the internet and running last couple of cells of the notebook.

[See Project](Image%20Captioning.ipynb)

### Markov Chain Monte Carlo using PyMC3

I have implemented basic bayesian machine learnings models using PyMC3 and infered using MCMC Sampling.

[See Project](Markov%20Chain%20Monte%20Corlo%20using%20PyMc3.ipynb)

### Variational Auto Encoder on MNIST

In this project, I have implemented a Variational Auto Encoder using TensorFlow that can generate MNIST like images of hand-written images. This is an excellent demonstration of a bayesian generative models.

[See Project](Variational%20Auto%20Encoder%20on%20MNIST.ipynb)

### GMM using EM Algorithm

Gaussian Mixture Models have been very well established in unsupervised machine learning tasks. In the project, I have implemented an Expectation Maximization algorithm from scratch in pure numpy. This algorithm is used to train a GMM and perform clustering tasks on images or other datasets.

[See Project](https://github.com/tirthasheshpatel/swissroll/blob/ebdce72c718ec3dd6cb065e60f4cfd5ff4256df6/swissroll/gmm.py)