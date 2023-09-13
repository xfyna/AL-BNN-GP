# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 08:39:56 2022

@author: frank
"""
"""Backend supported: tensorflow.compat.v1, tensorflow"""
import deepxde as dde
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.ticker import FormatStrFormatter
import math
import torch
import gpytorch
import sklearn.metrics
import time
import os
import glob
from osgeo import gdal
import pickle
import re
import cv2
import sys
import warnings
from numpy.linalg import inv

func_name = sys.argv[1]
noise_name = sys.argv[2]
explore_name = sys.argv[3]
dx = int(sys.argv[4])

# clear files in model directory
files = glob.glob('../model/*')
for f in files:
    os.remove(f)
    
#%% functions
# define the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):    
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
    
        if kernel_type == 'RBF':
        # RBF Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = 2)) 
        elif kernel_type == 'Matern':
        # Matern Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims = 2))
        elif kernel_type == 'Periodic':
        # Periodic Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())
        elif kernel_type == 'Piece_Polynomial':
        # Piecewise Polynomial Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel(ard_num_dims = 2))
        elif kernel_type == 'RQ':
        # RQ Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel(ard_num_dims = 2))
        elif kernel_type == 'Cosine': # !
        # Cosine Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel(ard_num_dims = 2))
        elif kernel_type == 'Linear':
        # Linear Kernel
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(ard_num_dims = 2))
        elif kernel_type == 'Polynomial':
        # Polynomial Kernel 
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(ard_num_dims = 2, power = 4))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
 
def kernel_print(i, training_iter, loss, model):
    if kernel_type == 'RBF' or kernel_type == 'Matern' or kernel_type == 'Piece_Polynomial':
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
                model.likelihood.noise.detach().numpy()
            ))
    elif kernel_type == 'Periodic': 
    # Periodic Kernel
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f  period_length: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy(),
                model.covar_module.base_kernel.period_length.detach().numpy()
            ))
    elif kernel_type == 'RQ':
    # RQ Kernel
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f %.3f   alpha: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][0],
                model.covar_module.base_kernel.lengthscale.detach().numpy()[0][1],
                model.covar_module.base_kernel.alpha.detach().numpy()
            )) 
    elif kernel_type == 'Cosine': # !
    # Cosine Kernel
        print('Iter %d/%d - Loss: %.3f   period_length: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.period_length.detach().numpy()
            ))
    elif kernel_type == 'Linear':
    # Linear Kernel
        print('Iter %d/%d - Loss: %.3f   variance: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.variance.item()
            ))
    elif kernel_type == 'Polynomial':
    # Polynomial Kernel 
        print('Iter %d/%d - Loss: %.3f   offset: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.offset.detach().numpy()
            ))
   
def GPtrain(x_train, y_train, training_iter):
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # train GP model
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        # kernel_print(i, training_iter, loss, model)
        optimizer.step()
    
    return likelihood, model, optimizer, output, loss

def GPeval(x_test, model, likelihood):
    
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.from_numpy(x_test).float()
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))
    
    f_preds = model(x_test)
    y_preds = likelihood(model(x_test))
    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_var = np.diag(f_preds.lazy_covariance_matrix.numpy())
    f_covar = f_preds.covariance_matrix
    
    with torch.no_grad():
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        
    return observed_pred, lower, upper

def RMS(y_1,y_2):
    return math.sqrt(sklearn.metrics.mean_squared_error(y_1, y_2))

def sample_disp_con(x,x_start,r_disp):
    # x_start = x[i_start,:]
    if func_name == 'parabola' or 'townsend':
        x_disp = np.sqrt((x[:,0]-x_start[0])**2 + (x[:,1]-x_start[1])**2) #+ (x[:,2]-x_start[2])**2)
        i_con = np.argwhere(x_disp<=r_disp)
        i_con = np.sort(i_con)
    elif func_name == 'lunar':
        x_disp = np.sqrt((x[:,0]-x_start[0])**2 + (x[:,1]-x_start[1])**2 + (x[:,2]-x_start[2])**2)
        i_con = np.argwhere(x_disp<=r_disp)
        i_con = np.sort(i_con)
    return list(i_con[:,0])

def RKHS_norm(y,sigma,K):
    n_row, n_col = K.shape
    alpha = inv(K + sigma**2 * np.eye(n_row)) @ y
    return alpha.transpose() @ K @ alpha

#%% plotting
def plotBoth():
    x_train = train_state.X_train.detach().numpy()
    x_test = train_state.X_test
    y_train = train_state.y_train
    y_test = train_state.y_test
    y_pred = train_state.y_pred_test
    ystd_pred = train_state.y_std_test
    residual = y_test[:] - y_pred[:]
    BNN_model.data.train_x = X_train
    BNN_model.data.train_y = y_train.reshape((len(y_train),1))
    
    # BNN traverse path
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    rover = ax1.scatter3D(BNN_model.data.train_x[-1, 0], BNN_model.data.train_x[-1, 1],
                          BNN_model.data.train_y[-1], s=100, color='yellow', marker='*', zorder=1)
    rover_path = ax1.plot3D(BNN_model.data.train_x[:,0], BNN_model.data.train_x[:,1], BNN_model.data.train_y[:,0], color='black')
    test_surf = ax1.plot_trisurf( x_test[:, 0], x_test[:, 1], y_pred[:,0], color='grey', alpha=0.25)
    true_surf = ax1.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno',
                            linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$y")
    ax1.set_title('rover on surface '+str(x_true[i_train, :]))
    
    # BNN uncertainty
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    ax2.set_title('B-NN uncertainty')
    std_pred = ax2.plot_trisurf( x_test[:, 0], x_test[:, 1], ystd_pred[:,0], color='grey', alpha=0.25)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_zlabel("std(y)")
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.e'))
    
    # BNN residual
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    ax3.set_title('B-NN residual')
    res_pred = ax3.plot_trisurf( x_test[:, 0], x_test[:, 1], residual[:,0], color='grey', alpha=0.25)
    ax3.set_xlabel("$x_1$")
    ax3.set_ylabel("$x_2$")
    ax3.set_zlabel("err(y)")
    ax3.zaxis.set_major_formatter(FormatStrFormatter('%.e'))
    
    # GP traverse path
    ax4 = fig.add_subplot(3, 3, 4, projection='3d')
    ax4.set_title('GP prediction vs truth')
    ax4.scatter3D(BNN_model.data.train_x[-1, 0], BNN_model.data.train_x[-1, 1],
                          BNN_model.data.train_y[-1], s=100, color='yellow', marker='*', zorder=1)
    train_points = ax4.plot3D( x_train[:, 0], x_train[:, 1], y_train[:, 0], color='black')
    test_surf = ax4.plot_trisurf( x_test[:, 0], x_test[:, 1], test_pred.mean.numpy(), color='grey', alpha=0.25)
    true_surf = ax4.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno',
                            linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))
    ax4.set_xlabel("$x_1$")
    ax4.set_ylabel("$x_2$")
    ax4.set_zlabel("$y")
    
    # GP uncertainty
    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    ax5.set_title('GP uncertainty')
    std_pred = ax5.plot_trisurf( x_test[:, 0], x_test[:, 1], upper_global - lower_global, color='grey', alpha=0.25)
    ax5.set_xlabel("$x_1$")
    ax5.set_ylabel("$x_2$")
    ax5.set_zlabel("std(y)")
    
    # GP residual
    ax6 = fig.add_subplot(3, 3, 6, projection='3d')
    ax6.set_title('GP residual')
    res_pred = ax6.plot_trisurf( x_test[:, 0], x_test[:, 1], test_pred.mean.numpy()-y_test[:,0], color='grey', alpha=0.25)
    ax6.set_xlabel("$x_1$")
    ax6.set_ylabel("$x_2$")
    ax6.set_zlabel("err(y)")
    
    # RMS error comparison
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.set_title('RMS error')
    BNN_RMS = ax7.plot(range(0, len(y_RMS_BNN)), y_RMS_BNN, color = 'black', marker='.')
    GP_RMS = ax7.plot(range(0, len(y_RMS_GP)), y_RMS_GP, color = 'blue', marker ='.')
    ax7.set_ylabel('RMS error')
    ax7.set_xlabel('number of samples')
    ax7.legend(['BNN', 'GP'], loc='lower right')
    
    # variance comparison
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.set_title('variance')
    BNN_RMS = ax8.plot(range(0, len(std_BNN)), std_BNN, color = 'black', marker='.')
    GP_RMS = ax8.plot(range(0, len(std_GP)), std_GP, color = 'blue', marker ='.')
    ax8.set_ylabel('variance')
    ax8.set_xlabel('number of samples')
    ax8.legend(['BNN', 'GP'], loc='lower right')
    
    # variance comparison
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_title('RKHS norm')
    BNN_RMS = ax9.plot(range(0, len(f2_H_BNN)), f2_H_BNN, color = 'black', marker='.')
    GP_RMS = ax9.plot(range(0, len(f2_H_GP)), f2_H_GP, color = 'blue', marker ='.')
    ax9.set_ylabel('variance')
    ax9.set_xlabel('number of samples')
    ax9.legend(['BNN', 'GP'], loc='lower right')
    
    
    return ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9

def nearest_neighbor(x_sample,x_list,y_list):
    disp = np.sqrt((x_sample[0]-x_list[:,0])**2+(x_sample[1]-x_list[:,1])**2)
    i_min = np.argmin(disp)
    y_sample = y_list[i_min]
    return y_sample

#%% surfaces
#let's create a 2D convex surface 
if func_name == 'parabola':
    grid_bounds = [(-1, 1), (-1, 1)]
    grid_size = 21
    x_1 = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_size)
    x_2 = x_1
    grid_diff = x_1[1] - x_1[0]
    x_plot, y_plot = np.meshgrid(x_1, x_2)
    z_plot = x_plot**2 + y_plot**2
    x_vec = np.array([x_plot.ravel()])
    y_vec = np.array([y_plot.ravel()])
    x_true = np.concatenate((x_vec.transpose(),y_vec.transpose()), axis=1)
    y_true = x_true[:, 0]**2 + x_true[:, 1]**2 
    n_samples = len(y_true)
    sigma = math.sqrt(0.02)
    y_obs = y_true
    if noise_name == 'noise':
        y_obs = y_true + np.random.rand(n_samples) * sigma
    
# and now a nonconvex surface, townsend function (https://en.wikipedia.org/w/index.php?title=Test_functions_for_optimization&oldid=787014841)
if func_name == 'townsend':
    grid_bounds = [(-1.75, 1.75), (-1.75, 1.75)]
    grid_size = 21
    x_1 = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_size)
    x_2 = x_1
    grid_diff = x_1[1] - x_1[0]
    x_plot, y_plot = np.meshgrid(x_1, x_2)
    z_plot = -(np.cos((x_plot-0.1)*y_plot))**2-x_plot*np.sin(3*x_plot+y_plot)
    x_vec = np.array([x_plot.ravel()])
    y_vec = np.array([y_plot.ravel()])
    x_true = np.concatenate((x_vec.transpose(),y_vec.transpose()), axis=1)
    y_true = -(np.cos((x_true[:,0]-0.1)*x_true[:,1]))**2-x_true[:,0]*np.sin(3*x_true[:,0]+x_true[:,1])
    n_samples = len(y_true)
    sigma = math.sqrt(0.02)
    y_obs = y_true
    if noise_name == 'noise':
        y_obs = y_true + np.random.rand(n_samples) * sigma
# # min resides at (2.5510, 0.0258)

elif func_name == 'lunar':
    grid_bounds = [(-1.5, 1.5), (-1.5, 1.5)]
    grid_size = 35
    file_name1 = 'Shoemaker_5mDEM.tif'
    # You need to multiply 0.5 for each pixel value to get the actual elevation.
    Aimg = gdal.Open(file_name1)
    A = Aimg.GetRasterBand(1).ReadAsArray()

    file_name2 = 'Shoemaker_280mIceExposures.tif'
    Bimg = gdal.Open(file_name2)
    B = Bimg.GetRasterBand(1).ReadAsArray()

    file_name3 = 'Shoemaker_250mLAMP-OnOffRatio.tif'
    Cimg = gdal.Open(file_name3)
    C = Cimg.GetRasterBand(1).ReadAsArray()

    # make DEMs and other maps
    # to build a DEM, each index in row and column is 5 m
    (n_y,n_x) = np.shape(A)
    spacing = 5.0
    x_vec_grid5 = np.array(range(n_x))*spacing
    y_vec_grid5 = np.array(range(n_y))*spacing
    x_mat5, y_mat5 = np.meshgrid(x_vec_grid5, y_vec_grid5)
    z_mat5 = A/2
    z_mat5 = np.where(z_mat5==32767/2, np.nan, z_mat5) 
    z_min5 = min(z_mat5[~np.isnan(z_mat5)])
    z_max5 = max(z_mat5[~np.isnan(z_mat5)])
    grid_diff = 0.25

    # unravel grid data
    x_DEM5 = x_mat5.ravel()
    y_DEM5 = y_mat5.ravel()
    z_DEM5 = z_mat5.ravel()

    #  parse ice data distance 280 m
    (n_y,n_x) = np.shape(B)
    spacing = 280.0
    x_vec_grid280 = np.array(range(n_x))*spacing
    y_vec_grid280 = np.array(range(n_y))*spacing
    x_mat280, y_mat280 = np.meshgrid(x_vec_grid280, y_vec_grid280)
    z_mat280 = z_mat5[::56,::56]
    z_mat280 = z_mat280[0:n_y,0:n_x]

    # unravel grid data
    x_DEM280 = x_mat280.ravel()
    y_DEM280 = y_mat280.ravel()
    z_DEM280 = z_mat280.ravel()
    ice_DEM280 = B.ravel()

    #  parse LAMP data distance 250m
    (n_y,n_x) = np.shape(C)
    spacing = 250.0
    x_vec_grid250 = np.array(range(n_x))*spacing
    y_vec_grid250 = np.array(range(n_y))*spacing
    x_mat250, y_mat250 = np.meshgrid(x_vec_grid250, y_vec_grid250)
    z_mat250 = z_mat5[::50,::50]
    # unravel grid data
    x_DEM250 = x_mat250.ravel()
    y_DEM250 = y_mat250.ravel()
    z_DEM250 = z_mat250.ravel()

    C = np.where(C==-9999, np.nan, C) 
    c_min = min(C[~np.isnan(C)])
    c_max = max(C[~np.isnan(C)])
    c_DEM250 = C.ravel()
    # let's make LAMP data the elevation
    LAMP_DEM280 = np.zeros(len(x_DEM280))
    x_list = np.array([x_DEM250,y_DEM250]).transpose()
    for i in range(len(x_DEM280)):
        x_sample = np.array([x_DEM280[i],y_DEM280[i]])
        LAMP_DEM280[i] = nearest_neighbor(x_sample,x_list,c_DEM250)
    # % clean up data
    # training data input is DEM position 
    x_true = np.array([x_DEM250/1000, y_DEM250/1000, z_DEM250/1000]).transpose()
    # training data output is LAMP
    y_obs =  np.double(c_DEM250)

    # get rid of elevation nan values
    y_obs =  y_obs[~np.isnan(x_true[:,2])]
    x_true = x_true[~np.isnan(x_true[:,2]),:]
    # get rid of LAMP data
    x_true = x_true[~np.isnan(y_obs),:]
    y_obs =  y_obs[~np.isnan(y_obs)]

    x_true_doub = x_true
    y_obs_doub = y_obs

    for i in range(x_true.shape[0]):
        y_obs_doub[i] = np.float64(y_obs[i])
        for j in range(x_true.shape[1]):
            x_true_doub[i, j] = np.float64(x_true[i, j])

    x_center_all = np.mean(x_true,0)
    x_disp = np.sqrt((x_true[:,0]-x_center_all[0])**2 + (x_true[:,1]-x_center_all[1])**2 + (x_true[:,2]-x_center_all[2])**2)
    i_min = np.argmin(x_disp)
    x_center = x_true[i_min,:]
    
    x_true = x_true_doub - x_center
    
    r = 6
    
    y_obs = y_obs[np.argwhere(x_true[:,0]>=-r/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,0]>=-r/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,1]>=-r/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,1]>=-r/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,0]<=r/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,0]<=r/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,1]<=r/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,1]<=r/2)[:,0]]
    
    y_true = y_obs #x_true[:,2] NOPE - DON'T DO THIS LOL
    x_true_2 = x_true
    x_true = x_true[:, :2]
    n_samples = len(y_obs)
    sigma = math.sqrt(0.02)

#explore_name = 'spiral' #'snake' or 'GPAL'
r_NN = np.sqrt(3)*grid_diff
r_con = 3*r_NN
#dx = 2

var_iter = []
var_iter_local = []
var_iter_global = []
rmse_local_obs = []
rmse_global_obs = []
rmse_local_true = []
rmse_global_true = []

trial_name = str(func_name)+'_'+str(noise_name)+'_'+str(explore_name)+'_'+str(dx)+'dx'
parent_dir = '../GPAL'
image_path = os.path.join(parent_dir, trial_name + '/')
os.mkdir(image_path)

stdoutOrigin=sys.stdout 
sys.stdout = open(image_path+"log.txt", "w")

#%% different sampling strategies
def snake(x_in, y_in, dx):
    i_train = []
    x1_min = np.min(x_in[:,0])
    x2_min = np.min(x_in[:,1])
    x1_max = np.max(x_in[:,0])
    x2_max = np.max(x_in[:,1])
    
    n_samples = len(y_in)
    i_seq = list(range(0,n_samples))
    
    x1_all = np.unique(x_in[:,0])
    x2_all = np.unique(x_in[:,1])

    x1 = np.linspace(x1_min, x1_max, int((x1_all.size-1)/dx+1))
    x2_plus = np.linspace(x2_min, x2_max, int((x2_all.size-1)/dx+1))
    x2_minus = np.linspace(x2_max, x2_min, int((x2_all.size-1)/dx+1))
    
    x2_status = 'forward'
    for x1_i in x1:
        i_1 = np.argwhere(x_in[:,0]==x1_i)[:,0]
        if x2_status == 'forward':
            x2 = x2_plus
            x2_status = 'reverse'
        else:
            x2 = x2_minus
            x2_status = 'forward'
        for x2_j in x2:
            i_2 = np.argwhere(x_in[:,1]==x2_j)[:,0]

            i_sample = np.argmin((x_in[:,0]-x1_i)**2 + (x_in[:,1]-x2_j)**2)
            # print(i_sample)
            # if not i_sample:
            #     continue
            i_train.append(i_sample)
            
    x_out = x_in[i_train]
    y_out = y_in[i_train]
    
    return x_out, y_out, i_train
        
def spiral(x_in, y_in, dx):
    
    i1_min = 0
    i1_max = len(np.unique(x_in[:,0]))-1
    i2_min = 0
    i2_max = len(np.unique(x_in[:,1]))-1
    
    r1_vec = np.linspace(np.min(x_in[:,0]),np.max(x_in[:,0]),int((i1_max)+1))
    r2_vec = np.linspace(np.min(x_in[:,1]),np.max(x_in[:,1]),int((i2_max)+1))

    i_train = []
    
    coord = np.array([],dtype='int')
    i2_right = np.arange(i2_min,i2_max+1,1)
    i1_right = i1_min*np.ones((len(i2_right),1))
    coord_right = np.hstack((i1_right,np.array([i2_right]).T))
    
    i1_down = np.arange(i1_min,i1_max,1)+1
    i2_down = i2_max*np.ones((len(i1_down),1))
    coord_down = np.hstack((np.array([i1_down]).T,i2_down))
    
    i2_left = np.arange(i2_max,i2_min,-1)-1
    i1_left = i1_max*np.ones((len(i2_left),1))
    coord_left = np.hstack((i1_left,np.array([i2_left]).T))
    i1_min = i1_min + 1
    
    i1_up = np.arange(i1_max,i1_min,-1)-1
    i2_up = i2_min*np.ones((len(i1_up),1))
    coord_up = np.hstack((np.array([i1_up]).T,i2_up))
    i2_max = i2_max - 1
    
    coord = np.vstack((coord_right, coord_down, coord_left, coord_up))
    
    while i1_min<=i1_max and i2_min<=i2_max:
        i2_right = np.arange(i2_min,i2_max,1)+1
        i1_right = i1_min*np.ones((len(i2_right),1))
        coord_right = np.hstack((i1_right,np.array([i2_right]).T))
        i1_max = i1_max - 1
        
        i1_down = np.arange(i1_min,i1_max,1)+1
        i2_down = i2_max*np.ones((len(i1_down),1))
        coord_down = np.hstack((np.array([i1_down]).T,i2_down))
        i2_min = i2_min + 1
        
        i2_left = np.arange(i2_max,i2_min,-1)-1
        i1_left = i1_max*np.ones((len(i2_left),1))
        coord_left = np.hstack((i1_left,np.array([i2_left]).T))
        i1_min = i1_min + 1
        
        i1_up = np.arange(i1_max,i1_min,-1)-1
        i2_up = i2_min*np.ones((len(i1_up),1))
        coord_up = np.hstack((np.array([i1_up]).T,i2_up))
        i2_max = i2_max - 1
        
        coord = np.vstack((coord, coord_right, coord_down, coord_left, coord_up))
    
    coord = np.array(coord,dtype='int')
    # print(range(len(coord)))
    for i in range(0,len(coord),dx):
        # print(i)
        # print(coord[i,0], coord[i,1])
        # print(r1_vec[coord[i,0]], r2_vec[coord[i,1]])
        i_sample = np.argmin((x_in[:,0]-r1_vec[coord[i,0]])**2 + (x_in[:,1]-r2_vec[coord[i,1]])**2)
        # print(i_sample)
        i_train.append(i_sample)
            
    x_out = x_in[np.flip(i_train)]
    y_out = y_in[np.flip(i_train)]
    return x_out, y_out, np.flip(i_train)

#%% iteratively train BNN and GP
if explore_name == 'snake':
    x_out, y_out, i_train_full = snake(x_true, y_obs, dx)
if explore_name == 'spiral':
    # what is this for? r_vec = np.linspace(grid_bounds[0][0],grid_bounds[0][1],int((grid_size-1)/dx+1))
    x_out, y_out, i_train_full = spiral(x_true, y_obs, dx)

print(explore_name)
print(i_train_full)

# create B-NN and train
i_test = set(range(0, n_samples-1)) - set(i_train_full)
X_train_full = x_true[list(i_train_full),:2]
y_train_full = y_true[list(i_train_full)]
X_test = x_true[list(i_test),:2]
y_test = y_true[list(i_test)]
i_train_BNN = []
i_train_GP = []

# initialize a small random dataset and train BNN with a large amount of iterations
n_start = 10
X_train = X_train_full[0:n_start,:]
y_train = y_train_full[0:n_start]

data = dde.data.DataSet(
    X_train,
    y_train.reshape((len(y_train),1)),
    X_test,
    y_test.reshape((len(y_test),1))
)

layer_size = [2] + [50] * 3 + [1]
activation = "sigmoid"
initializer = "Glorot uniform"
regularization = ["l2", 1e-5]
dropout_rate = 0.01
net = dde.nn.FNN(
    layer_size,
    activation,
    initializer,
    regularization,
    dropout_rate
)

BNN_model = dde.Model(data, net)
BNN_uncertainty = dde.callbacks.DropoutUncertainty(period=1000)
BNN_model.compile("adam", lr=0.001, metrics=["l2 relative error"])
checkpointer = dde.callbacks.ModelCheckpoint(
    filepath = "model/model.ckpt", 
    verbose = 0, 
    save_better_only = True, 
    period = 1000
)

train_time_BNN = []
BNN_start = time.time()
losshistory, train_state = BNN_model.train(iterations=10000, callbacks= [BNN_uncertainty, checkpointer])
BNN_end = time.time()
train_time_BNN.append(BNN_end - BNN_start)

y_pred = BNN_model.predict(X_test)

y_RMS_BNN = []
y_RMS_BNN.append(RMS(y_pred,y_test.reshape((len(y_test),1))))
std_BNN = []
std_BNN.append(np.mean(train_state.y_std_test[:,0]))
# print('y_RMS is %.3f' % (y_RMS_BNN[-1]) )

f2_H_BNN = []
K_test = train_state.y_std_test @ train_state.y_std_test.transpose()
sigma = math.sqrt(0.02)
f2_H_sample = RKHS_norm(y_pred,sigma,K_test)
f2_H_BNN.append(f2_H_sample[0,0])

# create GP and train
kernel_type = 'RBF'
training_iter = 100
x_train = torch.from_numpy(train_state.X_train)
y_train = torch.from_numpy(train_state.y_train[:,0])
x_train = x_train.float()
y_train = y_train.float()

x_test = torch.from_numpy(train_state.X_test)
y_test = torch.from_numpy(train_state.y_test[:,0])
x_test = x_test.float()
y_test = y_test.float()

# train model with GPyTorch model, which optimizes hyperparameters
train_time_GP = []
GP_start = time.time()
likelihood, model, optimizer, output, loss = GPtrain(x_train, y_train, training_iter)
GP_end = time.time()
train_time_GP.append(GP_end - GP_start)

train_pred, lower_train, upper_train = GPeval(x_train, model, likelihood)
rms_train = RMS(y_train, train_pred.mean.detach().numpy())
test_pred, lower_global, upper_global = GPeval(x_test, model, likelihood)

y_RMS_GP = []
y_RMS_GP.append(RMS(y_test, test_pred.mean.detach().numpy()))
std_GP = []
std_GP.append(np.mean(np.array(upper_global-lower_global)))
f2_H_GP = []
GP_uncertainty = np.array(upper_global-lower_global).reshape(len(y_test),1)
K_test = GP_uncertainty @ GP_uncertainty.transpose()
K_train = output._covar.detach().numpy()
y_test = y_test.numpy().reshape(len(y_test),1)
f2_H_sample = RKHS_norm(y_test,sigma,K_test)
f2_H_GP.append(f2_H_sample[0,0])

n_train = []
n_train.append(len(X_train))

fig = plt.figure(figsize=(24, 16))

#saph attempt at keeping track of indices
train_indices = []

#%% sample some other data and continue training BNN/GP with a small amount of iterations
for i_train in range(n_start,len(i_train_full)):
    
    # frankie where you left off! you realized you need to keep track of the indices of train samples
    # you need to also randomly initialize location
    i_con = sample_disp_con(x_true,x_true[i_train],r_con)
 
    train_indices.append(n_start + i_train)

    # BNN
    BNN_model.restore("model/model.ckpt-" + str(train_state.best_step) +".ckpt", verbose=1)
    
    X_train = X_train_full[0:int(i_train+1),:]
    y_train = y_train_full[0:int(i_train+1)]
    n_training = len(y_train)
    if isinstance(x_test, torch.Tensor):
        x_test = x_test.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()

    data = dde.data.DataSet(
        X_train,
        y_train.reshape((len(y_train),1)),
        X_test,
        y_test.reshape((len(y_test),1))
    )

    x_train = torch.from_numpy(X_train).float()

    assert isinstance(x_train, torch.Tensor)

    BNN_model.data.train_x = x_train
    BNN_model.data.train_y = y_train.reshape((len(y_train),1))
    
    BNN_model.compile("adam", lr=0.001, metrics=["l2 relative error"])

    BNN_start = time.time() 
    losshistory, train_state = BNN_model.train(iterations=10000, batch_size = n_training, callbacks = [BNN_uncertainty, checkpointer])
    BNN_end = time.time()
    train_time_BNN.append(BNN_end - BNN_start)
    print('BNN took: ', train_time_BNN[-1], 's')
    y_pred = train_state.y_pred_test
    ystd_pred = train_state.y_std_test
    
    y_RMS_BNN.append( RMS(y_pred,y_test.reshape((len(y_test),1))) )    
    std_BNN.append(np.mean(ystd_pred[:,0]))
    K_test = train_state.y_std_test @ train_state.y_std_test.transpose()
    f2_H_sample = RKHS_norm(y_pred,sigma,K_test)
    f2_H_BNN.append(f2_H_sample[0,0])

    i_train_BNN.append(i_train)

    # GP
    x_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float()
    y_train = y_train.float()
    n_train.append(len(x_train))
    x_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
    x_test = x_test.float()
    y_test = y_test.float()
    
    # train model with GPyTorch model, which optimizes hyperparameters
    GP_start = time.time()
    likelihood, model, optimizer, output, loss = GPtrain(x_train, y_train, training_iter)
    GP_end = time.time()
    train_time_GP.append(GP_end - GP_start)
    print('GP took: ', train_time_GP[-1], 's')
    train_pred, lower_train, upper_train = GPeval(x_train, model, likelihood)
    rms_train = RMS(y_train, train_pred.mean.detach().numpy())
    
    test_pred, lower_global, upper_global = GPeval(x_test, model, likelihood)
    rms_test = RMS(y_test, test_pred.mean.detach().numpy())
    y_RMS_GP.append(RMS(y_test, test_pred.mean.detach().numpy()))
    
    # print("    Uncertainty:")
    l_inf = np.max(np.abs(test_pred.mean.numpy()-y_test.numpy()))
    
    GP_uncertainty = np.array(upper_global-lower_global).reshape(len(y_test),1)
    std_GP.append(np.mean(GP_uncertainty))
    
    K_test = GP_uncertainty @ GP_uncertainty.transpose()
    y_global = y_test.numpy().reshape(len(y_test),1)
    f2_H_sample = RKHS_norm(y_global,sigma,K_test)
    f2_H_GP.append(f2_H_sample[0,0])
    
    i_train_GP.append(i_train)

    i_max = np.argmax(GP_uncertainty)

    # Test points are regularly spaced centered along the last index bounded by index displacement
    i_con_GP = sample_disp_con(x_true,x_true[i_train_GP[-1]],r_con)
    i_test_GP = list(set(i_train_full) - set(i_train_GP))
    i_test_local = list(set(i_con_GP) - set(i_train_GP))
    i_test_global = list(set(i_train_full) - set(i_train_GP))
    x_test_local_GP = torch.from_numpy(x_true[i_con_GP, 0:2])
    x_test_global_GP = torch.from_numpy(x_true[i_test_global, 0:2])
    x_test_local_GP = x_test_local_GP.float()
    x_test_global_GP = x_test_global_GP.float()

    # Evaluate RMS for local _BNN
    observed_pred_local, lower_local, upper_local = GPeval(x_test_local_GP, model, likelihood)
    with torch.no_grad():
        f_preds = model(x_test_local_GP)
        y_preds = likelihood(model(x_test_local_GP))
        f_mean = f_preds.mean
        f_var_local = f_preds.variance # variance = np.diag(f_preds.lazy_covariance_matrix.numpy())
        f_covar = f_preds.covariance_matrix
    var_iter_local.append(max(f_var_local.numpy()))
    mse_local_true = sklearn.metrics.mean_squared_error(y_true[i_con_GP], observed_pred_local.mean.numpy())
    rmse_local_true.append(math.sqrt(mse_local_true))
    mse_local_obs = sklearn.metrics.mean_squared_error(y_true[i_con_GP], observed_pred_local.mean.numpy())
    rmse_local_obs.append(math.sqrt(mse_local_obs))
    # and global
    observed_pred_global = GPeval(x_test_global_GP, model, likelihood)
    with torch.no_grad():
        f_preds = model(x_test_global_GP)
        y_preds = likelihood(model(x_test_global_GP))
        f_mean = f_preds.mean
        f_var_global = f_preds.variance
        f_covar = f_preds.covariance_matrix

    f_mean_numpy = f_mean.numpy()
    var_iter_global.append(max(f_var_global.numpy()))
    mse_global_true = sklearn.metrics.mean_squared_error(y_true[i_test_GP], f_mean_numpy)
    rmse_global_true.append(math.sqrt(mse_global_true))
    mse_global_obs = sklearn.metrics.mean_squared_error(y_obs[i_test_GP], f_mean_numpy)
    rmse_global_obs.append(math.sqrt(mse_global_obs))

    # plot real surface and the observed measurements
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = plotBoth()
    warnings.filterwarnings("ignore", category=gpytorch.utils.warnings.GPInputWarning)
    plt.savefig('figure.png')
       
    fig.tight_layout()
    fig.savefig(image_path+str(i_train)+'.png')
    fig.clear()
    

print('Length GP:'+str(len(i_train_GP)))
print('Length BNN:'+str(len(i_train_BNN)))

def find_distance(x_true,i_train):
    n_end = len(i_train)
    rover_distance = np.zeros(n_end)
    for i in range(n_end-1):
        x_1 = x_true[i_train[i]]
        x_2 = x_true[i_train[i+1]]
        # x_disp = (x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2 + (x_2[2]-x_1[2])**2
        x_disp = np.sqrt((x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2)
        rover_distance[i+1] = rover_distance[i] + x_disp  
    return rover_distance

rover_distance_GP = find_distance(x_true, i_train_GP)
print('GP total roving distance is '+str(rover_distance_GP[-1]))

rover_distance_BNN = find_distance(x_true, i_train_BNN)
print('BNN total roving distance is '+str(rover_distance_BNN[-1]))

observed_pred_global, lower_global, upper_global = GPeval(x_true, model, likelihood)
with torch.no_grad():
    f_preds = model(torch.from_numpy(x_true).float())
    f_mean = f_preds.mean.numpy()
i_min_GP = np.argmin(f_mean)
print('GP rover converged on min at '+str(x_true[i_min_GP]))
i_min_real = np.argmin(y_obs)
print('GP true min at '+str(x_true[i_min_real]))
x_1 = x_true[i_min_GP]
x_2 = x_true[i_min_real]
x_disp = np.sqrt((x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2)
print('GP min error is '+str(x_disp))

y_pred_BNN = train_state.y_pred_test
i_min_BNN = np.argmin(y_pred_BNN)
print('BNN rover converged on min at '+str(x_true[i_min_BNN]))
print('BNN true min at '+str(x_true[i_min_real]))
x_1 = x_true[i_min_BNN]
x_2 = x_true[i_min_real]
x_disp = np.sqrt((x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2)
print('BNN min error is '+str(x_disp))

# and RKHS norm
print("GP: the ending global RKHS_norm is " + str(f2_H_GP[-1]))
print("BNN: the ending global RKHS_norm is " + str(f2_H_BNN[-1]))

# and variance
print("GP: variance is " + str(std_GP[-1]))
print("BNN: variance is " + str(np.mean(ystd_pred)))
print("BNN: variance is (in case the other one is wrong) " + str(np.mean(ystd_pred[-1])))
print("BNN: variance is (in case the other one is wrong again) " + str(np.mean(train_state.y_std_test[:,0])))
print("BNN: variance is (...) " + str(std_BNN[-1]))

#%% plot performance

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 2, 1)
BNN_RMS = ax1.plot(n_train, y_RMS_BNN, label='BNN')
GP_RMS = ax1.plot(n_train, y_RMS_GP, label='GP')
ax1.legend()
ax1.set_xlabel('samples')
ax1.set_ylabel('y_RMS')
ax1.set_title('BNN vs GP error')

ax2 = fig.add_subplot(1, 2, 2)
BNN_time = ax2.plot(n_train, train_time_BNN, label='BNN')
GP_time = ax2.plot(n_train, train_time_GP, label='GP')
ax2.legend()
ax2.set_xlabel('samples')
ax2.set_ylabel('train time')
ax2.set_title('BNN vs GP train time')

fig.savefig(image_path + trial_name+'_traintime_BNN20000_GPRBF.png')

#%% convert images to video
video_name = image_path + 'incremental_'+trial_name+'.avi'

images = []
int_list = []
for img in os.listdir(image_path):
    if img.endswith(".png"):
        images.append(img)
        s = re.findall(r'\d+', img)
        try:
            int_list.append(int(s[0]))
        except:
            print("whatever")

arg_list = np.argsort(int_list)

frame = cv2.imread(os.path.join(image_path, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(
    video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

for i in range(len(arg_list)):
    image = images[arg_list[i]]
    video.write(cv2.imread(os.path.join(image_path, image)))

cv2.destroyAllWindows()
video.release()

sys.stdout.close()
sys.stdout=stdoutOrigin

#%% calculate convergence value of RMS error and distance until convergence

def find_convergence(rmse_global, rover_distance, model):
    v = rmse_global
    v0 = np.max(rmse_global)
    vf0 = rmse_global[-1]
    dv = v0 - vf0
    # band of noise allowable for 2% settling time convergence
    dv_2percent = 0.02*dv 
    
    # is there even enough data to confirm convergence?
    v_95thresh = v0 - 0.95*dv
    i_95thresh = np.where(v<v_95thresh)
    i_95thresh = np.array(i_95thresh[0],dtype=int)
    if len(i_95thresh)>=10:
        for i in range(len(i_95thresh)):
            v_con = v[i_95thresh[i]:-1]
            vf = np.mean(v_con)
            if np.all(v_con<= vf+dv_2percent) and np.all(v_con >= vf-dv_2percent):
                print(model + " convergence index is "+ str(i_95thresh[i])+" where the total samples is "+str(len(rover_distance)))
                print(model + " convergence rms error is "+str(rmse_global[i_95thresh[i]]))
                print(model + " convergence roving distance is "+ str(rover_distance[i_95thresh[i]]))
                print(model + " reduction of error is "+ str(max(rmse_global)/rmse_global[i_95thresh[i]]))
                # plotty plot plot converge wrt rms error and distance!
                fig = plt.figure(figsize=(12,6))
                ax1 = fig.add_subplot(1, 2, 1)
                # local_rms = ax1.plot(range(0,len(rmse_local)), rmse_local, color='blue', marker='.', label='local')
                global_rms = ax1.plot(range(0,len(rmse_global)), rmse_global, color='black', marker='*', label='global')
                ax1.plot([0,len(var_iter_global)], np.array([1,1])*(vf+dv_2percent), 'r--')
                ax1.plot([0,len(var_iter_global)], np.array([1,1])*(vf-dv_2percent), 'r--')
                ax1.plot(i_95thresh[i]*np.array([1,1]),[0,v0],'r--')
                ax1.set_xlabel('number of samples')
                ax1.set_ylabel('rmse')
                ax1.legend(['local','global','convergence bounds'], loc='upper right')
                ax1.set_title(model + ' rmse of learned model')
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(range(len(rover_distance)),rover_distance,'k*-')
                ax2.plot(i_95thresh[i]*np.array([1,1]),[0,max(rover_distance)],'r--')
                ax2.plot([0,len(rover_distance)],rover_distance[i_95thresh[i]]*np.array([1,1]),'r--')
                ax2.set_xlabel('number of samples')
                ax2.set_ylabel('roving distance')
                ax2.set_title(model + ' rover distance during exploration')
                plt.show()
                fig.tight_layout()
                fig.savefig(image_path+model + ' convergence.png')
                i_con = i_95thresh[i]
                n_con = len(rover_distance)
                d_con = rover_distance[i_95thresh[i]]
                return i_con, n_con, d_con
    else:
        print("not able to evaluate convergence")
        print("RMS error upon end is "+ str(rmse_global[-1]))
        print("reduction of error is "+ str(max(rmse_global)/rmse_global[-1]))
        i_con = [0]
        n_con = 0
        d_con = [0]
    return i_con, n_con, d_con
        
i_con_BNN, n_con_BNN, d_con_BNN = find_convergence(y_RMS_BNN, rover_distance_BNN, 'BNN')
i_con_GP, n_con_GP, d_con_GP = find_convergence(y_RMS_GP, rover_distance_GP, 'GP')
