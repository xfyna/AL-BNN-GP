# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:00:32 2023

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
import pickle
import re
import cv2
import re
from osgeo import gdal
import sklearn.metrics
import pickle
from numpy.linalg import inv
import random
import sys

func_name = sys.argv[1]
noise_name = sys.argv[2]
kernel_type = sys.argv[3]
poly_rank = int(sys.argv[4])
r_disp = float(sys.argv[5])
constraint_flag = int(sys.argv[6])
local_flag = int(sys.argv[7])

# clear files in model directory
files = glob.glob('../model/*')
for f in files:
    os.remove(f)

# kernel_type = 'RBF'
# poly_rank = 4
# r_disp = 6.0
# constraint_flag = 1
# local_flag = 2

if local_flag ==1:
    explore_name = 'local'
elif local_flag == 2:
    explore_name = 'NN'
else:
    explore_name = 'global'
    
if constraint_flag == 0:
    con_name = 'unconstrained'
else:
    con_name = 'constrained'

def nearest_neighbor(x_sample,x_list,y_list):
    disp = np.sqrt((x_sample[0]-x_list[:,0])**2+(x_sample[1]-x_list[:,1])**2)
    i_min = np.argmin(disp)
    y_sample = y_list[i_min]
    return y_sample

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
elif func_name == 'townsend':
    grid_bounds = [(-1.75, 1.75), (-1.75, 1.75)]
    grid_size = 21
    x_1 = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_size)
    x_2 = x_1
    grid_diff = x_1[1] - x_1[0]
    print(grid_diff)
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

# min resides at (2.5510, 0.0258)
elif func_name == 'lunar':
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

    #%% focus on a 1.5 km radius at the bottom of the crater. center about the lowest point
    r_disp = 3

    # x_center = x_true[np.argmin(x_true[:,2])]
    # x_center = x_true[int(np.median(range(0,len(y_obs))))]
    # x_center_all = np.mean(x_true,0)
    # x_disp = np.sqrt((x_true[:,0]-x_center_all[0])**2 + (x_true[:,1]-x_center_all[1])**2 + (x_true[:,2]-x_center_all[2])**2)
    # i_min = np.argmin(x_disp)
    # x_center = x_true[i_min,:]

    # i_con = sample_disp_con(x_true,x_center,r_disp)

    # x_true = x_true_doub[i_con,:] - x_center
    # y_obs = y_obs_doub[i_con]

    # n_samples = len(y_obs)

    x_center_all = np.mean(x_true,0)
    x_disp = np.sqrt((x_true[:,0]-x_center_all[0])**2 + (x_true[:,1]-x_center_all[1])**2 + (x_true[:,2]-x_center_all[2])**2)
    i_min = np.argmin(x_disp)
    x_center = x_true[i_min,:]
    
    x_true = x_true_doub - x_center
    
    y_obs = y_obs[np.argwhere(x_true[:,0]>=-r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,0]>=-r_disp/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,1]>=-r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,1]>=-r_disp/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,0]<=r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,0]<=r_disp/2)[:,0]]
    y_obs = y_obs[np.argwhere(x_true[:,1]<=r_disp/2)[:,0]]
    x_true = x_true[np.argwhere(x_true[:,1]<=r_disp/2)[:,0]]
    
    y_true = y_obs #x_true[:,2] NOPE - DON'T DO THIS LOL
    x_true_2 = x_true
    x_true = x_true[:, :2]
    n_samples = len(y_obs)
    sigma = math.sqrt(0.02)

trial_name = str(func_name)+'_'+str(noise_name)+'_'+str(con_name)+'_'+str(explore_name)+'_'+str(kernel_type)
parent_dir = '../GPAL'
image_path = os.path.join(parent_dir, trial_name + '/')
os.mkdir(image_path)
    
stdoutOrigin=sys.stdout 
sys.stdout = open(image_path+"log.txt", "w")
#%% functions
# calculate RKHS norm from covariance
def RKHS_norm(y,sigma,K):
    n_row, n_col = K.shape
    alpha = inv(K + sigma**2 * np.eye(n_row)) @ y
    return alpha.transpose() @ K @ alpha

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

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
   
def unique_sample(i_sample,i_set,i_train,i_max,x):
    if i_sample <= i_max and i_sample >= 0:
        if i_sample not in i_train:
            i_new = i_sample
        else:
            i_set_unique = set(i_set)-set(i_train)
            if not i_set_unique:
                return []
            i_set_unique = list(i_set_unique)
            x_start = x[i_sample,:]
            x_disp = np.sqrt((x[i_set_unique,0]-x_start[0])**2 + (x[i_set_unique,1]-x_start[1])**2)
            # disp_i = np.abs(np.array(i_set_unique)-np.array(i_sample))
            i_new =i_set_unique[np.argmin(x_disp)]
    elif i_sample > i_max:
        i_new = unique_sample(i_sample-1,i_set,i_train,i_max,x)
    else:
        i_new = unique_sample(i_sample+1,i_set,i_train,i_max,x)
    return i_new

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

#%% plotting
def plotBoth():
    
    X_train_BNN = X_train_full[i_train_BNN,0:2]
    y_train_BNN = y_train_full[i_train_BNN]
    X_test_BNN = X_train_full[i_test_BNN,0:2]
    y_test_BNN = y_train_full[i_test_BNN]
    y_pred_BNN = train_state.y_pred_test
    ystd_pred = train_state.y_std_test
    print(ystd_pred[:,0])
    residual = y_test_BNN[:] - y_pred_BNN[:]
    
    X_train_GP = X_train_full[i_train_GP,0:2]
    y_train_GP = y_train_full[i_train_GP]
    X_test_GP = X_train_full[i_test_GP,0:2]
    y_test_GP = y_train_full[i_test_GP]
    
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    rover = ax1.scatter3D(X_train_BNN[-1, 0], X_train_BNN[-1, 1],
                          y_train_BNN[-1], s=100, color='yellow', marker='*', zorder=1)
    rover = ax1.scatter3D(X_train_BNN[-2, 0], X_train_BNN[-2, 1],
                          y_train_BNN[-2], s=100, color='purple', marker='*', zorder=1)
    rover_path = ax1.plot3D(X_train_BNN[:,0], X_train_BNN[:,1], train_state.y_train[:, 0], color='black')
    
    test_surf = ax1.plot_trisurf(X_test_BNN[:, 0], X_test_BNN[:, 1], y_pred_BNN[:,0], color='grey', alpha=0.25)
    # lunar? points_pred = ax1.plot_trisurf(x_test_global[:, 0], x_test_global[:, 1], observed_pred_global.mean.numpy(), color='grey', alpha=0.25)
    true_surf = ax1.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno',
                            linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.set_zlabel("$y")
    ax1.set_title('rover on surface '+str(x_true[i_train, :]))
    
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    ax2.set_title('B-NN uncertainty')
    std_pred = ax2.plot_trisurf(X_test_BNN[:, 0], X_test_BNN[:, 1], ystd_pred[:,0], color='grey', alpha=0.25)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_zlabel("std(y)")
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.e'))

    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    ax3.set_title('B-NN residual')
    res_pred = ax3.plot_trisurf( X_test_BNN[:, 0], X_test_BNN[:, 1], residual[:,0], color='grey', alpha=0.25)
    ax3.set_xlabel("$x_1$")
    ax3.set_ylabel("$x_2$")
    ax3.set_zlabel("err(y)")
    ax3.zaxis.set_major_formatter(FormatStrFormatter('%.e'))
    
    ax4 = fig.add_subplot(3, 3, 4, projection='3d')
    ax4.set_title('GP prediction vs truth')
    ax4.scatter3D(X_train_GP[-1, 0], X_train_GP[-1, 1],
                          y_train_GP[-1], s=100, color='yellow', marker='*', zorder=1)
    ax4.scatter3D(X_train_GP[-2, 0], X_train_GP[-2, 1],
                          y_train_GP[-2], s=100, color='purple', marker='*', zorder=1)
    train_points = ax4.plot3D(X_train_GP[:, 0], X_train_GP[:, 1], y_train_GP, color='black')
    test_surf = ax4.plot_trisurf( X_test_GP[:, 0], X_test_GP[:, 1], y_pred_GP, color='grey', alpha=0.25)
    true_surf = ax4.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno',
                            linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))
    
    ax4.set_xlabel("$x_1$")
    ax4.set_ylabel("$x_2$")
    ax4.set_zlabel("$y")
    
    ax5 = fig.add_subplot(3, 3, 5, projection='3d')
    ax5.set_title('GP uncertainty')
    std_pred = ax5.plot_trisurf( X_test_GP[:, 0], X_test_GP[:, 1], upper_global - lower_global, color='grey', alpha=0.25)
    ax5.set_xlabel("$x_1$")
    ax5.set_ylabel("$x_2$")
    ax5.set_zlabel("std(y)")
    
    ax6 = fig.add_subplot(3, 3, 6, projection='3d')
    ax6.set_title('GP residual')
    res_pred = ax6.plot_trisurf( X_test_GP[:, 0], X_test_GP[:, 1], test_pred.mean.numpy()-y_test_GP, color='grey', alpha=0.25)
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
    BNN_variance = ax8.plot(range(0, len(std_BNN)), std_BNN, color = 'black', marker='.')
    GP_variance = ax8.plot(range(0, len(std_GP)), std_GP, color = 'blue', marker ='.')
    ax8.set_ylabel('variance')
    ax8.set_xlabel('number of samples')
    ax8.legend(['BNN', 'GP'], loc='lower right')
    # variance comparison
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_title('RKHS norm')
    BNN_RKHS = ax9.plot(range(0, len(f2_H_BNN)), f2_H_BNN, color = 'black', marker='.')
    GP_RKHS = ax9.plot(range(0, len(f2_H_GP)), f2_H_GP, color = 'blue', marker ='.')
    ax9.set_ylabel('variance')
    ax9.set_xlabel('number of samples')
    ax9.legend(['BNN', 'GP'], loc='lower right')
    
    return ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9

#%% randomly initialize location
i_0 = random.randrange(n_samples)
i_train = []
i_train.append(i_0)
i_train_full = list(range(0,n_samples))

# randomly sample next 10 data points with a displacement constraint of 10int
r_NN = np.sqrt(3)*grid_diff
r_con = 3*r_NN
# randomly sample next 10 data points with a displacement constraint of 10int
if func_name == 'lunar':
    for i in range(10):
        i_sample_set = sample_disp_con(x_true_2,x_true_2[i_train[-1]],r_NN) # nearest neighbor (within 0.25 km)
        # i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],r_con) # within 1 km
        i_sample = i_sample_set[random.randrange(len(i_sample_set))]
        i_train.append(int(i_sample))
else:
    for i in range(10):
        i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],r_NN) # nearest neighbor (within 0.25 km)
        # i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],r_con) # within 1 km
        i_sample = i_sample_set[random.randrange(len(i_sample_set))]
        i_train.append(int(i_sample))   
i_train_BNN = list(set(i_train))
i_train_GP = list(set(i_train))

#%% hyperparameters for exploration training
training_iter = 100
sample_iter = int(n_samples/2) #previously divided by 3 not 2?
var_iter = []
var_iter_local = []
var_iter_global = []
rmse_local_obs = []
rmse_global_obs = []
rmse_local_true = []
rmse_global_true = []
lengthscale = [0]

if kernel_type == 'RBF' or kernel_type == 'Matern'  or kernel_type == 'Piece_Polynomial':
# RBF Kernel
    noise = [0]
elif kernel_type == 'Periodic':
    period_length = [0]
elif kernel_type == 'RQ':
    alpha = [0]
elif kernel_type == 'Linear':
    variance = [0]
elif kernel_type == 'Polynomial':
    offset = [0]

covar_global = []
covar_trace = []
covar_totelements = []
covar_nonzeroelements = []
AIC = [] # Akaike Information Criterion
BIC = [] # Bayesian Information Criterion

f2_H_global_GP = []
f2_H_local_GP = []

# initialize animation
fig = plt.figure(figsize=(24, 16))

#%% initiailize BNN and GP

# create B-NN and train
i_test = set(i_train_full) - set(i_train)

X_train_full = x_true[list(i_train_full),:]
y_train_full = y_true[list(i_train_full)]

X_test_BNN = x_true[list(i_test),:]
y_test_BNN = y_true[list(i_test)]


# initialize a small random dataset and train BNN with a large amount of iterations
n_start = 10
X_train_BNN = X_train_full[i_train_BNN,:]
y_train_BNN = y_train_full[i_train_BNN]

data = dde.data.DataSet(
    X_train_BNN,
    y_train_BNN.reshape((len(y_train_BNN),1)),
    X_test_BNN,
    y_test_BNN.reshape((len(y_test_BNN),1))
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

y_pred_BNN = BNN_model.predict(X_test_BNN)

y_RMS_BNN = []
y_RMS_BNN.append(RMS(y_pred_BNN,y_test_BNN.reshape((len(y_test_BNN),1))))
# print('y_RMS is %.3f' % (y_RMS_BNN[-1]) )

std_BNN = []
std_BNN.append(np.mean(train_state.y_std_test[:,0]))
# print('y_RMS is %.3f' % (y_RMS_BNN[-1]) )

f2_H_BNN = []
K_test = train_state.y_std_test @ train_state.y_std_test.transpose()
f2_H_sample = RKHS_norm(y_pred_BNN,sigma,K_test)
f2_H_BNN.append(f2_H_sample[0,0])

# create GP and train
kernel_type = 'RBF'
training_iter = 100
X_train_GP = torch.from_numpy(X_train_full[i_train_GP,:])
y_train_GP = torch.from_numpy(y_train_full[i_train_GP])
X_train_GP = X_train_GP.float()
y_train_GP = y_train_GP.float()

x_test_GP = torch.from_numpy(x_true[list(i_test),0:2])
y_test_GP = torch.from_numpy(y_true[list(i_test)])
x_test_GP = x_test_GP.float()
y_test_GP = y_test_GP.float()

# train model with GPyTorch model, which optimizes hyperparameters
train_time_GP = []
GP_start = time.time()
likelihood, model, optimizer, output, loss = GPtrain(X_train_GP, y_train_GP, training_iter)
GP_end = time.time()
train_time_GP.append(GP_end - GP_start)

train_pred, lower_train, upper_train = GPeval(X_train_GP, model, likelihood)
rms_train = RMS(y_train_GP, train_pred.mean.detach().numpy())
test_pred, lower_global, upper_global = GPeval(x_test_GP, model, likelihood)

y_RMS_GP = []
y_RMS_GP.append(RMS(y_test_GP, test_pred.mean.detach().numpy()))
std_GP = []
std_GP.append(np.mean(np.array(upper_global-lower_global)))
f2_H_GP = []
GP_uncertainty = np.array(upper_global-lower_global).reshape(len(y_test_GP),1)
K_test = GP_uncertainty @ GP_uncertainty.transpose()
K_train = output._covar.detach().numpy()
y_test_GP = y_test_GP.numpy().reshape(len(y_test_GP),1)
f2_H_sample = RKHS_norm(y_test_GP,sigma,K_test)
f2_H_GP.append(f2_H_sample[0,0])

n_train = []
n_train.append(len(X_train_GP))

#%% sample some other data and continue training BNN/GP with a small amount of iterations
for i_train in range(sample_iter):
    
    # BNN
    try:
        BNN_model.restore("model/model.ckpt-" + str(train_state.best_step) +".ckpt", verbose=1)
    except:
        print("was not able to load best model")
        
    X_train_BNN = X_train_full[i_train_BNN,:]
    y_train_BNN = y_train_full[i_train_BNN]
    n_training = len(y_train_BNN)
    i_test_BNN = list(set(i_train_full) - set(i_train_BNN))
    X_test_BNN = X_train_full[i_test_BNN, :]
    y_test_BNN = y_train_full[i_test_BNN]

    data = dde.data.DataSet(
        X_train_BNN,
        y_train_BNN.reshape((len(y_train_BNN),1)),
        X_test_BNN,
        y_test_BNN.reshape((len(y_test_BNN),1))
    )
    
    y_train_BNN = torch.from_numpy(y_train_BNN).float()
    X_train_BNN = torch.from_numpy(X_train_BNN).float()
    X_test_BNN = torch.from_numpy(X_test_BNN).float()
    y_test_BNN = torch.from_numpy(y_test_BNN).float()

    assert isinstance(y_train_BNN, torch.Tensor)
    assert isinstance(X_train_BNN, torch.Tensor)
    assert isinstance(X_test_BNN, torch.Tensor)
    assert isinstance(y_test_BNN, torch.Tensor)    
    
    BNN_model.data.train_x = X_train_BNN
    BNN_model.data.train_y = y_train_BNN.reshape((len(y_train_BNN),1))
    BNN_model.data.test_x = X_test_BNN
    BNN_model.data.test_y = y_test_BNN.reshape((len(y_test_BNN),1))
    
    BNN_model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
    BNN_start = time.time()
    losshistory, train_state = BNN_model.train(iterations=10000, batch_size = n_training, callbacks = [BNN_uncertainty, checkpointer])
    BNN_end = time.time()
    train_time_BNN.append(BNN_end - BNN_start)
    print('BNN took: ', train_time_BNN[-1], 's')
    y_pred_BNN = train_state.y_pred_test
    ystd_pred = train_state.y_std_test
    
    y_RMS_BNN.append(RMS(y_pred_BNN,y_test_BNN.reshape((len(y_test_BNN),1))) )    
   
    sum_BNN = sum(train_state.y_std_test[:,0])
    n = len(train_state.y_std_test[:,0])
    mean = sum_BNN/n

    std_BNN.append(mean)
    K_test = train_state.y_std_test @ train_state.y_std_test.transpose()
    f2_H_sample = RKHS_norm(y_pred_BNN,sigma,K_test)
    f2_H_BNN.append(f2_H_sample[0,0])
    
    # Test points are regularly spaced centered along the last index bounded by index displacement
    i_con_BNN = sample_disp_con(x_true,x_true[i_train_BNN[-1]],r_con)
    i_test_local = list(set(i_con_BNN)) #- set(i_train_BNN))
    x_test_local_BNN = torch.from_numpy(x_true[i_test_local, :])
    x_test_local_BNN = x_test_local_BNN.float()
    i_test_global = list(set(i_train_full) - set(i_train_BNN))
    x_test_global_BNN = torch.from_numpy(x_true[i_test_global, :])
    x_test_global_BNN = x_test_global_BNN.float()

    # GP
    X_train_GP = X_train_full[i_train_GP,:]
    y_train_GP = y_train_full[i_train_GP]
    X_train_GP = torch.from_numpy(X_train_GP)
    y_train_GP = torch.from_numpy(y_train_GP)
    X_train_GP = X_train_GP.float()
    y_train_GP = y_train_GP.float()
    n_train.append(len(X_train_GP))
    i_test_GP = list(set(i_train_full) - set(i_train_GP))
    X_test_GP = X_train_full[i_test_GP,:]
    y_test_GP = y_train_full[i_test_GP]
    X_test_GP = torch.from_numpy(X_test_GP)
    y_test_GP = torch.from_numpy(y_test_GP)
    X_test_GP = X_test_GP.float()
    y_test_GP = y_test_GP.float()
    
    # train model with GPyTorch model, which optimizes hyperparameters
    GP_start = time.time()
    likelihood, model, optimizer, output, loss = GPtrain(X_train_GP, y_train_GP, training_iter)
    GP_end = time.time()
    train_time_GP.append(GP_end - GP_start)
    print('GP took: ', train_time_GP[-1], 's')
    train_pred, lower_train, upper_train = GPeval(X_train_GP, model, likelihood)
    rms_train = RMS(y_train_GP, train_pred.mean.detach().numpy())
    
    test_pred, lower_global, upper_global = GPeval(X_test_GP, model, likelihood)
    y_pred_GP = test_pred.mean.detach().numpy()
    rms_test = RMS(y_test_GP, y_pred_GP)
    y_RMS_GP.append(RMS(y_test_GP, y_pred_GP))
    
    # print("    Uncertainty:")
    l_inf = np.max(np.abs(test_pred.mean.numpy()-y_test_GP.numpy()))
    
    # Test points are regularly spaced centered along the last index bounded by index displacement
    i_con_GP = sample_disp_con(x_true,x_true[i_train_GP[-1]],r_con)
    i_test_local = list(set(i_con_GP) - set(i_train_GP))
    i_test_global = list(set(i_train_full) - set(i_train_GP))
    x_test_local_GP = torch.from_numpy(x_true[i_con_GP, :])
    x_test_global_GP = torch.from_numpy(x_true[i_test_global, :])
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
    observed_pred_global, lower_global, upper_global = GPeval(x_test_global_GP, model, likelihood)
    with torch.no_grad():
        f_preds = model(x_test_global_GP)
        y_preds = likelihood(model(x_test_global_GP))
        f_mean = f_preds.mean
        f_var_global = f_preds.variance
        f_covar = f_preds.covariance_matrix
    var_iter_global.append(max(f_var_global.numpy()))
    mse_global_true = sklearn.metrics.mean_squared_error(y_true[i_test_GP], observed_pred_global.mean.numpy())
    rmse_global_true.append(math.sqrt(mse_global_true))
    mse_global_obs = sklearn.metrics.mean_squared_error(y_obs[i_test_GP], observed_pred_global.mean.numpy())
    rmse_global_obs.append(math.sqrt(mse_global_obs))
    
    # evaluate covariance properties
    covar_global.append(f_covar)
    covar_trace.append(np.trace(f_covar.detach().numpy()))
    covar_totelements.append(np.size(f_covar.detach().numpy()))
    covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # now evaluate information criteria
    # akaike information criterion
    AIC_sample = 2*np.log(covar_nonzeroelements[-1]) - 2*np.log(mse_global_true)
    AIC.append(AIC_sample)
    # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
    BIC_sample = np.size(i_train)*np.log(covar_nonzeroelements[-1]) - 2*np.log(mse_global_true)
    BIC.append(BIC_sample)
    
    GP_uncertainty = np.array(upper_global-lower_global).reshape(len(y_test_GP),1)
    std_GP.append(np.mean(GP_uncertainty))
    
    # and finally evaluate RKHS norm
    K_global_GP = output._covar.detach().numpy()
    y_global_GP = y_train_GP.numpy().reshape(len(y_train_GP),1)
    f2_H_sample_GP = RKHS_norm(y_global_GP,sigma,K_global_GP)
    f2_H_global_GP.append(f2_H_sample_GP[0,0])
    
    n_set = len(i_train_GP)
    n_sub = math.floor(n_set/2)
    i_sub = random.sample(range(1,n_set),n_sub)
    i_sub.sort()
    K_local_GP = K_global_GP[np.ix_(i_sub,i_sub)]
    y_local_GP = y_global_GP[i_sub]
    f2_H_sample_GP = RKHS_norm(y_local_GP,sigma,K_local_GP)
    f2_H_GP.append(f2_H_sample_GP[0,0])

    # plot real surface and the observed measurements
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = plotBoth()
    plt.show()
    
    fig.tight_layout()
    fig.savefig(image_path+str(i_train)+'.png')
    fig.clear()
    
    # sampling strategy
    # move toward location of maximum variance and sample nearest neighbor along the way
    if local_flag ==1 or local_flag ==2:
        # GP waypoint within r_con with maximum variance, nearest neighbor along the way
        GP_uncertainty = upper_local-lower_local
        i_max_GP = np.argmax(GP_uncertainty)
        x_max = x_test_local_GP[i_max_GP,:].numpy()
        i_NN = sample_disp_con(x_true,x_true[i_train_GP[-1]],r_NN)
        dx_NN = np.sqrt((x_true[i_NN,0]-x_max[0])**2 + (x_true[i_NN,1]-x_max[1])**2)
        i_dx = np.argsort(dx_NN) # finds the nearest neigbor along the way to the point of highest variance
        i_sample_GP = []
        j = 0
        while not np.array(i_sample_GP).size:
            i_sample_GP = unique_sample(i_NN[i_dx[j]],i_con_GP,i_train_GP,n_samples-1,x_true)
            j = j+1
        i_train_GP.append(int(i_sample_GP))
        # BNN waypoint generation
        max_index = train_state.y_std_test.shape[0] - 1
        i_con_BNN = [idx for idx in i_con_BNN if 0 <= idx <= max_index]
        BNN_std = ystd_pred[i_con_BNN,0]
        i_max_BNN = np.argmax(BNN_std)
        x_max = x_test_local_BNN[i_max_BNN,:].numpy()
        i_NN = sample_disp_con(x_true,x_true[i_train_BNN[-1]],r_NN)
        dx_NN = np.sqrt((x_true[i_NN,0]-x_max[0])**2 + (x_true[i_NN,1]-x_max[1])**2)
        i_dx = np.argsort(dx_NN) # finds the nearest neigbor along the way to the point of highest variance
        i_sample_BNN = []
        j = 0
        while not np.array(i_sample_BNN).size:
            i_sample_BNN = unique_sample(i_NN[i_dx[j]],i_con_BNN,i_train_BNN,n_samples-1,x_true)
            j = j+1
        i_train_BNN.append(int(i_sample_BNN))
    else:
        # GP waypoint within entire space with max variance, nearest neighbor
        GP_uncertainty = upper_global-lower_global
        i_max_GP = np.argmax(GP_uncertainty)
        if constraint_flag == 1:
            x_max = x_test_global_GP[i_max_GP,:].numpy()
            i_NN = sample_disp_con(x_true,x_true[i_train_GP[-1]],r_NN)
            dx_NN = np.sqrt((x_true[i_NN,0]-x_max[0])**2 + (x_true[i_NN,1]-x_max[1])**2)
            i_dx = np.argsort(dx_NN)
            # finds the nearest neigbor along the way to the point of highest variance
            i_sample_GP = []
            j = 0
            while not np.array(i_sample_GP).size:
                i_sample_GP = unique_sample(i_NN[i_dx[j]],i_con_GP,i_train_GP,n_samples-1,x_true)
                j = j+1
            i_train_GP.append(int(i_sample_GP))
        else:
            i_train_GP.append(i_max_GP)
        # BNN waypoint within entire space with max variance, nearest neighbor
        BNN_std = ystd_pred
        i_max_BNN = np.argmax(BNN_std)
        if constraint_flag == 1:
            x_max = x_test_global_BNN[i_max_BNN,:].numpy()
            i_NN = sample_disp_con(x_true,x_true[i_train_GP[-1]],r_NN)
            dx_NN = np.sqrt((x_true[i_NN,0]-x_max[0])**2 + (x_true[i_NN,1]-x_max[1])**2)
            i_dx = np.argsort(dx_NN)
            # finds the nearest neigbor along the way to the point of highest variance
            i_sample_BNN = []
            j = 0
            while not np.array(i_sample_BNN).size:
                i_sample_BNN = unique_sample(i_NN[i_dx[j]],i_con_BNN,i_train_GP,n_samples-1,x_true)
                j = j+1
            i_train_BNN.append(int(i_sample_BNN))
        else:
            i_train_BNN.append(i_max_BNN)
    
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

fig.savefig(image_path + trial_name+'_traintime_BNN10000_GPRBF.png')

#%% convert images to video
video_name = image_path + 'GPvsBNN_AL_'+trial_name+'.avi'

images = []
int_list = []
for img in os.listdir(image_path):
    if img.endswith(".png"):
        images.append(img)
        s = re.findall(r'\d+', img)
        try:
            int_list.append(int(s[0]))
        except:
            print("saf so cool wow")

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

#%% calculate rover path distance for each exploration strategy and ability to find true min
n_end = len(i_train_GP)

X_train_BNN = X_train_full[i_train_BNN,:]
y_train_BNN = y_train_full[i_train_BNN]
X_test_BNN = x_true
y_test_BNN = y_obs
data = dde.data.DataSet(
    X_train_BNN,
    y_train_BNN.reshape((len(y_train_BNN),1)),
    X_test_BNN,
    y_test_BNN.reshape((len(y_obs),1))
)

y_train_BNN = torch.from_numpy(y_train_BNN).float()
X_train_BNN = torch.from_numpy(X_train_BNN).float()
X_test_BNN = torch.from_numpy(X_test_BNN).float()
y_test_BNN = torch.from_numpy(y_test_BNN).float()

assert isinstance(y_train_BNN, torch.Tensor)
assert isinstance(X_train_BNN, torch.Tensor)
assert isinstance(X_test_BNN, torch.Tensor)
assert isinstance(y_test_BNN, torch.Tensor)

BNN_model.data.train_x = X_train_BNN
BNN_model.data.train_y = y_train_BNN.reshape((len(y_train_BNN),1))
BNN_model.data.test_x = X_test_BNN
BNN_model.data.test_y = y_test_BNN.reshape((len(y_obs),1))
BNN_model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = BNN_model.train(iterations=10000, batch_size = n_training, callbacks = [BNN_uncertainty, checkpointer])

def find_distance(x_true,i_train):
    rover_distance = np.zeros(n_end)
    x_disp = np.zeros(n_end-1)
    for i in range(n_end-1):
        x_1 = x_true[i_train[i]]
        x_2 = x_true[i_train[i+1]]
        # x_disp = (x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2 + (x_2[2]-x_1[2])**2
        x_disp = np.sqrt(((x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2 )**2)
        rover_distance[i+1] = rover_distance[i] + x_disp  
    return rover_distance
    
rover_distance_GP = find_distance(x_true, i_train_GP)
print('GPAL total roving distance is '+str(rover_distance_GP[-1]))

rover_distance_BNN = find_distance(x_true, i_train_BNN)
print('BNNAL total roving distance is '+str(rover_distance_BNN[-1]))

observed_pred_global, lower_global, upper_global = GPeval(torch.from_numpy(x_true).float(), model, likelihood)
with torch.no_grad():
    f_preds = model(torch.from_numpy(x_true).float())
    f_mean = f_preds.mean.numpy()
i_min_GP = np.argmin(f_mean)
print('GPAL rover converged on min at '+str(x_true[i_min_GP]))
i_min_real = np.argmin(y_obs)
print('GPAL true min at '+str(x_true[i_min_real]))
x_1 = x_true[i_min_GP]
x_2 = x_true[i_min_real]
x_disp = np.sqrt((x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2)
print('GPAL min error is '+str(x_disp))

y_pred_BNN = train_state.y_pred_test
i_min_BNN = np.argmin(y_pred_BNN)
print('BNNAL rover converged on min at '+str(x_true[i_min_BNN]))
print('BNNAL true min at '+str(x_true[i_min_real]))
x_1 = x_true[i_min_BNN]
x_2 = x_true[i_min_real]
x_disp = np.sqrt((x_2[0]-x_1[0])**2 + (x_2[1]-x_1[1])**2)
print('BNNAL min error is '+str(x_disp))

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

# %% describe some covariance characteristics
if kernel_type == 'RBF' or kernel_type == 'Matern'  or kernel_type == 'Piece_Polynomial':
    print("the optimal lengthscale at the end of training is " + str(lengthscale[-1]))
    print("the optimal noise at the end of training is" + str(noise[-1]))
elif kernel_type == 'Periodic':
    print("the period length at the end of training is" + str(period_length[-1]))
    print("the optimal lengthscale at the end of training is " + str(lengthscale[-1]))
elif kernel_type == 'RQ':
    print("the optimal alpha at the end of training is " + str(alpha[-1]))
    print("the optimal lengthscale at the end of training is " + str(lengthscale[-1]))
elif kernel_type == 'Linear':
    print("the optimal variance at the end of training is " + str(variance[-1]))
elif kernel_type == 'Polynomial':
    print("the optimal offset at the end of training is " + str(offset[-1]))
    
# print("the added pointwise variance is 0.02")
print("the final covariance trace is "+ str(covar_trace[-1]))

# let's talk about the information criteria
print("GP: the ending AIC is " + str(AIC[-1]))
print("GP: the ending BIC is " + str(BIC[-1]))

# and RKHS norm
print("GP: the ending global RKHS_norm is " + str(f2_H_GP[-1]))
print("BNN: the ending global RKHS_norm is " + str(f2_H_BNN[-1]))
