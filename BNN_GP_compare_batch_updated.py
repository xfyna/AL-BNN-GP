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
import sys
import glob
from osgeo import gdal
import os
from numpy.linalg import inv

func_name = sys.argv[1]
noise_name = sys.argv[2]
kernel_type = sys.argv[3]
poly_rank = int(sys.argv[4])
r_disp = float(sys.argv[5])

#func_name = 'lunar'
#noise_name = 'noise'
#kernel_type = 'RBF'
#poly_rank = '4'
#r_disp = 3

# clear files in model directory
files = glob.glob('../model/*')
for f in files:
    os.remove(f)
    
#%% functions
# calculate RKHS norm from covariance
def RKHS_norm(y,sigma,K):
    n_row, n_col = K.shape
    alpha = inv(K + sigma**2 * np.eye(n_row)) @ y
    return alpha.transpose() @ K @ alpha

def nearest_neighbor(x_sample,x_list,y_list):
    disp = np.sqrt((x_sample[0]-x_list[:,0])**2+(x_sample[1]-x_list[:,1])**2)
    i_min = np.argmin(disp)
    y_sample = y_list[i_min]
    return y_sample

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

def sample_disp_con(x,x_start,r_disp):
    # x_start = x[i_start,:]
    x_disp = np.sqrt((x[:,0]-x_start[0])**2 + (x[:,1]-x_start[1])**2)
    i_con = np.argwhere(x_disp<=r_disp)
    i_con = np.sort(i_con)
    return list(i_con[:,0])
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
    grid_bounds = [(-2.5, 2.5), (-2.5, 2.5)]
    grid_size = 51
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
    
    n_samples = len(y_obs)
    
# %%
trial_name = str(func_name)+'_'+str(noise_name)+'_batch_'+str(kernel_type)
parent_dir = '..' #\GPAL
image_path = os.path.join(parent_dir, trial_name + '/')
os.mkdir(image_path)
    
stdoutOrigin=sys.stdout 
sys.stdout = open(image_path+"log.txt", "w")

i_train = set(range(0, n_samples, 10))
i_test = set(range(0, n_samples-1)) - i_train
X_train = x_true[list(i_train),0:2]
y_train = y_obs[list(i_train)]
X_test = x_true[list(i_test),0:2]
y_test = y_obs[list(i_test)]
i_train_GP = list(set(i_train))
i_train_full = list(range(0, n_samples))
r_NN = np.sqrt(3)*grid_diff
r_con = 3*r_NN
#%% create B-NN and train

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
net = dde.nn.FNN(layer_size, activation, initializer)

BNN_model = dde.Model(data, net)
uncertainty = dde.callbacks.DropoutUncertainty(period=1000)
BNN_model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = BNN_model.train(iterations=20000, callbacks=[uncertainty])

BNN_RMS = RMS(train_state.best_y,y_test.reshape((len(y_test),1)))
print('BNN_RMS is %.3e' % (BNN_RMS) )

std_BNN = np.mean(train_state.y_std_test[:,0])
print('BNN_std is %.3f' % (std_BNN) )

K_test = train_state.y_std_test @ train_state.y_std_test.transpose()
sigma = math.sqrt(0.02)
f2_H_BNN = RKHS_norm(train_state.best_y,sigma,K_test)

print('RKHS norm of GP is %.3e' % (f2_H_BNN) )
#%% train GP
training_iter = 100

x_train = torch.from_numpy(train_state.X_train)
y_train = torch.from_numpy(train_state.y_train[:,0])
x_train = x_train.float()
y_train = y_train.float()

x_test = torch.from_numpy(train_state.X_test)
y_test = torch.from_numpy(train_state.y_test[:,0])
x_test = x_test.float()
y_test = y_test.float()

# Test points are regularly spaced centered along the last index bounded by index displacement
i_con_GP = sample_disp_con(x_true,x_true[i_train_GP[-1]],r_con)
i_test_global = list(set(i_train_full) - set(i_train_GP))
x_test_global_GP = torch.from_numpy(x_true[i_test_global, :2])
x_test_global_GP = x_test_global_GP.float()

print(x_test_global_GP.shape)

# train model with GPyTorch model, which optimizes hyperparameters
GP_start = time.time()
likelihood, model, optimizer, output, loss = GPtrain(x_train, y_train, training_iter)
GP_end = time.time()
print('GP took: ', (GP_end - GP_start), 's')

train_pred, lower_train, upper_train = GPeval(x_train, model, likelihood)
with torch.no_grad():
    f_preds = model(x_test_global_GP)
    f_mean = f_preds.mean

rms_train = math.sqrt(sklearn.metrics.mean_squared_error(y_train, train_pred.mean.detach().numpy()))
print("Best GP model:")
print('    training RMS: %.2e' % ( rms_train ))

test_pred, lower_global, upper_global = GPeval(x_test, model, likelihood)
GP_RMS = math.sqrt(sklearn.metrics.mean_squared_error(y_test, test_pred.mean.detach().numpy()))
print('    test loss: %.2e' % ( GP_RMS ))

print("    Uncertainty:")
l_inf = np.max(np.abs(test_pred.mean.numpy()-y_test.numpy()))
print('       l_inf: %.6f' % ( l_inf ))

uncertainty = upper_global-lower_global
i_max = np.argmax(uncertainty)
print('    max uncertainty location: [%.8f, %.8f]'% ( x_test[i_max,0],x_test[i_max,1]))

print("'train' took %.6f s" % (GP_end - GP_start))

GP_std = np.sqrt(test_pred.variance)
GP_samples = GP_std * torch.randn_like(test_pred.mean) + test_pred.mean
GP_samples = GP_samples.detach().numpy().reshape(-1,1)
print('GP_std is', GP_samples)

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

K_test = uncertainty.permute(*torch.arange(uncertainty.ndim - 1, -1, -1))
K_test = uncertainty @ K_test
K_train = output._covar.detach().numpy()

#Make sure K_test and K_train have valid shapes
if len(K_test.shape) == 2 and len(K_train.shape) == 2:
	n_row_test, n_col_test = K_test.shape
	n_row_train, n_col_train = K_train.shape
	
	y_test_GP = test_pred.numpy().reshape(len(test_pred),1)
	f2_H_GP = RKHS_norm(y_test_GP,sigma,K_test)
	print('RKHS norm of GP is %.3e' % (f2_H_GP) )

#%% plotting
y_train = train_state.y_train
y_test = train_state.y_test
best_y = train_state.best_y
best_ystd = train_state.best_ystd
residual = y_test[:] - best_y[:]

fig = plt.figure(figsize=(24, 16))
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.set_title('B-NN prediction vs truth')
train_points = ax1.scatter( train_state.X_train[:, 0], train_state.X_train[:, 1], train_state.y_train[:, 0], color='black')
test_surf = ax1.plot_trisurf( train_state.X_test[:, 0], train_state.X_test[:, 1], best_y[:,0], color='grey', alpha=0.25)
true_surf = ax1.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno',
                        linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))

ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$y")

ax2 = fig.add_subplot(2, 3, 2, projection='3d')
ax2.set_title('B-NN uncertainty')
std_pred = ax2.plot_trisurf( train_state.X_test[:, 0], train_state.X_test[:, 1], best_ystd[:,0], color='grey', alpha=0.25)
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_zlabel("std(y)")
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.e'))

ax3 = fig.add_subplot(2, 3, 3, projection='3d')
ax3.set_title('B-NN residual')
res_pred = ax3.plot_trisurf( train_state.X_test[:, 0], train_state.X_test[:, 1], residual[:,0], color='grey', alpha=0.25)
ax3.set_xlabel("$x_1$")
ax3.set_ylabel("$x_2$")
ax3.set_zlabel("err(y)")
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.e'))

ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.set_title('GP prediction vs truth')
train_points = ax4.scatter( train_state.X_train[:, 0], train_state.X_train[:, 1], train_state.y_train[:, 0], color='black')
test_surf = ax4.plot_trisurf( train_state.X_test[:, 0], train_state.X_test[:, 1], test_pred.mean.numpy(), color='grey', alpha=0.25)
true_surf = ax4.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno',
                        linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))

ax4.set_xlabel("$x_1$")
ax4.set_ylabel("$x_2$")
ax4.set_zlabel("$y")

ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.set_title('GP uncertainty')
std_pred = ax5.plot_trisurf( train_state.X_test[:, 0], train_state.X_test[:, 1], upper_global - lower_global, color='grey', alpha=0.25)
ax5.set_xlabel("$x_1$")
ax5.set_ylabel("$x_2$")
ax5.set_zlabel("std(y)")

ax6 = fig.add_subplot(2, 3, 6, projection='3d')
ax6.set_title('GP residual')
res_pred = ax6.plot_trisurf( train_state.X_test[:, 0], train_state.X_test[:, 1], test_pred.mean.numpy()-y_test[:,0], color='grey', alpha=0.25)
ax6.set_xlabel("$x_1$")
ax6.set_ylabel("$x_2$")
ax6.set_zlabel("err(y)")

image_name = str(func_name)+'_'+str(noise_name)
fig.savefig(image_path + image_name+'_BNN20000_GPRBF.png')
