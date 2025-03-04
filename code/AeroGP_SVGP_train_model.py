#need description of what this script does
#Example of how to run:
# to train the model with 0xEA_SO2 as the held-out experiment:
# python AeroGP_SVGP_train_model.py ../config_files/ea0so2_config2.yml --opt 
#
# to load the model from checkpoint and test on 0xEA_SO2:
# python AeroGP_SVGP_train_model.py ../config_files/ea0so2_config2.yml
#
# If you want to test the model with a different experiment than the one that was left 
# out in training, specify the test data and the model to load in the config file.



import numpy as np
import xarray as xr
import gpflow
import tensorflow as tf
import glob
from utils_GP import *
from gpflow.ci_utils import reduce_in_tests
import sys
import argparse
from optionsparser import get_parameters
import os
import time


def main(cfg):

    #unpack config:
    data_dir = cfg.data_dir #training data (eg. '../../training_data_m')
    test_dir = cfg.test_str #experiment for testing (eg. 'glbOC)
    log_dir = cfg.log_dir  #where to save model checkpoints or load model from (eg. 'logs_glboc')
    external_test = cfg.external_test #if not testing on the held-out experiment, specify the test data directory
    opt = cfg.opt #train the model (True) or load from checkpoint (False)

    #setup train/test data
    train_output_norm, out_mean, out_std, train_input_norm, test_input_norm, out_coords, out_dims, out_shape, test_name = setup_data(test_dir, data_dir, external_test)
    num_data = len(train_input_norm)
    training_data = (train_input_norm.astype(np.float64), train_output_norm.astype(np.float64))
    
    #make model
    model = make_model(num_data, train_input_norm)

    #setup checkpointing:
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, log_dir, max_to_keep=5)
    if opt:
        manager.save()
    checkpoint_task = gpflow.monitor.ExecuteCallback(manager.save)
    task_group = gpflow.monitor.MonitorTaskGroup(checkpoint_task, period=5)
    monitor = gpflow.monitor.Monitor(task_group)

    #train model or load pre-trained from log_dir:
    if opt:
        MAXITER = reduce_in_tests(6000)
        logf = optimize_with_Adam_NatGrad(model, training_data, num_data, manager, MAXITER, minib=True)
        elbo_df = pd.DataFrame(logf, columns=['elbo'])
        elbo_df.to_csv(log_dir + '/elbo.csv')
        print(gpflow.utilities.print_summary(model))
    else: 
        checkpoint.restore(manager.latest_checkpoint)
        print("Model weights loaded from checkpoint: ", manager.latest_checkpoint)
        print(gpflow.utilities.print_summary(model))

    #test model and save posterior:
    test_model(model, test_input_norm, out_coords, out_dims, out_shape, out_mean, out_std, log_dir, test_name)


def setup_data(test_str, data_dir, external_test='None'):

    #data folders:
    in_files = glob.glob(data_dir + '/inputs*.nc')
    out_files = glob.glob(data_dir + '/outputs*.nc')

    #make train/test split
    #if external test data is provided, use that for testing
    print('Making train/test data split.')
    if external_test != 'None':
        test_in = [external_test]
        print('Test experiment: ', external_test)
    else:
        test_in = [s for s in in_files if test_str in s]
        print('Test experiment: ', test_str)
    
    #by default 'test_str' is the held-out experiment (not including in training data, even if external test data is provided)
    print('Training experiments: ')
    train_in = [s for s in in_files if test_str not in s]
    train_out = [s for s in out_files if test_str not in s]

    #prepare training data:
    train_input_df = prep_inputs(train_in)
    train_output_df = prep_outputs(train_out)

    #prepare test data:
    test_input_df = prep_inputs(test_in)
    
    #normalize data:
    out_norm, out_mean, out_std = normalize_data(train_output_df)
    in_norm, in_mean, in_std = normalize_data(train_input_df)

    #check array sizes are compatible:
    if (np.shape(in_norm)[0] == np.shape(out_norm)[0]):
        print ("Training input/output data shapes compatible.")
        print("Test inputs file: ", test_in)
    else:
        print("Error! Training data is mis-shapen!")
        sys.exit()

    #normalize test data:
    test_in_norm = (test_input_df - in_mean) / in_std

    #test data name for saving:
    test_name = os.path.splitext(os.path.basename(test_in[0]))[0].split('_',1)[1]

    #coords for test output:
    test_inputs = xr.open_dataset(test_in[0])['SO2']
    out_coords = test_inputs.coords
    out_dims = test_inputs.dims
    out_shape = np.shape(test_inputs)

    return out_norm, out_mean, out_std, in_norm, test_in_norm, out_coords, out_dims, out_shape, test_name


def make_model(num_data, train_input_norm):
    """
    Create a GPflow SVGP model
    """
    print("Creating model.")
    #model parameters
    N = num_data # number of training points
    D = num_features = 27 # number of input features
    M = 100 # number of inducing points
    L = num_latent = 1 # number of latent functions
    P = 13824 # number of pixels/output locations

    # Make base kernel
    kernel_list = [gpflow.kernels.Linear(active_dims=[0, 1, 2]) +
                gpflow.kernels.Matern32(lengthscales=8 * [1.], active_dims=[3, 6, 9, 12, 15, 18, 21, 24]) +
                gpflow.kernels.Matern32(lengthscales=8 * [1.], active_dims=[4, 7, 10, 13, 16, 19, 22, 25]) +
                gpflow.kernels.Matern32(lengthscales=8 * [1.], active_dims=[5, 8, 11, 14, 17, 20, 23, 26]) +
                gpflow.kernels.White()
    for i in range(L)
    ]

    #LMC kernel
    kernel = gpflow.kernels.LinearCoregionalization(
        kernel_list, W=np.random.randn(P,L)
    )

    #choose inducing points
    Zinit = (train_input_norm.sample(M)).to_numpy()
    Z = Zinit.copy() 

    #inducing points
    iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Z)
    )

    # initialize mean of variational posterior to be of shape MxL
    q_mu = np.zeros((M, L))
    # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0

    # create SVGP model
    m = gpflow.models.SVGP(
        kernel,
        gpflow.likelihoods.Gaussian(variance=0.8),
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )

    return m



def optimize_model_with_scipy(model, data, monitor, MAXITER):
    """
    Optimize the model using Scipy 
    """
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": 50, "maxiter": MAXITER},
        step_callback=monitor
    )


def optimize_with_Adam_NatGrad(model, data, num_data, manager, MAXITER, minib):
    """
    Optimize the model using Adam and Natural Gradients
    Also optionally uses minibatching
    """
    print("Optimizing model with Adam and Natural Gradients")

    if minib:
        N = num_data
        minibatch_size = 200
        data_minibatch = (
            tf.data.Dataset.from_tensor_slices(data)
            .repeat()
            .shuffle(N)
            .batch(minibatch_size))
        train_dataset = iter(data_minibatch)
    else:
        train_dataset = data

    # stop Adam from optimizing variational parameters:
    gpflow.set_trainable(model.q_mu, False)
    gpflow.set_trainable(model.q_sqrt, False)

    # set variational parameters:
    var_params = [(model.q_mu, model.q_sqrt)]
    
    # NatGrad optimization of variational parameters:
    natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)

    # Adam optimization of kernel parameters:
    adam_opt = tf.keras.optimizers.Adam(0.01)

    # define the loss closure:
    training_loss = model.training_loss_closure(train_dataset)

    # track elbo during training:
    logf = []

    for step in range(MAXITER):
        adam_opt.minimize(training_loss, var_list=model.trainable_variables)
        natgrad_opt.minimize(training_loss, var_list=var_params)
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
        if step % 100 == 0:
            manager.save()
    return logf
    
def test_model(model, test_input_norm, out_coords, out_dims, out_shape, out_mean, out_std, log_dir, test_name):
    """
    Test the model and save the posterior
    """
    # predict on test data:
    print("Predicting on test data.")
    post_mean, post_var = model.predict_y(test_input_norm.values)

    # un_normalize and reshape outputs:
    posterior_mean = (post_mean * out_std) + out_mean
    train_std = tf.expand_dims(out_std, axis=0)
    posterior_std = np.sqrt(post_var) * train_std

    posterior_tas = np.reshape(posterior_mean, out_shape) #[years, 96, 144])
    posterior_tas_std = np.reshape(posterior_std, out_shape) #[years, 96, 144])
    posterior_data = xr.DataArray(posterior_tas, dims=out_dims, coords=out_coords)
    posterior_data_std = xr.DataArray(posterior_tas_std, dims=out_dims, coords=out_coords)

    # save posterior:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    posterior_df = xr.Dataset({'posterior': posterior_data, 'posterior std': posterior_data_std})
    posterior_df.to_netcdf(log_dir + '/posterior_predictY_'+ test_name +'_'+ timestr +'.nc')
    print("Posterior saved to ", log_dir + '/posterior_predictY_'+ test_name +'_'+ timestr +'.nc')

    # sample latent function:
    #num_samples = 10
    #print("Sampling latent function. Number of samples: ", num_samples)
    #samples = model.predict_f_samples(test_input_norm.values, num_samples)

    # un_normalize and reshape outputs:
    #samples = (samples * out_std) + out_mean
    #samples = np.reshape(samples, (num_samples, *out_shape))
    #samples = xr.DataArray(samples, dims=['sample', *out_dims], coords=out_coords)

    # save samples:
    #samples.to_netcdf(log_dir + '/samples_predictf_'+ test_name +'_'+ timestr +'.nc')
    #print("Samples saved to ", log_dir + '/samples_predictf_'+ test_name +'_'+ timestr +'.nc')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="Config file name")
    # moving option to train or test into config file, so things are easier to run automatically:
    #parser.add_argument("--opt", default=False, action='store_true', help="Train the model. Default is load from checkpoint (opt=False)")
    args = parser.parse_args()

    #set required input parameters:
    required = {
        "data_dir"  : str,  # training data directory
        "test_str"  : str,  # experiment to test on/left out from training
        "log_dir"   : str,  # output directory
        "opt"       : bool, # False = load model from last checkpoint, True = train new model
    }
    defaults = {
        "external_test" : 'None'
    }

    #parse the config file:
    cfg = get_parameters(args.cfg,required,defaults)
    #opt = args.opt
 
    #run main function
    main(cfg)