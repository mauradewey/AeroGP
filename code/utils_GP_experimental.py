import numpy as np
import pandas as pd
import xarray as xr

seconds_in_year = 60 * 60 * 24 * 365.25
area_of_earth = 5.10072e14 # m^2 

#define perturbation regions as  [lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat]
#define perturbation regions as  [lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat]
pert_regions = dict([
    ('gb', [0, -90, 360, 90]),
    ('eu', [165, 37, 210, 73]),
    ('ru', [210, 45, 380, 80]),
    ('ea', [270, -10, 330, 45]),
    ('wa', [240, 0, 270, 45]),
    ('au', [285, -45, 340, -10]),
    ('na', [10, 15, 125, 75]),
    ('sa', [95, -60, 150, 15]),
    ('af', [155, -40, 240, 37])
])

def create_predictor_regions(pert_regions, data_set):
    #for given input dataset and coordingates of regions, sum emissions in those regions and return as dataframe.
    #note: emission units are Tg/yr per gridbox (so summing over lat/lon gives Tg/yr for that region) (not fluxes!)
    inputs = pd.DataFrame()
    X = data_set

    for key_region in pert_regions:
       
        #set up lat/lon slice values:
        lat_slice = slice(pert_regions[key_region][1],pert_regions[key_region][3])
        lon_slice = slice(pert_regions[key_region][0],pert_regions[key_region][2])

        Y_SO2 = X['SO2'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('SO2')
        Y_SO2 = Y_SO2.add_suffix(key_region)
        inputs = pd.concat([inputs, Y_SO2], axis=1)

        Y_BC = X['BC'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('BC')
        Y_BC = Y_BC.add_suffix(key_region)
        inputs = pd.concat([inputs, Y_BC], axis=1)

        Y_OC = X['OC'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('OC')
        Y_OC = Y_OC.add_suffix(key_region)
        inputs = pd.concat([inputs, Y_OC], axis=1)

    return inputs

#pert_regions = dict([
#    ('gb', [0, -90, 360, 90]),
#    ('ru', [55, 45, 180, 80]),
#    ('ea', [95, -10, 150, 45]),
#    ('wa', [60, 0, 95, 45]),
#    ('au', [105, -45, 165, -10]),
#    ('na', [190, 15, 305, 78]),
#    ('sa', [270, -60, 330, 15]),
#    ('af', [-25, -40, 60, 45]),
#    ('eu', [-20, 45, 55, 80]),
#])


#def create_predictor_regions(pert_regions, data_set):
#    #for given input dataset and coordingates of regions, sum emissions in those regions and return as dataframe.
#    #note: emission units are Tg/yr per gridbox (so summing over lat/lon gives Tg/yr for that region) (not fluxes!)
#    inputs = pd.DataFrame()
#    X = data_set#

    #for EU and AF, which cross the maridian, we need to adjust the longitude range to be between -180 and 180
    #make a copy of the dataset and adjust the longitude values:
#    X_mod = X.copy()
#    X_mod['lon'] = np.mod(X_mod['lon'] + 180, 360) - 180
#    X_mod = X_mod.sortby(X_mod.lon)

#    for key_region in pert_regions:

#        #set up lat/lon slice values:
#        lat_slice = slice(pert_regions[key_region][1],pert_regions[key_region][3])
#        lon_slice = slice(pert_regions[key_region][0],pert_regions[key_region][2])

        #select from modified dataset if region is 'eu' or 'af':
#        if key_region == 'eu' or key_region == 'af':
#            Y_SO2 = X_mod['SO2'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('SO2')
#            Y_SO2 = Y_SO2.add_suffix(key_region)
#            inputs = pd.concat([inputs, Y_SO2], axis=1)

#            Y_BC = X_mod['BC'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('BC')
#            Y_BC = Y_BC.add_suffix(key_region)
#            inputs = pd.concat([inputs, Y_BC], axis=1)

#            Y_OC = X_mod['OC'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('OC')
#            Y_OC = Y_OC.add_suffix(key_region)
#            inputs = pd.concat([inputs, Y_OC], axis=1)
        
#        else:       
#            Y_SO2 = X['SO2'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('SO2')
#            Y_SO2 = Y_SO2.add_suffix(key_region)
#            inputs = pd.concat([inputs, Y_SO2], axis=1)

#            Y_BC = X['BC'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('BC')
#            Y_BC = Y_BC.add_suffix(key_region)
#            inputs = pd.concat([inputs, Y_BC], axis=1)

#            Y_OC = X['OC'].sel(lat=lat_slice,lon=lon_slice).sum(('lat','lon')).to_dataframe('OC')
#            Y_OC = Y_OC.add_suffix(key_region)
#            inputs = pd.concat([inputs, Y_OC], axis=1)

#    return inputs

#prepare input data:
# data are annual emissions on 1.9x2.4 degree grid. For training we want a dataframe with total emissions for each region for each year and species, and then concatentate all experiments together.
# data is normalized by mean and std of each species for each experiment later.
# NaNs replaced with 0s.

def add_year(ds):
    #add year as a coordinate to the dataset if it doesn't exist (ie for equilibrium runs where year is not a coordinate)
    if 'year' in ds.coords:
        return ds
    else:
        ds = ds.assign_coords(year=9999)
        ds = ds.expand_dims('year')
        return ds

def prep_inputs(in_files):
    #loop over input files and create a dataframe of emissions for each pert region for each year and experiment.
    input_df = pd.DataFrame()

    data = xr.open_mfdataset(in_files, combine='nested', concat_dim='year', preprocess=add_year)

    inputs = create_predictor_regions(pert_regions, data)
    inputs0 = inputs.fillna(0)
    input_df = pd.concat([input_df, inputs0], axis=0)

    return input_df

#perpare output:
# data are 1.9x2.4 gridded tas difference from baseline for each experiment.
# training dataframe size is n_years x 96*144
# drop rows that contain NaNs (and drop corresponding rows in input_df)

def prep_outputs(out_files):

    output_df = pd.DataFrame()
    
    for file in out_files:
        print(file)
        data = xr.open_dataset(file)
        data = add_year(data)

        outputs = pd.DataFrame(data['tas_diff'].values.reshape(-1, 96 * 144),index=(data['year']))
        output_df = pd.concat([output_df, outputs], axis=0)

    if output_df.isnull().any(axis=1).sum()>0:
        print('NaNs found in output_df')
        #output_df = output_df.dropna()
        #input_df = input_df.dropna() #fix to drop same rows as output_df
        #print('NaNs dropped')

    return output_df

def lat_weighted_mean(x):
    weights = np.cos(np.deg2rad(x.lat))
    return x.weighted(weights).mean(['lat', 'lon'])

def normalize_data(data):
    #normalize data by mean and std
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    return data, mean, std

