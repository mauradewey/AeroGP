import numpy as np
import pandas as pd
import xarray as xr
from xskillscore import crps_gaussian

seconds_in_year = 60 * 60 * 24 * 365.25
area_of_earth = 5.10072e14 # m^2 

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

def lat_weighted_sum(x):
    weights = np.cos(np.deg2rad(x.lat))
    return x.weighted(weights).sum(['lat', 'lon'])

def normalize_data(data):
    #normalize data by mean and std
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    return data, mean, std


#NRMSE:
def get_nrmse(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return np.sqrt(((truth - pred)**2).weighted(weights).mean(['lat', 'lon'])).data / np.abs(truth.weighted(weights).mean(['lat','lon']).data)

#mean bias:
def get_bias(truth, pred):
    weights = np.cos(np.deg2rad(truth.lat))
    return (pred - truth).weighted(weights).mean(['lat', 'lon']).data

#crps:
def get_crps(truth, pred, std):
    weights = np.cos(np.deg2rad(truth.lat))
    return crps_gaussian(truth, pred, std, weights=weights).data



def get_bounds(arr):
    if 'lat_bnds' in arr:
        bounds = {}
        bounds["lon"] = arr["lon"].values
        bounds["lat"] = arr["lat"].values
        bounds["lon_b"] = np.append(arr['lon_bnds'][:,0].values, arr['lon_bnds'][-1,1].values)
        bounds["lat_b"] = np.append(arr['lat_bnds'][:,0].values, arr['lat_bnds'][-1,1].values)

    else:    
        lonMin = np.nanmin(arr["lon"].values)
        latMin = np.nanmin(arr["lat"].values)
        lonMax = np.nanmax(arr["lon"].values)
        latMax = np.nanmax(arr["lat"].values)
        
        sizeLon = len(arr["lon"])
        sizeLat = len(arr["lat"])

        gridSize_lon = (lonMax-lonMin)/sizeLon
        gridSize_lat = (latMax-latMin)/sizeLat
        
        bounds = {}
        
        bounds["lon"] = arr["lon"].values
        bounds["lat"] = arr["lat"].values
        bounds["lon_b"] = np.linspace(lonMin-(gridSize_lon/2), lonMax+(gridSize_lon/2), sizeLon+1)
        bounds["lat_b"] = np.linspace(latMin-(gridSize_lat/2), latMax+(gridSize_lat/2), sizeLat+1).clip(-90, 90)
    
    return bounds

def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    -----------
    modified from
    https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda

def earth_radius(lat):
    '''
    -----------
    Copied from
    https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r



