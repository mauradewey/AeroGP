#utils
import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import t, ttest_ind, false_discovery_control

# some constants:
SECONDS_IN_YEAR = 60*60*24*365 #seconds in a year
AREA_of_EARTH = 5.10072e14 #m²

# get region bounds:
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

# calculate area of a grid cell:
def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    -----------
    Copied from
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

# calculate earth radius at a given latitude (required for grid cell area calculation):
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

# area weighted methods :
def global_sum(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    glb_sum = ds.weighted(weights).sum(("lon", "lat"))
    return glb_sum

def lat_weighted_mean(ds):
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = "weights"
    gl_mean = ds.weighted(weights).mean(("lon", "lat"))
    return gl_mean

# converts coordinates to plane coordinates
def lla_to_xy(lat,lon):
    a = 6378137
    f = 1/298.257224
    e = np.sqrt(2*f-f**2)

    lat = np.radians(lat)
    lon = np.radians(lon)

    #eccentricity
    C = 1/(np.sqrt((np.cos(lat)**2 + (1-f)**2 * (np.sin(lat))**2)))

    x = (a*C)*np.cos(lat)*np.cos(lon)
    y = (a*C)*np.cos(lat)*np.sin(lon)
    return x,y

# Welch's t-test with FDR correction:
def welch_ttest_fdr(da1, da2, window_size=20):
    """
    Performs t-tests on two time series using windows of given size (default 20 years),
    then applies False Discovery Rate (FDR) correction (significance level 0.05).

    Parameters:
    - da1, da2 (xr.DataArray): Time series data with time as first dimension
    - window_size (int): Size of the window (default is 20)

    Returns:
    - xr.Dataset: A dataset with start & end years, raw p-values, and FDR-adjusted p-values
    """
    # get time dimension for shortest dataset
    if len(da1['year']) < len(da2['year']):
        time_values = da1['year']
    else:
        time_values = da2['year']

    p_values = []
    windows = []

    for end in range(time_values[-1].values, time_values[0].values, -window_size):
        start = end - (window_size - 1)
        if start < time_values[0].values:
            start = time_values[0].values
        
        # Perform Welch's t-test:
        t_stat, p_val = ttest_ind(da1.sel(year=slice(start,end)), da2.sel(year=slice(start,end)), axis=0, equal_var=False, nan_policy='omit', alternative='two-sided')
        p_values.append(p_val)
        windows.append((start, end))

    #apply false discovery control:
    p_values_fdr = false_discovery_control(p_values, method='bh') 

    #save results:
    results = xr.Dataset({
        "start_year": ("window", [w[0] for w in windows]),
        "end_year": ("window", [w[1] for w in windows]),
        "raw_p_values": (["window","lat","lon"], p_values),
        "fdr_adjusted_p_values": (["window","lat","lon"], p_values_fdr)
    },
    coords={
        "window": np.arange(len(windows)),
        "lat": da1['lat'],
        "lon": da1['lon']
    })

    return results