import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os

def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype('float32'), src.profile

def resample_to_match(path, target_shape):
    with rasterio.open(path) as src:
        data_resampled = src.read(
            out_shape=(1, target_shape[0], target_shape[1]),
            resampling=Resampling.bilinear
        )[0].astype('float32')
        return data_resampled

def mask_clouds(qa_float):
    qa = np.nan_to_num(qa_float, nan=0).astype('uint16')
    cloud_bit = 1 << 3
    shadow_bit = 1 << 4
    clear = ((qa & cloud_bit) == 0) & ((qa & shadow_bit) == 0)
    return clear.astype('float32')

def normalize_array(arr, mask=None):
    arr_flat = arr.flatten()
    if mask is not None:
        if mask.shape != arr.shape:
            raise ValueError(f"掩膜和数组形状不一致: {mask.shape} vs {arr.shape}")
        mask_flat = mask.flatten().astype(bool)
    else:
        mask_flat = ~np.isnan(arr_flat)

    scaler = MinMaxScaler()
    arr_scaled = np.full_like(arr_flat, np.nan)
    arr_scaled[mask_flat] = scaler.fit_transform(arr_flat[mask_flat, None]).flatten()
    return arr_scaled.reshape(arr.shape)

def export_tif(arr, profile, filename):
    profile.update(dtype='float32', count=1)
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(arr.astype('float32'), 1)

def compute_rsei(data):
    cloud_mask = mask_clouds(data['QA'])
    valid_mask = cloud_mask * data['Water_mask']
    for k in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']:
        data[k] *= valid_mask

    ndvi = (data['B5'] - data['B4']) / (data['B5'] + data['B4'] + 1e-6)
    wet = (0.1509 * data['B2'] + 0.1973 * data['B3'] + 0.3279 * data['B4'] +
           0.3406 * data['B5'] - 0.7112 * data['B6'] - 0.4572 * data['B7'])
    lst = (data['LST_day'] + data['LST_night']) / 2

    numerator = (
        2 * data['B6'] / (data['B6'] + data['B5']) -
        (data['B5'] / (data['B5'] + data['B4']) + data['B3'] / (data['B3'] + data['B6']))
    )
    denominator = (
        2 * data['B6'] / (data['B6'] + data['B5']) +
        (data['B5'] / (data['B5'] + data['B4']) + data['B3'] / (data['B3'] + data['B6']))
    )
    ibi = numerator / (denominator + 1e-6)

    si = ((data['B6'] + data['B4']) - (data['B5'] + data['B2'])) / \
         ((data['B6'] + data['B4']) + (data['B5'] + data['B2']) + 1e-6)

    ndbsi = (ibi + si) / 2

    ndvi_n = normalize_array(ndvi, valid_mask)
    wet_n = normalize_array(wet, valid_mask)
    lst_n = normalize_array(lst, valid_mask)
    ndbsi_n = normalize_array(ndbsi, valid_mask)

    stacked = np.stack([ndvi_n, wet_n, lst_n, ndbsi_n], axis=-1)
    flat = stacked.reshape(-1, 4)
    mask = ~np.any(np.isnan(flat), axis=1)
    pca = PCA(n_components=4)
    pc_scores = np.full_like(flat, np.nan)
    pc_scores[mask] = pca.fit_transform(flat[mask])
    pc1 = pc_scores[:, 0].reshape(ndvi.shape)

    rsei_raw = 1 - pc1
    rsei = normalize_array(rsei_raw)
    return rsei, valid_mask, {
        'ndvi': ndvi_n,
        'wet': wet_n,
        'lst': lst_n,
        'ndbsi': ndbsi_n
    }

def compute_rsei_for_year(year_dir):
    bands = {
        'B2': os.path.join(year_dir, 'B2.tif'),
        'B3': os.path.join(year_dir, 'B3.tif'),
        'B4': os.path.join(year_dir, 'B4.tif'),
        'B5': os.path.join(year_dir, 'B5.tif'),
        'B6': os.path.join(year_dir, 'B6.tif'),
        'B7': os.path.join(year_dir, 'B7.tif'),
        'QA': os.path.join(year_dir, 'QA_PIXEL.tif'),
        'LST_day': os.path.join(year_dir, 'LST_Day.tif'),
        'LST_night': os.path.join(year_dir, 'LST_Night.tif'),
        'Water_mask': os.path.join(year_dir, 'water_mask.tif')
    }

    data = {}
    for k in ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'QA', 'Water_mask']:
        data[k], profile = load_band(bands[k])

    target_shape = data['B2'].shape
    data['LST_day'] = resample_to_match(bands['LST_day'], target_shape)
    data['LST_night'] = resample_to_match(bands['LST_night'], target_shape)

    rsei, valid_mask, factors = compute_rsei(data)
    return rsei, profile, factors

def export_all_outputs(rsei, factors, profile, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    export_tif(rsei, profile, os.path.join(output_dir, 'RSEI.tif'))
    export_tif(factors['ndvi'], profile, os.path.join(output_dir, 'NDVI_normalized.tif'))
    export_tif(factors['wet'], profile, os.path.join(output_dir, 'WET_normalized.tif'))
    export_tif(factors['lst'], profile, os.path.join(output_dir, 'LST_normalized.tif'))
    export_tif(factors['ndbsi'], profile, os.path.join(output_dir, 'NDBSI_normalized.tif'))
