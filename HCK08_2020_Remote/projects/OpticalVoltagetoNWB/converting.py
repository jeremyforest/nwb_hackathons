from datetime import datetime
from dateutil import tz
import os
import numpy as np
import pandas as pd
import pickle

from pynwb import NWBFile
from pynwb.file import Subject
from pynwb import NWBHDF5IO
from pynwb.device import Device
from pynwb.ophys import OpticalChannel
from pynwb.ophys import TwoPhotonSeries
from pynwb.ophys import ImageSegmentation
from pynwb.ophys import RoiResponseSeries
from pynwb.ophys import Fluorescence, DfOverF


### sample data import
def images_list(folder):
    """
    Generate an ordered list of images that will be used for subsequent analysis
    based on the number of files in the desired folder.
    Necessary because of the numbering of the files which makes the extracting based
    on name un-ordered.
    Input: folder where are stored the images
    Output: image list ordered from 0 to last.
    """
    # folder = path_input
    files_len = len(os.listdir(folder))
    image_list = []
    for i in range(files_len):
        image_name = 'image'+ str(i) + '.npy'
        image_list.append(image_name)
    return image_list

def open_roi_file(roi_file_path):
    """
    Open picke file containing roi masks
    """
    with open(roi_file_path, "rb") as file:
        rois = pickle.load(file)
    return rois

def rois_signal(data, rois_masks):
    # rois_masks[0][1].shape
    rois_data_list = []
    for roi in range(len(rois)):
        # roi = 0
        for frame in range(data.shape[0]):
            # frame = 0
            if frame == 0:
                roi_data = data[frame][rois[roi][1]]
                roi_data.shape
            else:
                roi_data = np.vstack((roi_data, data[frame][rois[roi][1]]))
                roi_data.shape
        rois_data_list.append(roi_data)
    return rois_data_list




path_input =  "/home/jeremy/Documents/Formations_Mooc_Workshop/2020_NWB_workshop/nwb_hackathons/HCK08_2020_Remote/projects/OpticalVoltagetoNWB/sample_data/experiment_144/"
path_input_raw_data = "/home/jeremy/Documents/Formations_Mooc_Workshop/2020_NWB_workshop/nwb_hackathons/HCK08_2020_Remote/projects/OpticalVoltagetoNWB/sample_data/experiment_144/raw_data/"
path_output_data = '/home/jeremy/Documents/Formations_Mooc_Workshop/2020_NWB_workshop/nwb_hackathons/HCK08_2020_Remote/projects/OpticalVoltagetoNWB/sample_data/experiment_144/'
files = images_list(path_input_raw_data)
rois = open_roi_file(f'{path_input}roi_masks.txt')


for file in files:
    # file=files[3]
    try:
        if file.endswith('.npy'):
            if file == files[0]:
                data = np.load(f'{path_input_raw_data}/{file}')
                size_img = (int(np.sqrt(data.size)), int(np.sqrt(data.size)))
            else:
                img_array = np.load(f'{path_input_raw_data}/{file}')
                data = np.vstack((data, img_array))
    except:
        print(f'cannot open {file}')

assert data.shape[0] == len(files)

data.shape
data = data.reshape(data.shape[0], size_img[0], size_img[1])
data.shape


### transforing to nwb format

## this needs to be the time of the experiment that will be query from the json file
start_time = datetime(2018, 4, 25, 2, 30, 3, tzinfo=tz.gettz('US/Eastern'))

## setting up NWB file
nwbfile = NWBFile(session_description='Optopatch experiment 2020_03_02',
                  identifier='Mouse 1',
                  session_start_time=start_time,
                  session_id='Experiment 144',
                  experimenter='Jeremy Forest',
                  lab='Reyes Lab',
                  institution='New York University')
# print(nwbfile)

## Mice info
nwbfile.subject = Subject(subject_id='001',
                          age='P0',
                          description='Date Culture 1',
                          species='Optopatch-Cre Mice',
                          sex='U')
# print(nwbfile)

### do I needs trial data structure ??

## Setup info
device_1 = nwbfile.create_device(name='Microscope',
                               description='BX51WI',
                               manufacturer='Olympus')
device_2 = nwbfile.create_device(name='Camera',
                               description='C13440-20CU',
                               manufacturer='Hamamatsu')
device_3 = nwbfile.create_device(name='Dlp',
                               description='DLP-LightCrafter-DM365',
                               manufacturer='Texas Instrument')
device_4 = nwbfile.create_device(name='Laser',
                               description='',
                               manufacturer='')
device_5 = nwbfile.create_device(name='Controler',
                               description='SM5',
                               manufacturer='Luigs and Neumann')


optical_channel_1 = OpticalChannel(name='Cheriff-GFP',
                                 description='optogenetic stimulation',
                                 emission_lambda=000.)
optical_channel_2 = OpticalChannel(name='Quasar-2',
                                 description='voltage recording',
                                 emission_lambda=000.)

imaging_plane = nwbfile.create_imaging_plane(name='Culture',
                                             optical_channel=optical_channel_2,
                                             imaging_rate=400.,
                                             description='Voltage imaging',
                                             device=device_2,
                                             excitation_lambda=000.,
                                             indicator='Quasar-2',
                                             location='',
                                             grid_spacing=[.01, .01], ## how much µm is 1 pixel ?
                                             grid_spacing_unit='µm')


# using internal data. this data will be stored inside the NWB file as acquired data that is not-modifiable
# can add multiple series if wanted
image_series1 = TwoPhotonSeries(name='fluorescence data',
                                data=data,
                                imaging_plane=imaging_plane,
                                rate=400.0,
                                unit='pixel intensity')
nwbfile.add_acquisition(image_series1)


## Create a ProcessingModule to store the future processed data
ophys_module = nwbfile.create_processing_module(name='ophys',
                                                description='optical physiology processed data')

## ImageSegmentation object and add the ImageSegmentation to the ophy module.
img_seg = ImageSegmentation() ## imagesegmentation can contain multiple planesegmentation
ophys_module.add(img_seg)

## Create PlaneSegmentation tables within the targetted imaging_plane
ps = img_seg.create_plane_segmentation(name='ROIs',
                                       description='ROIs derived from manual contour of the images',
                                       imaging_plane=imaging_plane,
                                       reference_images=image_series1)


## will be easier down the line with image_mask
for roi in rois:
    # roi = rois[0]
    x_y_true = np.transpose(np.where(roi[1] == True))
    pixel_mask = []
    for _ in x_y_true:
        ix = x_y_true[0][0]
        iy = x_y_true[0][1]
        pixel_mask.append((ix, iy, 1))
    # add pixel mask to plane segmentation
    ps.add_roi(pixel_mask=pixel_mask)

## check if right number of rois has been stored
assert len(ps.to_dataframe().index) == len(rois)


# Create a DynamicTableRegion that references the ROIs of the PlaneSegmentation table.
rt_region = ps.create_roi_table_region(region=list(range(len(rois))),
                                        description='rois')



## Store the data of each the ROIs in their RoiResponseSeries

## first get the roi data signal
rois_data = rois_signal(data=data, rois_masks=rois)
rois_data_mean = [rois_data[i].mean(axis=1) for i in range(len(rois_data))]
[rois_data_mean[i].shape for i in range(len(rois_data_mean))]

rois_data_mean = np.stack(rois_data_mean)




## Then create RoiResponseSeries to hold the data of those ROIs
roi_resp_series = RoiResponseSeries(name='RoiResponseSeries',
                                    data=rois_data_mean,
                                    rois=rt_region,
                                    unit='intensity',
                                    rate=400.)




fl = Fluorescence(roi_response_series=roi_resp_series)
ophys_module.add(fl)

# df_over_f = DfOverF(roi_response_series=roi_resp_series)
# ophys_module.add(df_over_f)


with NWBHDF5IO('sample_data.nwb', 'w') as io:
    io.write(nwbfile)
