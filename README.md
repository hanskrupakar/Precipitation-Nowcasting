# Precipitation-Nowcasting

STEP 1: Download the Agrimet Dataset here: https://www.usbr.gov/pn-bin/instant.pl?station=FOGO&year=2014&month=1&day=1&year=2017&month=4&day=1&pcode=OB&pcode=OBX&pcode=OBM&pcode=OBN&pcode=TU&pcode=TUX&pcode=TUN&pcode=EA&pcode=TP&pcode=WD&pcode=WG&pcode=WS&pcode=UI&pcode=SQ&pcode=SI&pcode=PC&pcode=WDS

FOGO (Forest Grove Oregon Station) 1/1/2014 to 4/1/2017 all the values. Download and remove the title row such that it starts with the first row of values.

STEP 2: Download the NEXRAD Radar Level 3 images here: https://www.ncdc.noaa.gov/cdo-web/datatools/selectlocation

Select Portland, Oregon region and download the NEXRAD Level 3 data for the station for 2014 to March, 2017. Extract the directories present as tar files in the FTP into the dataset folder `RADAR DATA/Dataset`

STEP 3: Install all dependencies necessary

Python 2.7/3.x dependencies:
    
    Built-in:
        math
        re
    
    Installed:
        h5py
        NumPy
    
Lua Torch dependencies (http://torch.ch/docs/getting-started.html):

    Torch7 LuaRocks packages:
        
        gnuplot
        cutorch
        cunn
    
    Unofficial packages from other sources:    
        rnn - ElementResearch
        hdf5 - Twitter
        
Amount of disk space required:

    12MB for the CSV dataset file
    322MB for the hdf5 file created from dataset for torch
    1.6GB each for every checkpoint file created
    
STEP 4: Run the preprocessing script `data.py` for Text-only dataset extraction and `extract.py` both Agrimet and NEXRAD extraction dataset. 

Command Line Options:

Data Preprocessing

Options

    -batch_size        No of entries per batch
    -method		Preferred method of preprocessing (zscore/minmax)	
 
STEP 5: Run RNN.lua for training, checkpointing, testing and visualization using the same HDF5 file created during preprocessing.
 
Command Line Options:

Precipitation Nowcasting

Options

    -num_layers        No of hidden LSTM layers [4]
    -test                    Train/Test Flag [false]
    -iters                 No. of iterations on dataset [100]
    -batch_size        Batch size for BGD [32]
    -seqlen                No. of sequences of 15 min precipitation parameters (should be same as preprocessing data.py script) [24]
    -hidden_size     Hidden Layer Size [1000]
    -input_size        No. of parameters (15) [15]
    -learning_rate Learning rate for training [0.001]
    -output_size     Size of predicted output (1 - precipitation values) [1]
    -load_from         Checkpoint save file to load model from []
    -lr_decay            Learning Rate Decay [0.8]
    -decay_rate        Num epochs per every learning rate decay [3]
    -finetune            Finetune on large error batches to account for lesser # of precipitation values compared to 0 prec (0.83%) [false]
    -finetune_err    Error threshold to select finetune batches [0.005]
    -num_finetune    Number of times to finetune the data [2]
    
Example Invoke Command:

    $ th RNN.lua -finetune -finetune_err 1 -load_from minmax_full_4LSTMs_11.2000.t7 -decay_rate 1 -lr_decay 0.01 -num_finetune 3 >> minmax_log\(4LSTMs\,Norm\).txt 

