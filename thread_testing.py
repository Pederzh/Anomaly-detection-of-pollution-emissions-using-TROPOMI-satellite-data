import datetime
import io

import math
import json
from pathlib import Path

import copy
import numpy
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm_notebook
from itertools import product
import scipy.ndimage
import scipy.fftpack
import scipy.signal
import scipy.stats as st
import numpy as np
from numpy import polyfit
from PIL import Image
import statistics as stst
from scipy.ndimage import median_filter, median, maximum_filter, gaussian_filter
from skimage.filters import threshold_otsu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import griddata
from numpy import array

import time
from threading import Thread


def test(message, sleep_duration):
    print(message)
    time.sleep(sleep_duration)
    print(str(sleep_duration) + " secondi passati")



t = Thread(target=test, args=("sono il thread", 10))
t.start()
print(" ")
print("sono già passato")