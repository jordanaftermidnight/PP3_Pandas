# PP3 Pandas Project Setup
# Author: George Dorochov
# Email: jordanaftermidnight@gmail.com
# Description: Creating directory structure for PP3 Pandas project

import os
import urllib.request

def download_notebook():
    """Download the PP3 Pandas notebook from GitHub"""
    url = "https://raw.githubusercontent.com/MindaugasBernatavicius/DeepLearningCourse/master/03_Pandas/PP3_Pandas.ipynb"
    filename = "PP3_Pandas.ipynb"
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading notebook: {e}")

if __name__ == "__main__":
    download_notebook()