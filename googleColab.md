# Google Colab

## Go to your Google drive
drive.google.com

## Go to any folder you want
cd {folder}

## Create a Google collaboratory
New -> More -> Google Collaboratory # a glorified Jupyter notebook

## Mount your Google drive so that you don't lose your results if the network gets disrupted
from google.colab import drive
drive.mount('/drive')

## You can now use the permanent drive just like any location
cd MyDrive/
cd {folder}

## You can now use your drive
with open ('/drive/MyDrive/{folder}/{fileName}') as f:
    f.write('Yay! New File!')

## Be sure to flush and unmount the drive when you are done
drive.flush_and_unmount()

