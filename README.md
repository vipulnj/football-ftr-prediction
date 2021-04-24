# football-ftr-prediction

## Quick guide

- We first explore the data given to us which is available in `original_data` directory. This is covered in the `1_exploration.ipynb` notebook file.
- As part of the same file, we check for missing values and possible discrepancies in the data. To fill in the missing values, we have downloaded data from the internet and saved it in the `downloaded_data` directory. It contains the information for all the five leagues from the 2008-09 season to 2017-18 season. The script to download these CSVs is `0_downloadMoreData.py`. Once done, we use the `imputed_data` directory to save the imputed train and test dataframes for use in the next steps. 
- Finally, now that there are no null values, we perform feature-engineering on the train and test dataframes keeping in mind the models we might use. This is covered in `2_featureEngineering.ipynb`
- Further feature-engineering steps which are specific to the model we will be using is covered in the three `3_modelBuilding_*.ipynb` files.
- The predictons for each of these three models are saved in `predictions` directory.

More information about steps taken in each steps can be found in the Ipython notebook files. 