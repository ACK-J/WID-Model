from sklearn.ensemble import RandomForestClassifier
from Metrics import *
import sklearn
import pandas as pd
import os
import pickle

"""
Author:         Jack Hyland
Date:           4/26/2020
Description:    This file uses a random-forest machine learning model to predict whether a Windows machine
                is infected with malware based on telemetry gathered by Windows Defender.
Data Source:    https://www.kaggle.com/c/microsoft-malware-prediction/data
Usage:          python3 WIDModel.py
"""

#  Colors
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'

def create_dataframe(DATA):
    """
    This function takes in a .csv file and normalizes it into a pandas dataframe that can be
    used by a random forest. It also splits the data into training and testing sets
    """
    TEST_SIZE = 0.2  # Percentage in decimal form for the test split (default 0.2)
    X = pd.read_csv(DATA, low_memory=False)
    #  Extract the labels and then remove them from the data
    y = list(X['HasDetections'])
    X = X.drop(['HasDetections'], axis='columns')
    #  Removes the machine ID's from both training and testing since it isn't needed
    X = X.drop(['MachineIdentifier'], axis='columns')
    #  OHE the dataset
    X = one_hot_encode(X)
    #  Split the data up traditionally into 80% training and 20% training
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE, shuffle=True)
    return X_train, X_test, y_train, y_test, X



def one_hot_encode(dataframe):
    """
    Since some of the features (columns) contain strings such as 'ProductName' we need to represent it as a number.
    After multiple different solutions I found that the best option is to create a hashmap, where the key is the string
    in the column and the value is a unique identifier such as 1,2,3... For example for the 'ProductName' column if the
    first records value was 'win8defender' we would check to see if it was already in the hash map, if true we would
    grab the unique ID attached to it and input that into the pandas dataframe. If we haven't seen 'win8defender' before
    we will add it to the hashmap and assign its value to be a unique ID as well as input that ID to the pandas df.
    :param dataframe: A pandas dataframe
    :return: A pandas dataframe that is OHE
    """
    #  This will grab all of the column names and store them in a list
    column_names = list(dataframe.columns)
    #  This is a list of all the column names which are only composed of numbers. This is used as a reference because
    #  if a column name is not in this list it probably contains strings which we will have to OHE
    dont_OHE = ['Census_IsPenCapable', 'OsBuild', 'Wdft_RegionIdentifier', 'Census_IsWIMBootEnabled',
                'Census_ProcessorManufacturerIdentifier', 'Census_IsFlightsDisabled', 'HasTpm',
                'AVProductStatesIdentifier', 'Census_OSUILocaleIdentifier', 'IsBeta',
                'Census_internalPrimaryDisplayResolutionVertical', 'AVProductsEnabled', 'SMode', 'Wdft_IsGamer',
                'Census_PrimaryDiskTotalCapacity', 'Census_IsAlwaysOnAlwaysConnectedCapable',
                'Census_IsPortableOperatingSystem', 'Census_OSBuildRevision', 'Census_OEMModelIdentifier',
                'Census_OEMNameIdentifier', 'IsSxsPassiveMode', 'Census_ThresholdOptIn', 'CountryIdentifier',
                'OsSuite', 'IeVerIdentifier', 'Census_OSInstallLanguageIdentifier',
                'Census_IsVirtualDevice', 'Census_OSBuildNumber', 'RtpStateBitfield', 'AutoSampleOptIn',
                'Census_ProcessorModelIdentifier', 'Census_IsSecureBootEnabled', 'Census_TotalPhysicalRAM',
                'UacLuaenable', 'OrganizationIdentifier', 'GeoNameIdentifier', 'Census_IsFlightinginternal',
                'Census_FirmwareManufacturerIdentifier', 'Census_HasOpticalDiskDrive',
                'LocaleEnglishNameIdentifier', 'AVProductsInstalled', 'Firewall', 'IsProtected',
                'Census_IsTouchEnabled', 'Census_FirmwareVersionIdentifier', 'CityIdentifier',
                'Census_internalPrimaryDisplayResolutionHorizontal', 'Census_SystemVolumeTotalCapacity',
                'Census_ProcessorCoreCount', 'DefaultBrowsersIdentifier', 'OsBuild']
    #  Loop over all the column names in the dataframe
    for el in column_names:
        #  If there is any Null entries in the column they will be replaced with a 0
        dataframe[el] = dataframe[el].fillna(0)
        #  If the column doesn't exist in the dont_OHE list
        if el not in dont_OHE:
            hashmap = {}
            num = 1
            #  Iterate over the size of the dataframe
            for i in range(len(dataframe)):
                #  if the value in the dataframe is not in the hashmap
                if dataframe[el].values[i] not in hashmap.keys():
                    #  Add the key value pair to the hash map
                    hashmap[dataframe[el].values[i]] = num
                    #  Set the current UID to be the value in the dataframe
                    dataframe[el].values[i] = num
                    #  Increment the UID
                    num+=1
                #  if the value in the dataframe is in the hashmap
                else:
                    # Lookup the UID and set it in the dataframe
                    dataframe[el].values[i] = hashmap[dataframe[el].values[i]]
    #  The old method
    # for column in column_names:
    #     dataframe = pd.concat([dataframe, pd.get_dummies(dataframe[column], prefix=column, dummy_na=True)],axis=1).drop([column], axis=1)
    return dataframe


if __name__ == '__main__':
    # Hyper-parameters for the Random Forest
    N_ESTIMATORS = 10              # The amount of trees (default 101 [keep at an odd number])
    MAX_DEPTH = 1000                  # Depth of the decision trees (default 10)
    RANDOM_STATE = 1                # Random seed (default 1)

    DATA = "data.csv"    #  Full path to the csv. \\CHANGE ME

    #  Checks to see if the data was serialized in the current directory
    if os.path.isfile("X_train.data") and os.path.isfile("X_test.data") \
        and os.path.isfile("y_train.data") and os.path.isfile("y_test.data")\
        and os.path.isfile("X.data"):

        print("Loading data...")
        with open("X_train.data", 'rb') as fp:
            X_train = pickle.load(fp)
        with open("X_test.data", 'rb') as fp:
            X_test = pickle.load(fp)
        with open("y_train.data", 'rb') as fp:
            y_train = pickle.load(fp)
        with open("y_test.data", 'rb') as fp:
            y_test = pickle.load(fp)
        with open("X.data", 'rb') as fp:
            X = pickle.load(fp)
        print("Loaded!")
    else:
        # Returns the csv as dataframes
        X_train, X_test, y_train, y_test, X = create_dataframe(DATA)

        #  Serializes the data so next time you run the program its faster
        pickle.dump(X_train, open("X_train.data", 'wb'), protocol=4)
        pickle.dump(X_test, open("X_test.data", 'wb'), protocol=4)
        pickle.dump(y_train, open("y_train.data", 'wb'), protocol=4)
        pickle.dump(y_test, open("y_test.data", 'wb'), protocol=4)
        pickle.dump(X, open("X.data", 'wb'), protocol=4)

    #  Create the classifier and train the model
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)

    #  Train the model
    in_sample_accuracy = model.fit(X_train, y_train).score(X_train, y_train)

    print("In Sample Accuracy:", in_sample_accuracy)
    test_accuracy = model.score(X_test, y_test)
    print("Test Accuracy:", test_accuracy)


    run_metrics(model, X_train, X_test, y_train, y_test, in_sample_accuracy, test_accuracy)
    print("\n" + RED + "CLOSE CONFUSION MATRIX PLOT WINDOW TO GO TO THE NEXT PLOT!" + END)
    plot_confusion_matrix(model, X_test, y_test)

    print("\n" + RED + "CLOSE N_ESTIMATORS PLOT WINDOW TO GO TO THE NEXT PLOT!" + END)
    plot_n_estimators(X_train, y_train, X_test, y_test)

    print("\n" + RED + "CLOSE MAX DEPTH PLOT WINDOW TO CLOSE THE PROGRAM!" + END)
    plot_max_depth(X_train, y_train, X_test, y_test)