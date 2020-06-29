from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

"""

"""

def plot_confusion_matrix(model, X_test, y_test):
    #  Colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'
    y_predicted = model.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_predicted)
    #  roc Area Under Curve is a good way to evaluate binary classification
    roc_auc = auc(false_positive_rate, true_positive_rate)

    print("False Positive Rate: ", RED + "{:.5}%".format(false_positive_rate[1] * 100) + END)
    print("True Positive Rate: ", GREEN + "{:.5}%".format(true_positive_rate[1] * 100) + END)
    print()
    print("Area Under the Receiver Operating Characteristics: " + GREEN + "{:.5}%".format(roc_auc * 100) + END)
    print()
    cm = confusion_matrix(y_test, y_predicted)
    print("Confusion Matrix for Test Set: \n" + str(cm))
    #  Heat map
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def plot_n_estimators(X_train, y_train, X_test, y_test):
    # Identifying the ideal number of N_estimators
    n_estimators = [11,20,30,50,102]
    train_results = []
    test_results = []
    for estimator in n_estimators:
        rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)

        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")

    line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('n_estimators')
    plt.show()


def plot_max_depth(X_train, y_train, X_test, y_test):
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:
        rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1, n_estimators=10)
        rf.fit(X_train, y_train)
        train_pred = rf.predict(X_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        train_results.append(roc_auc)
        y_pred = rf.predict(X_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
    line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()


def run_metrics(model, X_train, X_test, y_train, y_test, in_sample_accuracy, test_accuracy):

    #  Colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

    print()

    #  18 features
    features = ["ProductName","EngineVersion","AppVersion","AvSigVersion","IsBeta","RtpStateBitfield",
                "IsSxsPassiveMode","DefaultBrowsersIdentifier","AVProductStatesIdentifier","AVProductsInstalled",
                "AVProductsEnabled","HasTpm","CountryIdentifier","CityIdentifier","OrganizationIdentifier",
                "GeoNameIdentifier","LocaleEnglishNameIdentifier","Platform","Processor","OsVer","OsBuild","OsSuite",
                "OsPlatformSubRelease","OsBuildLab","SkuEdition","IsProtected","AutoSampleOptIn","PuaMode","SMode",
                "IeVerIdentifier","SmartScreen","Firewall","UacLuaenable","Census_MDC2FormFactor","Census_DeviceFamily",
                "Census_OEMNameIdentifier","Census_OEMModelIdentifier","Census_ProcessorCoreCount",
                "Census_ProcessorManufacturerIdentifier","Census_ProcessorModelIdentifier","Census_ProcessorClass",
                "Census_PrimaryDiskTotalCapacity","Census_PrimaryDiskTypeName","Census_SystemVolumeTotalCapacity",
                "Census_HasOpticalDiskDrive","Census_TotalPhysicalRAM","Census_ChassisTypeName",
                "Census_InternalPrimaryDiagonalDisplaySizeInInches","Census_InternalPrimaryDisplayResolutionHorizontal",
                "Census_InternalPrimaryDisplayResolutionVertical","Census_PowerPlatformRoleName",
                "Census_InternalBatteryType","Census_InternalBatteryNumberOfCharges","Census_OSVersion",
                "Census_OSArchitecture","Census_OSBranch","Census_OSBuildNumber","Census_OSBuildRevision",
                "Census_OSEdition","Census_OSSkuName","Census_OSInstallTypeName","Census_OSInstallLanguageIdentifier",
                "Census_OSUILocaleIdentifier","Census_OSWUAutoUpdateOptionsName","Census_IsPortableOperatingSystem",
                "Census_GenuineStateName","Census_ActivationChannel","Census_IsFlightingInternal",
                "Census_IsFlightsDisabled","Census_FlightRing","Census_ThresholdOptIn",
                "Census_FirmwareManufacturerIdentifier","Census_FirmwareVersionIdentifier","Census_IsSecureBootEnabled",
                "Census_IsWIMBootEnabled","Census_IsVirtualDevice","Census_IsTouchEnabled","Census_IsPenCapable",
                "Census_IsAlwaysOnAlwaysConnectedCapable","Wdft_IsGamer","Wdft_RegionIdentifier"]

    #  Metrics
    print("Model Hyper-parameters:\n", model)
    print()

    print("Model feature importance:")
    for idx, importance in enumerate(model.feature_importances_, start=0):
        if importance >= 0.05:
            print(GREEN + "{:20}\t\t\t\t\t\t{:.10f}".format(features[idx], importance) + END)
        else:
            print("{:20}\t\t\t\t\t\t{:.10f}".format(features[idx], importance))
    print()

    print("Overall In-Sample Accuracy: \t\t" + BLUE + "{:.5}%".format(in_sample_accuracy*100) + END)
    print("Overall Test Accuracy: \t\t\t\t" + BLUE + "{:.5}%".format(test_accuracy*100) + END)
    print()
    print("Number of samples in X_train:\t{:,}".format(len(X_train)))
    print("Number of samples in y_train:\t{:,}".format(len(y_train)))
    print("Number of samples in X_test:\t{:,}".format(len(X_test)))
    print("Number of samples in y_test:\t{:,}".format(len(y_test)))
    print()


