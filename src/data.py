import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress a deprecation warning in pandas.
import copy
import numpy as np
import os
import pandas as pd
from typing import List, Tuple, Union


# Intake a path, where the header starts (None if there is no header), the header for the class labels, and those headers that should be removed from the returned 
# dataset. The data is processed and returned as two numpy arrays (data and labels).
def data_read_and_format(directory_path: str, header_line: Union[int, None], label_header: Union[str, None], remove_headers: Union[List[str], None]) -> Tuple[np.ndarray, np.ndarray]:
    # Verify that the path exists.
    try:
        file_name = os.listdir(directory_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{os.path.basename(os.path.normpath(directory_path))}'s folder is not located in the following path from the present working directory: ./../../../440-project-data/") from None
    
    # This processing function is designed to work on a specific data layout (only one tabular file). Raise an error if it is not formatted in this manner.
    if len(file_name) > 1:
        raise ValueError(f"There is more than one file located in the '{os.path.basename(os.path.normpath(directory_path))}' directory. Please make sure that all data is merged into one file.")
    
    # Read in the file according to its extension type. If it is not one of the 4 options listed below, raise an error saying that this function does not support 
    # that type of file.
    file_ext = os.path.splitext(file_name[0])[-1]
    if (file_ext == '.csv') or (file_ext == '.txt') or (file_ext == '.all-data'):
        data = pd.read_csv(directory_path + file_name[0], header=header_line)
    elif file_ext == '.xlsx':
        data = pd.read_excel(directory_path + file_name[0], header=header_line)
    else:
        raise TypeError(f"The '{file_ext}' file extension is not supported by this data pre-processing function.")
    
    # Drop columns where there is missing data.
    data = data.dropna(axis=1)

    # Ordinally encode the "object" type features.
    for col_name, _ in data.iteritems():
        if data[col_name].dtype == 'O':
            data[col_name] = pd.factorize(data[col_name])[0]

    # If there is a header, remove the features specified and create the data and labels as numpy arrays. If there is not a header, it is assumed that the labels are 
    # the very last column of the input dataset, a warning is output letting the user know that they should verify that their data is formatted in this manner. Otherwise, 
    # no other columns are removed and they are converted into numpy arrays like in the other branch.
    if (header_line == 0) or (header_line):
        remove_headers.append(label_header)

        headers = data.columns.values.tolist()

        for header in remove_headers:
            headers.remove(header)

        X = data[headers].to_numpy()
        y = data[label_header].to_numpy()
    else:
        warnings.warn("If there are no headers in the file, then this function assumes that the labels are in the final column and everything before is data. Please verify that your dataset takes this form before moving forward.", Warning)
        num_cols = data.shape[1]

        headers = data.columns.values.tolist()

        headers.remove(num_cols - 1)

        X = data[headers].to_numpy()
        y = data[num_cols - 1].to_numpy()

    # Return the data and labels.
    return (X, y)


# Call the formatting function on the banknote authentication dataset and return it to the user.
def get_banknote_authentication() -> Tuple[np.ndarray, np.ndarray]:
    directory_path = r'./../../../440-project-data/banknote-authentication/'
    header_line = None
    label_header = None
    remove_headers = None
    return data_read_and_format(directory_path, header_line, label_header, remove_headers)


# Call the formatting function on the Brazilian COVID-19 dataset and return it to the user.
def get_brazil_covid() -> Tuple[np.ndarray, np.ndarray]:
    directory_path = r'./../../../440-project-data/brazil-covid/'
    header_line = 0
    label_header = "SARS-Cov-2 exam result"
    remove_headers = ["Patient ID"]
    return data_read_and_format(directory_path, header_line, label_header, remove_headers)


# Call the formatting function on the heart failure dataset and return it to the user.
def get_heart_failure() -> Tuple[np.ndarray, np.ndarray]:
    directory_path = r'./../../../440-project-data/heart-failure/'
    header_line = 0
    label_header = 'HeartDisease'
    remove_headers = []
    return data_read_and_format(directory_path, header_line, label_header, remove_headers)


# Call the formatting function on the Pima Indian diabetes dataset and return it to the user.
def get_pima_diabetes() -> Tuple[np.ndarray, np.ndarray]:
    directory_path = r'./../../../440-project-data/pima-diabetes/'
    header_line = 0
    label_header = 'Outcome'
    remove_headers = []
    return data_read_and_format(directory_path, header_line, label_header, remove_headers)


# Call the formatting function on the sonar mines dataset and return it to the user.
def get_sonar_mines() -> Tuple[np.ndarray, np.ndarray]:
    directory_path = r'./../../../440-project-data/sonar-mines/'
    header_line = None
    label_header = None
    remove_headers = None
    return data_read_and_format(directory_path, header_line, label_header, remove_headers)


# Call the formatting function on the Titanic survivors dataset and return it to the user.
def get_titanic_survivors() -> Tuple[np.ndarray, np.ndarray]:
    directory_path = r'./../../../440-project-data/titanic-survivors/'
    header_line = 0
    label_header = 'survived'
    remove_headers = []
    return data_read_and_format(directory_path, header_line, label_header, remove_headers)


# Call the formatting function on the wine quality dataset and return it to the user.
def get_wine_quality() -> Tuple[np.ndarray, np.ndarray]:
    directory_path = r'./../../../440-project-data/wine-quality/'
    header_line = 0
    label_header = 'quality'
    remove_headers = ['Id']
    return data_read_and_format(directory_path, header_line, label_header, remove_headers)


def cv_split(
        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False, replace: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    #     Conducts a cross-validation split on the given data. This can conduct BOTH stratified and non-stratified CV.

    #     Args:
    #         X: Data of shape (n_examples, n_features).
    #         y: Labels of shape (n_examples,).
    #         folds: Number of CV folds.
    #         stratified: If True, returns a stratified CV split, else a random set of splits.

    #     Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    #     for each fold. the tuple is of dimenesion (num_folds, 4).
    #     """

    # Control for the maximum number of folds in the data.
    if folds > len(y):
        raise ValueError("You cannot have more folds than data examples.")

    # Set the RNG seed to 12345 to ensure repeatability.
    np.random.seed(12345)

    split_sizes = [((len(y) // folds) + (len(y) % folds))]
    split_sizes += [(len(y) // folds)] * (folds - 1)

    if stratified:
        store_X = []
        store_y = []

        sort_indices = np.argsort(y)
        y = y[sort_indices]
        X = X[sort_indices]

        unique_elements, first_index, element_counts = np.unique(y[::-1], return_index=True, return_counts=True)
        first_index = np.abs(len(y) - first_index)

        lagging_index = 0
        for index in first_index:
            store_X.append(X[lagging_index:index].tolist())
            store_y.append(y[lagging_index:index].tolist())
            lagging_index = index

        probabilities = [(count / len(y)) for count in element_counts]

        data_folds = [[] for _ in range(folds)]
        label_folds = [[] for _ in range(folds)]

        if replace:
            for i in range(folds):
                while split_sizes[i] > 0:
                    temp_index = np.random.choice(range(len(element_counts)), p=probabilities)
                    temp_len = len(store_y[temp_index])
                    if temp_len != 0:
                        rand_index = np.random.randint(low=0, high=temp_len)
                        data_folds[i].append(store_X[temp_index][rand_index])
                        label_folds[i].append(store_y[temp_index][rand_index])
                        split_sizes[i] -= 1
        else:
            for i in range(folds):
                while split_sizes[i] > 0:
                    temp_index = np.random.choice(range(len(element_counts)), p=probabilities)
                    temp_len = len(store_y[temp_index])
                    if temp_len != 0:
                        rand_index = np.random.randint(low=0, high=temp_len)
                        data_folds[i].append(store_X[temp_index].pop(rand_index))
                        label_folds[i].append(store_y[temp_index].pop(rand_index))
                        split_sizes[i] -= 1

    else:
        if replace:
            temp_len = len(y)
            rand_indices = [np.random.randint(low=0, high=temp_len) for _ in range(temp_len)]
            rand_data = X[rand_indices]
            rand_labels = y[rand_indices]

            data_folds = np.array_split(rand_data, folds)
            label_folds = np.array_split(rand_labels, folds)
        else:
            # Create a permutation of the range [1..len(class_labels)] to randomly order the data and labels.
            rand_permutation = np.random.permutation(y.size)
            rand_data = X[rand_permutation]
            rand_labels = y[rand_permutation]

            # Split the data into their respective folds.
            data_folds = np.array_split(rand_data, folds)
            label_folds = np.array_split(rand_labels, folds)

    # Create the tuple to hold the CV datasets.
    cv_folds = ()

    # Iterate through each fold and build them in the tuple.
    for fold in range(folds):
        # Deep copy of the lists so that we can use them during each construction.
        temp_data = copy.deepcopy(data_folds)
        temp_labels = copy.deepcopy(label_folds)

        # Remove the validation sets.
        data_val = temp_data.pop(fold)
        label_val = temp_labels.pop(fold)

        # Add the elements in tuple form to the encompassing tuple.
        cv_folds = cv_folds + ((np.concatenate(temp_data), np.concatenate(temp_labels), data_val, label_val), )

    return cv_folds