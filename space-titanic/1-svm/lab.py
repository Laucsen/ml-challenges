import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from il.id.IDOperations import IDOperations
from il.fe.FEOperations import FEOperations
from il.fe.Frame import Frame
from il.fe.FrameEvaluator import FrameEvaluator


# Function to list the artifacts
def listArtifacts(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


# Load csv into a DataFrame (pandas)
def loadDataFrame(file_path):
    # Load a dataset into a Pandas Dataframe
    return pd.read_csv(file_path)


# ----------
# Input Lab definition to test all the data frame possibilities
def evaluator(dataset, pipeline):
    # Function to crete new columns about total spent
    def fn(df): return df['RoomService'] + df['FoodCourt'] + \
        df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    # Columns to set to zero when the traveler is on Cryo
    columns_to_zero = ['TotalSpent', 'RoomService',
                       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    frame = Frame([
        # 1. Dropping PasengerId
        ['PassengerId', IDOperations.DROP],
        # 2. Teste with and without name column
        ['Name', FEOperations.DROP_NO_DROP],
        # 3. Fill CryoSleep nan with Unknown and apply One Hot
        ['CryoSleep', IDOperations.TO_STR],
        ['CryoSleep', IDOperations.REPLACE, {
            'match': 'nan',
            'value': 'Unknown'}],
        ['CryoSleep', IDOperations.ONE_HOT],
        # Fill empty values on VIP with mode
        ['VIP', IDOperations.FILL_NAN, {'strategy': 'most_frequent'}],
        ['VIP', IDOperations.TO_INT],
        # Transform Transported to Int, but not on test set
        # Passing empty options and the filter paramether
        ['Transported', IDOperations.TO_INT, {}, ['test']],
        # Split Cabin and drop old column
        ['Cabin', IDOperations.SPLIT_BY, {
            'pattern': '/',
            'new_columns': ['Deck', 'Cabin_num', 'Side']}],
        ['Cabin', IDOperations.DROP],
        # Work on newly created columns
        # Fill nan with Mode and apply One Hot
        ['Deck', IDOperations.FILL_NAN,
            {'strategy': 'most_frequent'}],
        ['Deck', IDOperations.ONE_HOT],
        # Fill nana with zeroes and convert to int
        ['Cabin_num', IDOperations.FILL_NAN,
            {'strategy': 'constant',
             'fill_value': 0}],
        ['Cabin_num', IDOperations.TO_INT],
        # Encode to binary. P are tru and other are false.
        ['Side', IDOperations.ENCODE_TO_BINARY,
            {'positive_label': 'P'}],
        # Fill nan for: (replace with Mode)
        ['FoodCourt', IDOperations.FILL_NAN,
            {'strategy': 'most_frequent'}],
        ['ShoppingMall', IDOperations.FILL_NAN,
            {'strategy': 'most_frequent'}],
        ['Spa', IDOperations.FILL_NAN,
            {'strategy': 'most_frequent'}],
        ['VRDeck', IDOperations.FILL_NAN,
            {'strategy': 'most_frequent'}],
        ['RoomService', IDOperations.FILL_NAN,
            {'strategy': 'most_frequent'}],
        # Filling nan with Unknown and apply One Hot
        ['HomePlanet', IDOperations.FILL_NAN,
            {'strategy': 'constant',
             'fill_value': 'Unknown'}],
        ['HomePlanet', IDOperations.ONE_HOT],
        ['Destination', IDOperations.FILL_NAN,
            {'strategy': 'constant',
             'fill_value': 'Unknown'}],
        ['Destination', IDOperations.ONE_HOT],
        # Fill age with median
        ['Age', IDOperations.FILL_NAN,
            {'strategy': 'median'}],
        # Create a new feature, calle TotalSpent, with all money spent by each traveler
        ['TotalSpent', IDOperations.CREATE_NEW_COLUMN, {'function': fn}],
        # Lets supose that, people on cryo sleep will never spend money on stuff...
        [columns_to_zero, IDOperations.CONDITIONAL_SET_TO,
            {'value': 0,
             'condition_col': 'CryoSleep_True',
             'condition_value': 1}],
    ])

    # Execute the definition over the dataframe
    fe = FrameEvaluator(frame, verbose=False, CV=10)
    return fe.evaluate(dataset, pipeline, 'Transported')
# ----------


# ----------
# Load data
# ----------
# Data path (change it for other locations)
data_path = './data'
working_path = './'

# List artifacts
listArtifacts(data_path)
# Load data
dataset_df = loadDataFrame(f'{data_path}/train.csv')
# ----------

# ----------
# Creating a pipeline with model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', gamma=0.01, C=10)),
    # ('mlp', MLPClassifier(max_iter=500, verbose=True))
])
# ----------

# ----------
# Inpute Lab To Work on loaded table
reports = evaluator(dataset_df, pipeline)
# ----------

# ----------
# Check best results and get best slice configuration
# TODO
print('...')
reports.print()
# ----------
