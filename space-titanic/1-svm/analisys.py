import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from il.id.ImputeDesigner import ImputeDesigner
from il.id.IDOperations import IDOperations
from il.id.ImputeDesignerExecutor import ImputeDesignerExecutor


# =========================================================
# =========================================================
# =========================================================
# =========================================================


# Function to list the artifacts
def listArtifacts(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


# Load csv into a DataFrame (pandas)
def loadDataFrame(file_path):
    # Load a dataset into a Pandas Dataframe
    return pd.read_csv(file_path)


# Cleans the table. Various operations are performed here to prepare the table for training and prediction.
def manipulateTable(dataset, testing=False):
    # Function to crete new columns about total spent
    def fn(df): return df['RoomService'] + df['FoodCourt'] + \
        df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    # Columns to set to zero when the traveler is on Cryo
    columns_to_zero = ['TotalSpent', 'RoomService',
                       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Design table modifications
    tms = ImputeDesigner([
        # 1. Dropping PasengerId and Name
        ['PassengerId', IDOperations.DROP],
        ['Name', IDOperations.DROP],
        # 2. Fill CryoSleep nan with Unknown and apply One Hot
        ['CryoSleep', IDOperations.SEQUENCE, {
            'sequence': [
                ['CryoSleep', IDOperations.TO_STR],
                ['CryoSleep', IDOperations.REPLACE, {
                    'match': 'nan',
                    'value': 'Unknown'}],
                ['CryoSleep', IDOperations.ONE_HOT],
            ]
        }],
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
    isTesting = 'test' if testing else None
    tme = ImputeDesignerExecutor(tms)
    return tme.execute(dataset, filter=isTesting)


# Execute Cross Validation with a given pipeline and values
def executeCrossValidation(pipeline, X, y, cv=10, scoring='accuracy'):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

    print('---')
    print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std() * 2))
    print('---')


# Execute GridSearchCV to optimize Hiperparams
def executeHiperParamethersSearch(pipeline, X, y):
    # GridSearchCV
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],          # Valores comuns para C
        'svm__gamma': [1, 0.1, 0.01, 0.001],  # Valores comuns para gamma
        # Tipos de kernel (removed poly because of the time taken for this exercise)
        'svm__kernel': ['rbf', 'sigmoid']
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5,
                               scoring='accuracy', verbose=2)
    grid_search.fit(X, y)

    print('---')
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)
    print('---')


# Function to train the model
def trainingTime(pipeline, X, y):
    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Model training
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculation of confusion matrix and classification report
    conf_mat = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # Evaluate Model
    evaluateModel(pipeline, X_train, X_test, y_train, y_test)

    # Uncomment this line to se amazing graphs and data
    # printResults(conf_mat, cr)
    return pipeline


# Evaluate given model
def evaluateModel(pipeline, X_train, X_test, y_train, y_test):
    # Evaluating the model on the training set.
    train_accuracy = pipeline.score(X_train, y_train)
    print(f"Precisão no Treino: {train_accuracy}")

    # "Evaluating the model on the test set.
    test_accuracy = pipeline.score(X_test, y_test)
    print(f"Precisão no Teste: {test_accuracy}")

    # I set myself a threshould of 2,5% for my work
    threshould = 0.025

    # Check for possible overfit
    if train_accuracy > test_accuracy + threshould:
        print("Possível overfitting detectado.")


# Function to display the results on the screen and in graphs
def printResults(conf_mat, cr):
    print('===')

    df_cm = pd.DataFrame(conf_mat)

    # Criar um mapa de calor
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

    report_df = pd.DataFrame(cr).transpose()
    report_df = report_df.drop('support', axis=1)
    # Imprimindo a tabela
    print(report_df)

    # Explicando as métricas
    print("\nExplanations of the Metrics:")
    print("Accuracy: Proportion of correct positive identifications (the higher, the better).")
    print("F1-score: Harmonic mean of precision and recall (the higher, the better).")

    print('===')


# Function to test the model, cross-referencing it with a test dataset.
# This test dataset does not contain the result we want to predict for verification.
def testingTime(pipeline, testpath):
    test_df = loadDataFrame(testpath)
    submission_id = test_df.PassengerId
    # Clean test Table to same standards as the validation table
    test_df = manipulateTable(test_df, testing=True)
    # ----------

    # Predict
    predictions = pipeline.predict(test_df)
    # Convert to 0 or 1
    n_predictions = (predictions > 0.5).astype(bool)

    # Create a new DataFrame only with passangers ID and the prediction result
    output = pd.DataFrame({'PassengerId': submission_id,
                          'Transported': n_predictions.squeeze()})

    return (n_predictions, output)


# Generates a prediction file to submit to the challenge.
def generateSubmissionFile(n_predictions, writePath):
    sample_submission_df = pd.read_csv(f'{data_path}/sample_submission.csv')
    sample_submission_df['Transported'] = n_predictions
    sample_submission_df.to_csv(f'{writePath}/submission.csv', index=False)
    print(sample_submission_df.head())
# =========================================================
# =========================================================
# =========================================================
# =========================================================


# Data path (change it for other locations)
data_path = './data'
working_path = './'

# List artifacts
listArtifacts(data_path)
# Load data
dataset_df = loadDataFrame(f'{data_path}/train.csv')
# ----------

# ----------
# Print table info before modifitaion
print(dataset_df.head(5))
# Check missing values and print it
print(dataset_df.isnull().sum().sort_values(ascending=False))
# Clean Data Table (Modification and preparation of the data)
dataset_df = manipulateTable(dataset_df)
# Print modified table
print(dataset_df.head())
print(dataset_df.describe())
# check missing values
print(dataset_df.isnull().sum().sort_values(ascending=False))
print("Final train dataset shape is {}".format(dataset_df.shape))
# ----------

# ----------
# Creates the X and y for training
X = dataset_df.drop("Transported", axis=1)
y = dataset_df["Transported"]
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
# Accuracy and Hiperparams search
# I tested with various cv values, 10 gave a good result
# Uncomment for Cross Validation and HP Search
executeCrossValidation(pipeline, X, y, cv=10, scoring='accuracy')
# Take care, this operation can take a long time to process
# executeHiperParamethersSearch(pipeline, X, y)
# ----------

# ----------
# Trains the model
pipeline = trainingTime(pipeline, X, y)
# ----------

# ----------
# Submission - Testing Time
(n_predictions, output) = testingTime(pipeline, f'{data_path}/test.csv')
# print(output.head())
# ----------
# Generate Submission File
# generateSubmissionFile(n_predictions, working_path)
# ----------
