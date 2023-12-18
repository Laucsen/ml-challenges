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

from tm.TalbeModificatorScheme import TalbeModificatorScheme
from tm.TalbeModificatorOperationsEnum import TalbeModificatorOperationsEnum
from tm.TalbeModificatorExecution import TalbeModificatorExecution


# =========================================================
# =========================================================
# =========================================================
# =========================================================

# Function to list the artifacts


def listArtifacts(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


# Load train data frame from csv
def loadDataFrame(train):
    # Load a dataset into a Pandas Dataframe
    dataset_df = pd.read_csv(train)
    print("Full train dataset shape is {}".format(dataset_df.shape))
    # Display the first 5 examples
    print(dataset_df.head(5))

    # Describe
    print(dataset_df.describe())

    return dataset_df


# Cleans the table. Various operations are performed here to prepare the table for training and prediction.
def manipulateTable(dataset, testing=False):
    tms = TalbeModificatorScheme()

    # Create a definition of what is to do
    # Dropping PasengerId and Name
    tms.add('PassengerId', TalbeModificatorOperationsEnum.DROP)
    tms.add('Name', TalbeModificatorOperationsEnum.DROP)
    # Fill CryoSleep nana with Unknown and apply One Hot
    tms.add('CryoSleep', TalbeModificatorOperationsEnum.TO_STR)
    tms.add('CryoSleep', TalbeModificatorOperationsEnum.REPLACE,
            {'match': 'nan',
             'value': 'Unknown'})
    tms.add('CryoSleep', TalbeModificatorOperationsEnum.ONE_HOT)
    # Fill empty values on VIP with mode
    tms.add('VIP', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'most_frequent'})
    tms.add('VIP', TalbeModificatorOperationsEnum.TO_INT)

    # TODO: think if is better to have an if like this, or a param on add to execute on test mode
    # Convedrt Transported to 0 or 1
    if not testing:
        # This column is not present when training
        tms.add('Transported', TalbeModificatorOperationsEnum.TO_INT)

    # Split Cabin and drop old column
    tms.add('Cabin', TalbeModificatorOperationsEnum.SPLIT_BY,
            {'pattern': '/',
             'new_columns': ['Deck', 'Cabin_num', 'Side']})
    tms.add('Cabin', TalbeModificatorOperationsEnum.DROP)

    # Work on newly created columns
    # Fill nan with Mode and apply One Hot
    tms.add('Deck', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'most_frequent'})
    tms.add('Deck', TalbeModificatorOperationsEnum.ONE_HOT)
    # Fill nana with zeroes and convert to int
    tms.add('Cabin_num', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'constant',
             'fill_value': 0})
    tms.add('Cabin_num', TalbeModificatorOperationsEnum.TO_INT)
    # Encode to binary. P are tru and other are false.
    tms.add('Side', TalbeModificatorOperationsEnum.ENCODE_TO_BINARY,
            {'positive_label': 'P'})

    # Fill nan for: (replace with Mode)
    tms.add('FoodCourt', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'most_frequent'})
    tms.add('ShoppingMall', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'most_frequent'})
    tms.add('Spa', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'most_frequent'})
    tms.add('VRDeck', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'most_frequent'})
    tms.add('RoomService', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'most_frequent'})

    # Filling nan with Unknown and apply One Hot
    tms.add('HomePlanet', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'constant',
             'fill_value': 'Unknown'})
    tms.add('HomePlanet', TalbeModificatorOperationsEnum.ONE_HOT)
    tms.add('Destination', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'constant',
             'fill_value': 'Unknown'})
    tms.add('Destination', TalbeModificatorOperationsEnum.ONE_HOT)

    # Fill age with median
    tms.add('Age', TalbeModificatorOperationsEnum.FILL_NAN,
            {'strategy': 'median'})

    # Create a new feature, calle TotalSpent, with all money spent by each traveler
    def fn(df): return df['RoomService'] + df['FoodCourt'] + \
        df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    tms.add('TotalSpent', TalbeModificatorOperationsEnum.CREATE_NEW_COLUMN,
            {'function': fn})

    # Lets supose that, people on cryo sleep will never spend money on stuff...
    columns_to_zero = ['TotalSpent', 'RoomService',
                       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    tms.add(columns_to_zero,
            TalbeModificatorOperationsEnum.CONDITIONAL_SET_TO,
            {'value': 0,
             'condition_col': 'CryoSleep_True',
             'condition_value': 1})

    # Execute the definition over the dataframe
    tme = TalbeModificatorExecution(tms)
    return tme.execute(dataset)


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
    return (pipeline, conf_mat, cr)


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
    test_df = cleanTable(test_df, testing=True)
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
# (pipeline, conf_mat, cr) = trainingTime(pipeline, X, y)
# ----------

# ----------
# Submission - Testing Time
# (n_predictions, output) = testingTime(pipeline, f'{data_path}/test.csv')
# print(output.head())
# ----------
# Generate Submission File
# generateSubmissionFile(n_predictions, working_path)
# ----------
