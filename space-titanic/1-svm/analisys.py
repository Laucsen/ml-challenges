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


# Class for manipulating the DataFrame with inline operations
# The constructor accepts two attributes.
# First, the DataFrame to be modified, and then the name of the column.
# Each instance of the class is responsible for modifying only one column of the table.
class TalbeModificator:
    def __init__(self, datafame, columnName):
        self.datafame = datafame
        self.columnName = columnName

    # Encode a colum to binary (0 or 1)
    def encodeToBinary(self, positive_label):
        self.datafame[self.columnName] = self.datafame[self.columnName].apply(
            lambda x: 1 if x == positive_label else 0)
        return self

    # One Hot encode. Creates a new column for each value and then deletes the original column.
    def oneHot(self):
        dummies = pd.get_dummies(
            self.datafame[self.columnName], prefix=self.columnName)
        dummies = dummies.astype(int)

        self.datafame = pd.concat(
            [self.datafame.drop(self.columnName, axis=1), dummies], axis=1)
        return self

    # Fill nan with mode
    def fillnaWithMode(self):
        moda = self.datafame[self.columnName].mode()[0]
        self.datafame[self.columnName].fillna(moda, inplace=True)
        return self

    # Fill nan with median value from specified column
    def fillnaWithMedian(self):
        col_median = self.datafame[self.columnName].median()
        self.datafame[self.columnName].fillna(col_median, inplace=True)
        return self

    # Fill nana with zeroes
    def fillnaWithZeroes(self):
        self.datafame[self.columnName].fillna(0, inplace=True)
        return self

    # Fill nan with specified value
    def fillnaWith(self, value):
        self.datafame[self.columnName].fillna(value, inplace=True)
        return self

    # Replace specified value (valueToReplace) with valueToSet
    def replaceWithValue(self, valueToReplace, valueToSet):
        self.datafame[self.columnName] = self.datafame[self.columnName].replace(
            valueToReplace, valueToSet)
        return self

    # Convert a column to int
    def toInt(self):
        self.datafame[self.columnName] = self.datafame[self.columnName].astype(
            int)
        return self

    # Convert a column to string
    def toStr(self):
        self.datafame[self.columnName] = self.datafame[self.columnName].astype(
            str)
        return self

    # Splits a column according to a pattern. The pattern should contain the token to be split,
    # and newColumns should contain an array with the names of the new columns.
    def split(self, pattern, newColumns):
        self.datafame[newColumns] = self.datafame[self.columnName].str.split(
            pattern, expand=True)
        self.datafame = self.datafame.drop(self.columnName, axis=1)
        return self

    # Creates a new column with the name provided in the constructor. Func should be a lambda that returns the value for each column for each row.
    def createNewColumn(self, func):
        self.datafame[self.columnName] = func(self.datafame)
        return self

    # Returns the modified DataFrame.
    def get(self):
        return self.datafame


# Cleans the table. Various operations are performed here to prepare the table for training and prediction.
def cleanTable(dataset, testing=False):
    # Preparing the dataset
    # Dropping PasengerId and Name
    dataset = dataset.drop(['PassengerId', 'Name'], axis=1)
    print(dataset.head(5))
    # Check missing values and print it
    print(dataset.isnull().sum().sort_values(ascending=False))
    # ----------

    # Fill CryoSleep with more frequent and convert to -0 or 1
    dataset = TalbeModificator(dataset, 'CryoSleep').toStr(
    ).replaceWithValue('nan', 'Unknown').oneHot().get()
    # Same with VIP
    dataset = TalbeModificator(dataset, 'VIP').fillnaWithMode().get()

    # Convedrt Transported to 0 or 1
    if not testing:
        # This column is not present when training
        dataset = TalbeModificator(dataset, 'Transported').toInt().get()

    # Split cabin and drop old column
    dataset = TalbeModificator(dataset, 'Cabin').split(
        '/', ["Deck", "Cabin_num", "Side"]).get()

    # Fill nan for: (replace with Mode)
    dataset = TalbeModificator(dataset, 'FoodCourt').fillnaWithMode().get()
    dataset = TalbeModificator(dataset, 'ShoppingMall').fillnaWithMode().get()
    dataset = TalbeModificator(dataset, 'Spa').fillnaWithMode().get()
    dataset = TalbeModificator(dataset, 'VRDeck').fillnaWithMode().get()
    dataset = TalbeModificator(dataset, 'RoomService').fillnaWithMode().get()

    # Filling nan with Unknown and apply One Hot
    dataset = TalbeModificator(dataset, 'HomePlanet').fillnaWith(
        'Unknown').oneHot().get()
    dataset = TalbeModificator(dataset, 'Destination').fillnaWith(
        'Unknown').oneHot().get()

    # Fill nan with Mode and apply One Hot
    dataset = TalbeModificator(dataset, 'Deck').fillnaWithMode().oneHot().get()
    # Fill nana with zeroes and convert to int
    dataset = TalbeModificator(
        dataset, 'Cabin_num').fillnaWithZeroes().toInt().get()
    # Encode to binary. P are tru and other are false.
    dataset = TalbeModificator(dataset, 'Side').encodeToBinary('P').get()

    # Fill age with median
    dataset = TalbeModificator(dataset, 'Age').fillnaWithMedian().get()

    # Create a new Feature, called TotalSpent
    dataset = TalbeModificator(dataset, 'TotalSpent').createNewColumn(lambda df: df['RoomService'] + df['FoodCourt'] +
                                                                      df['ShoppingMall'] + df['Spa'] + df['VRDeck']).get()

    # Lets supose that, people on cryo sleep will never spend money on stuff...
    columns_to_zero = ['TotalSpent', 'RoomService',
                       'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    dataset.loc[dataset['CryoSleep_True'] == 1, columns_to_zero] = 0

    # Print modified table
    print(dataset.head())
    print(dataset.describe())
    # check missing values
    print(dataset.isnull().sum().sort_values(ascending=False))
    print("Final train dataset shape is {}".format(dataset.shape))

    return dataset


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
# Clean Data Table (Modification and preparation of the data)
dataset_df = cleanTable(dataset_df)

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
(pipeline, conf_mat, cr) = trainingTime(pipeline, X, y)
# ----------

# ----------
# Submission - Testing Time
(n_predictions, output) = testingTime(pipeline, f'{data_path}/test.csv')
print(output.head())
# ----------
# Generate Submission File
generateSubmissionFile(n_predictions, working_path)
# ----------
