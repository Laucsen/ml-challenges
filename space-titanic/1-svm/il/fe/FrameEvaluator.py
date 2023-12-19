from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from ..id.ImputeDesignerExecutor import ImputeDesignerExecutor
from ..id.ImputeDesigner import ImputeDesigner

from .Frame import Frame
from .FrameEvaluatorReports import FrameEvaluatorReports
from .FrameEvaluatorReport import FrameEvaluatorReport


class FrameEvaluator:
    def __init__(self, frame: Frame, verbose=False, CV=None):
        self.report = FrameEvaluatorReports()

        self.frame = frame
        self.verbose = verbose
        self.CV = CV

    def evaluate(self, dataframe: DataFrame, pipeline, validationColumn: str):
        print('--------------------')
        print('Strating frame evaluation...')
        print('--------------------')

        # ----------
        # Print Original's Head table info before modifitaion
        if self.verbose:
            print('--------------------')
            print(dataframe.head(5))
            # Check missing values and print it
            print(dataframe.isnull().sum().sort_values(ascending=False))
            print('--------------------')
        # ----------

        # ----------
        # Get data from frame
        slices = self.frame.get_slices()
        for index, slice in enumerate(slices):
            design: ImputeDesigner = slice.get_design()

            # ----------
            # Apply current Slice configuration on dataframe
            # isTesting = 'test' if testing else None
            tme = ImputeDesignerExecutor(design)
            c_dataframe = tme.execute(dataframe.copy(), filter=False)

            # ----------
            # Print modified table heand a describe it (for current slice)
            if self.verbose:
                print('--------------------')
                print(c_dataframe.head())
                print(c_dataframe.describe())
                # check missing values
                print(c_dataframe.isnull().sum().sort_values(ascending=False))
                print('--------------------')
                print("Final train dataset shape is {}".format(c_dataframe.shape))
                print('--------------------')
            # ----------

            # ----------
            # Creates the X and y for training
            X = c_dataframe.drop(validationColumn, axis=1)
            y = c_dataframe[validationColumn]
            # ----------

            # ----------
            # Cross Validation over configured
            scores = cross_val_score(
                pipeline, X, y, cv=self.CV, scoring='accuracy')

            print(
                f'[Slice {index}] Cross-Validation Accuracy: {scores.mean()} (+/- {scores.std() * 2})')
            # ----------

            # ----------
            # Training Time
            # Split train and test data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3)
            # Model training
            pipeline.fit(X_train, y_train)
            # y_pred = pipeline.predict(X_test)
            # ----------

            # ----------
            # Evaluation
            # Evaluate Model
            # Evaluating the model on the training set.
            train_accuracy = pipeline.score(X_train, y_train)
            test_accuracy = pipeline.score(X_test, y_test)
            print(
                f'[Slice {index}] Train/Test accuracy: {train_accuracy} / {test_accuracy}')

            # I set myself a threshould of 2,5% for my work
            threshould = 0.025
            # Check for possible overfit
            if train_accuracy > test_accuracy + threshould:
                print(f'[Slice {index}] Possível overfitting detectado.')

            frer = FrameEvaluatorReport(
                index,
                slice,
                scores,
                train_accuracy,
                test_accuracy)
            self.report.add(frer)
            print('.')

        return self.report
