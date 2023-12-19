import pandas as pd

from .ImputeDesigner import ImputeDesigner
from .IDOperations import IDOperations


class ImputeDesignerExecutor:
    def __init__(self, tfs: ImputeDesigner):
        self.tfs = tfs

    def apply_operation(self, dataframe: pd.DataFrame, idOperation: IDOperations, columnName: str, opOptions):
        if idOperation == IDOperations.DROP:
            dataframe = dataframe.drop(columnName, axis=1)
        elif idOperation == IDOperations.TO_STR:
            dataframe[columnName] = dataframe[columnName].astype(str)
        elif idOperation == IDOperations.TO_INT:
            dataframe[columnName] = dataframe[columnName].astype(int)
        elif idOperation == IDOperations.ENCODE_TO_BINARY:
            # TODO: check for right arguments
            positive_label = opOptions['positive_label']
            dataframe[columnName] = dataframe[columnName].apply(
                lambda x: 1 if x == positive_label else 0)
        elif idOperation == IDOperations.FILL_NAN:
            # TODO: check for right arguments
            strategy = opOptions['strategy']
            strategy_value = 0
            if strategy == 'most_frequent':
                strategy_value = dataframe[columnName].mode()[0]
            elif strategy == 'constant':
                strategy_value = opOptions['fill_value']
            elif strategy == 'median':
                strategy_value = dataframe[columnName].median()
            elif strategy == 'mean':
                strategy_value = dataframe[columnName].mean()
            else:
                print('ERROR: todo other strategies')
            dataframe[columnName].fillna(strategy_value, inplace=True)
        elif idOperation == IDOperations.REPLACE:
            # TODO: check for right arguments
            match = opOptions['match']
            value = opOptions['value']
            dataframe[columnName] = dataframe[columnName].replace(match, value)
        elif idOperation == IDOperations.SPLIT_BY:
            # TODO: check for right arguments
            pattern = opOptions['pattern']
            new_columns = opOptions['new_columns']
            dataframe[new_columns] = dataframe[columnName].str.split(
                pattern, expand=True)
        elif idOperation == IDOperations.ONE_HOT:
            dummies = pd.get_dummies(dataframe[columnName], prefix=columnName)
            dummies = dummies.astype(int)

            dataframe = pd.concat(
                [dataframe.drop(columnName, axis=1), dummies], axis=1)
        elif idOperation == IDOperations.CREATE_NEW_COLUMN:
            # TODO: check for right arguments
            function = opOptions['function']
            dataframe[columnName] = function(dataframe)
        elif idOperation == IDOperations.CONDITIONAL_SET_TO:
            # TODO: check for right arguments
            value = opOptions['value']
            condition_col = opOptions['condition_col']
            condition_value = opOptions['condition_value']
            dataframe.loc[dataframe[condition_col]
                          == condition_value, columnName] = value
        elif idOperation == IDOperations.SEQUENCE:
            # TODO: check for right arguments
            sequence = opOptions['sequence']
            for opt in sequence:
                extract_n_options = opt[2] if len(opt) >= 3 else {}
                dataframe = self.apply_operation(
                    dataframe, opt[1], opt[0], extract_n_options)
        else:
            raise Exception(f'Invalid IDOperation: {idOperation}')

        return dataframe

    def execute(self, dataframe: pd.DataFrame, filter=None):
        operations = self.tfs.getOperations()
        for op in operations:
            cop_name = op.getColumnName()
            cop_op = op.getOperation()
            cop_options = op.getOptions()
            cop_filters = op.getFilters()

            if filter:
                if filter in cop_filters:
                    continue

            dataframe = self.apply_operation(
                dataframe, cop_op, cop_name, cop_options)

        return dataframe
