import pandas as pd

from .ImputeDesigner import ImputeDesigner
from .IDOperations import IDOperations


class ImputeDesignerExecutor:
    def __init__(self, tfs: ImputeDesigner):
        self.tfs = tfs

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

            if cop_op == IDOperations.DROP:
                dataframe = dataframe.drop(cop_name, axis=1)
            elif cop_op == IDOperations.TO_STR:
                dataframe[cop_name] = dataframe[cop_name].astype(str)
            elif cop_op == IDOperations.TO_INT:
                dataframe[cop_name] = dataframe[cop_name].astype(int)
            elif cop_op == IDOperations.ENCODE_TO_BINARY:
                # TODO: check for right arguments
                positive_label = cop_options['positive_label']
                dataframe[cop_name] = dataframe[cop_name].apply(
                    lambda x: 1 if x == positive_label else 0)
            elif cop_op == IDOperations.FILL_NAN:
                # TODO: check for right arguments
                strategy = cop_options['strategy']
                strategy_value = 0
                if strategy == 'most_frequent':
                    strategy_value = dataframe[cop_name].mode()[0]
                elif strategy == 'constant':
                    strategy_value = cop_options['fill_value']
                elif strategy == 'median':
                    strategy_value = dataframe[cop_name].median()
                else:
                    print('ERROR: todo other strategies')
                dataframe[cop_name].fillna(strategy_value, inplace=True)
            elif cop_op == IDOperations.REPLACE:
                # TODO: check for right arguments
                match = cop_options['match']
                value = cop_options['value']
                dataframe[cop_name] = dataframe[cop_name].replace(match, value)
            elif cop_op == IDOperations.SPLIT_BY:
                # TODO: check for right arguments
                pattern = cop_options['pattern']
                new_columns = cop_options['new_columns']
                dataframe[new_columns] = dataframe[cop_name].str.split(
                    pattern, expand=True)
            elif cop_op == IDOperations.ONE_HOT:
                dummies = pd.get_dummies(dataframe[cop_name], prefix=cop_name)
                dummies = dummies.astype(int)

                dataframe = pd.concat(
                    [dataframe.drop(cop_name, axis=1), dummies], axis=1)
            elif cop_op == IDOperations.CREATE_NEW_COLUMN:
                # TODO: check for right arguments
                function = cop_options['function']
                dataframe[cop_name] = function(dataframe)
            elif cop_op == IDOperations.CONDITIONAL_SET_TO:
                # TODO: check for right arguments
                value = cop_options['value']
                condition_col = cop_options['condition_col']
                condition_value = cop_options['condition_value']
                dataframe.loc[dataframe[condition_col]
                              == condition_value, cop_name] = value
            else:
                print('ERROR')

        return dataframe
