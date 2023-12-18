from .TalbeModificatorOperationsEnum import TalbeModificatorOperationsEnum


class TableModificatorItem:
    columnName: str = None
    operation: TalbeModificatorOperationsEnum = None
    options = {}

    def __init__(self, columnName: str, operation: TalbeModificatorOperationsEnum, options={}):
        self.columnName = columnName
        self.operation = operation
        self.options = options

    def getColumnName(self):
        return self.columnName

    def getOperation(self):
        return self.operation

    def getOptions(self):
        return self.options
