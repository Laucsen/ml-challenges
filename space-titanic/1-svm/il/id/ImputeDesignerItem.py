from .IDOperations import IDOperations


class ImputeDesignerItem:

    def __init__(self, columnName: str, operation: IDOperations, options={}):
        self.columnName = columnName
        self.operation = operation
        self.options = options

    def getColumnName(self):
        return self.columnName

    def getOperation(self):
        return self.operation

    def getOptions(self):
        return self.options
