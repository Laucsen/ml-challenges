from .IDOperations import IDOperations


class ImputeDesignerItem:

    def __init__(self, columnName: str, operation: IDOperations, options={}, filters=[]):
        self.columnName = columnName
        self.operation = operation
        self.options = options
        self.filters = filters

    def getColumnName(self):
        return self.columnName

    def getOperation(self):
        return self.operation

    def getOptions(self):
        return self.options

    def getFilters(self):
        return self.filters
