from .IDOperations import IDOperations
from .ImputeDesignerItem import ImputeDesignerItem


# Class with the configuration of a DataFrame alteration operation.
class ImputeDesigner:

    def __init__(self) -> None:
        self.mods = []

    # Set a strategy for a column alteration
    def add(self, columnName: str, operation: IDOperations, options={}):
        ti = ImputeDesignerItem(columnName, operation, options)
        self.mods.append(ti)

    def getOperations(self):
        return self.mods
