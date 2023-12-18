from .TalbeModificatorOperationsEnum import TalbeModificatorOperationsEnum
from .TableModificatorItem import TableModificatorItem


# Class with the configuration of a dataframe alteration operation.
# Internally, it creates a copy to perform the alterations.
class TalbeModificatorScheme:
    mods = []

    # Set a strategy for a column alteration
    def add(self, columnName: str, operation: TalbeModificatorOperationsEnum, options={}):
        ti = TableModificatorItem(columnName, operation, options)
        self.mods.append(ti)

    def getOperations(self):
        return self.mods
