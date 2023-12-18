from .IDOperations import IDOperations
from .ImputeDesignerItem import ImputeDesignerItem


# Class with the configuration of a DataFrame alteration operation.
class ImputeDesigner:

    def __init__(self, designs) -> None:
        self.mods = []
        self.set(designs)

    # Set a design, that will be added to current designs and rules
    def set(self, designs):
        for design in designs:
            if len(design) == 2:
                ti = ImputeDesignerItem(design[0], design[1], {})
            elif len(design) == 3:
                ti = ImputeDesignerItem(design[0], design[1], design[2])
            elif len(design) == 4:
                ti = ImputeDesignerItem(
                    design[0], design[1], design[2], design[3])

            self.mods.append(ti)

    # Add a strategy for a column alteration
    def add(self, columnName: str, operation: IDOperations, options={}):
        ti = ImputeDesignerItem(columnName, operation, options)
        self.mods.append(ti)

    def getOperations(self):
        return self.mods
