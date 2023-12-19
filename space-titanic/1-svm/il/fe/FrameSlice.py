import copy

from ..id.ImputeDesigner import ImputeDesigner


class FrameSlice:
    def __init__(self):
        self.design: ImputeDesigner = ImputeDesigner([])

    def add(self, design):
        self.design.add_design(design)

    def get_design(self):
        return self.design

    def copy(self):
        new_copy = FrameSlice()
        new_copy.design = copy.deepcopy(self.design)
        return new_copy
