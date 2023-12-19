from ..id.IDOperations import IDOperations
from ..id.ImputeDesignerItem import ImputeDesignerItem

from .FEOperations import FEOperations
from .FrameSlice import FrameSlice


class Frame:
    def __init__(self, designs):
        self.slices: FrameSlice = []
        # Append an initial empty FrameSlice
        self.slices.append(FrameSlice())
        self.set(designs)

    def set(self, designs):
        for design in designs:
            column_name = design[0]
            operation = design[1]

            options = design[2] if len(design) >= 3 else {}
            filters = design[3] if len(design) >= 4 else []

            # Check if it is an ID or FD Operation
            if isinstance(operation, IDOperations):
                # Normal Operation
                idi = ImputeDesignerItem(
                    column_name, operation, options, filters)
                self.append_desig(idi)
            elif isinstance(operation, FEOperations):
                # Slice Operation
                self.create_slice_operation(
                    column_name, operation, options, filters)
            else:
                print(f'Error: operation not supported: {operation}')

    def create_slice_operation(self, column_name, operation: FEOperations, options, filters):
        if operation == FEOperations.DROP_NO_DROP:
            new_elements = []
            # Copy each Slice
            for slice in self.slices:
                new_elements.append(slice.copy())
            # DROP_NO_DROP we just add DROP IDOperation to newly copied elements
            for nslice in new_elements:
                idi = ImputeDesignerItem(
                    column_name, IDOperations.DROP, options, filters)
                self.append_desig(idi)
                nslice.add(idi)
            self.slices.extend(new_elements)
        else:
            print(f'Error: operation not suported: {operation}')

    def append_desig(self, idi):
        for i in range(0, len(self.slices)):
            des = self.slices[i]
            des.add(idi)

    def append_slice(self):
        # TODO
        pass

    def get_slices(self):
        return self.slices
