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
                self.append_design(idi)
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
                # Create new Item Design and appen to new slices
                idi = ImputeDesignerItem(
                    column_name, IDOperations.DROP, options, filters)
                nslice.add(idi)
            self.slices.extend(new_elements)

        elif operation == FEOperations.FILL_NAN_ALL_STRATEGIES:
            # TODO: check options values
            fill_value = options['fill_value']

            # Making 4 copies, one for each strategy
            all_cases = []

            # Create one copy of current slices for each case
            for i in range(5):
                current_block = []
                for slice in self.slices:
                    current_block.append(slice.copy())
                all_cases.append(current_block)

            # Clear slices
            self.slices = []

            strategy_options = [{
                'strategy': 'most_frequent'}, {
                'strategy': 'median'}, {
                'strategy': 'mean'}, {
                'strategy': 'constant',
                'fill_value': fill_value
            }]

            # For each case, add one strategy
            # For each new slice candidate
            for index in range(len(strategy_options)):
                slice_block = all_cases[index]
                for block in slice_block:
                    c_strategy = strategy_options[index]

                    s_column = column_name
                    s_operation = IDOperations.FILL_NAN
                    s_options = c_strategy
                    s_filters = []  # TODO: on the future, how to filter only one strategy?

                    idi_switch = ImputeDesignerItem(
                        s_column, s_operation, s_options, s_filters)
                    block.add(idi_switch)

                self.slices.extend(slice_block)

        elif operation == FEOperations.SWITCH:
            # TODO: check options values
            # Get witch options (more then 1)
            switch_options = options['options']

            all_cases = []

            # Create one copy of current slices for each case
            casesCount = len(switch_options)
            for i in range(casesCount):
                current_block = []
                for slice in self.slices:
                    current_block.append(slice.copy())
                all_cases.append(current_block)

            # Clear current Slices (we will add all back later)
            self.slices = []

            if len(all_cases) != len(switch_options):
                raise Exception(
                    'Internal Error: switch lenght and all cases must have the same size')

            # For each new slice candidate
            for index in range(len(switch_options)):
                slice_block = all_cases[index]
                for block in slice_block:
                    switch_option = switch_options[index]

                    s_column = switch_option[0]
                    s_operation = switch_option[1]
                    s_options = switch_option[2] if len(
                        switch_option) >= 3 else {}
                    s_filters = switch_option[3] if len(
                        switch_option) >= 4 else []

                    idi_switch = ImputeDesignerItem(
                        s_column, s_operation, s_options, s_filters)
                    block.add(idi_switch)

                self.slices.extend(slice_block)

        else:
            print(f'Error: operation not suported: {operation}')

    def append_design(self, idi):
        for i in range(0, len(self.slices)):
            des = self.slices[i]
            des.add(idi)

    def get_slices(self):
        return self.slices
