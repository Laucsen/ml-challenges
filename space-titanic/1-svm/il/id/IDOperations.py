import enum


# Definindo a enumeração
class IDOperations(enum.Enum):
    DROP = 1
    TO_STR = 2
    TO_INT = 3
    ENCODE_TO_BINARY = 4
    FILL_NAN = 5
    REPLACE = 6
    SPLIT_BY = 7
    ONE_HOT = 8
    CREATE_NEW_COLUMN = 9
    CONDITIONAL_SET_TO = 10
    SEQUENCE = 11
