"""
A transform which takes a string or list of strings
and optionally converts it to lower case and replaces 
underscores with spaces 
"""

class LowerAndReplace_Transform:
    def __init__(self, replace_=False, lower=True) -> None:
        self.replace_ = replace_
        self.lower = lower

    def process_string(self, x):
        if self.lower:
            x = x.lower()

        if self.replace_:
            x = x.replace('_', ' ')
        return x

    def __call__(self, x):
        if isinstance(x, str):
            return self.process_string(x)
        return [ self.process_string(s) for s in x]

