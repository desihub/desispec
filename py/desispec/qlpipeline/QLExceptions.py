
class ParameterException(Exception):
    def __init__(self,value):
        self.value=value
    def __str__(self):
        return "Parameter Exception: %s"%(repr(self.value))
