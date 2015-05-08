"""
Utility functions for desispec
"""

def night2ymd(night):
    """
    parse night YEARMMDD string into tuple of integers (year, month, day)
    """
    assert isinstance(night, str)
    assert len(night) == 8, 'invalid YEARMMDD night string '+night
    
    year = int(night[0:4])
    month = int(night[4:6])
    day = int(night[6:8])
    if month < 1 or 12 < month:
        raise ValueError('YEARMMDD month should be 1-12, not {}'.format(month))
    if day < 1 or 31 < day:
        raise ValueError('YEARMMDD day should be 1-31, not {}'.format(day))
        
    return (year, month, day)
    
def ymd2night(year, month, day):
    """
    convert year, month, day integers into cannonical YEARMMDD night string
    """
    return "{:04d}{:02d}{:02d}".format(year, month, day)
    
