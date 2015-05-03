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
    assert 1 <= month <= 12, 'YEARMMDD month should be 1-12, not {}'.format(month)
    assert 1 <= day <= 31, 'YEARMMDD day should be 1-31, not {}'.format(day)
    return (year, month, day)
    
def ymd2night(year, month, day):
    """
    convert year, month, day integers into cannonical YEARMMDD night string
    """
    return "{:04d}{:02d}{:02d}".format(year, month, day)
    
if __name__ == '__main__':
    assert ymd2night(2015, 1, 2) == '20150102'
    assert night2ymd('20150102') == (2015, 1, 2)