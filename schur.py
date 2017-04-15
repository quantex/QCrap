def getCd(j,m):
    '''
    Lowering ladder coefficient, SQUARED!
    '''
    return j*(j+1)-m*(m-1)

def getCu(j,m):
    '''
    Lifting ladder coefficient, SQUARED!
    '''
    return j*(j+1)-m*(m+1)