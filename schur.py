import numpy as np
from numba import jit, prange

@jit(nopython=True)
def getCd(j,m):
    '''
    Lowering ladder coefficient, SQUARED!
    '''
    return j*(j+1)-m*(m-1)

@jit(nopython=True)
def getCu(j,m):
    '''
    Lifting ladder coefficient, SQUARED!
    '''
    return j*(j+1)-m*(m+1)

def fuse(j1,j2):
    '''
    Given j1 and j2, generate a matrix that maps a separable
    basis (with m2 as LSB in lexicographical ordering), into
    a joint basis, in descending order of J, followed by m.
    '''
    lb = abs(j1-j2) # Lower bound on J
    ub = j1+j2 # Upper bound on J
    
    ct1 = int(2*j1+1)
    ct2 = int(2*j2+1)
    ct = ct1*ct2 # Total output dimension.
    output = np.zeros((ct,ct1,ct2))
    
    out_idx = 0
    j = ub
    while j>=lb:
        #print(">>> Doing J=",j,", and starting index is: ",out_idx)
        # Populate uppermost m state
        # m1 really should run from j-j2 to j1 inclusive.
        # Shift in loop because m indices logically run from -j to +j
        m1lb = int(j-j2+j1) #Lower-bound for m1, in (int)
        coef = 1.
        output[out_idx,ct1-1,m1lb-ct1+ct2] = 1.
        for m1 in range(ct1-2,m1lb-1,-1):
            # Now "m1" runs from 0 to 2j.
            # We want m2 to be logically j-m1
            #m2 = j-(m1-j1)+j2
            m2 = m1lb-m1+ct2-1 #The above, re-expressed in (int)
            
            # Calculate ladder ops
            m1l = m1-j1
            c_ladder = np.sqrt( getCu(j2,j-m1l-1)/getCu(j1,m1l) )
            coef = -coef*c_ladder
            output[out_idx,m1,m2] = coef
        # Normalize
        #print(out_idx,";\n",output[out_idx])
        output[out_idx] = output[out_idx]/np.sqrt(np.sum(output[out_idx]**2))

        # Now recursively populate all other m-values in the J sector.
        for idx in range(int(2*j)):
            # Top-level m-values
            for m2 in range(0,ct2-1):
                output[out_idx+1][ct1-1][m2] = np.sqrt(getCd(j2,m2+1-j2))*output[out_idx][ct1-1][m2+1]
            for m1 in range(0,ct1-1):
                output[out_idx+1][m1][ct2-1] = np.sqrt(getCd(j1,m1+1-j1))*output[out_idx][m1+1][ct2-1]
            # Intermediate m-values
            for m1 in range(0,ct1-1):
                for m2 in range(0,ct2-1):
                    output[out_idx+1][m1][m2] = np.sqrt(getCd(j1,m1+1-j1))*output[out_idx][m1+1][m2] + np.sqrt(getCd(j2,m2+1-j2))*output[out_idx][m1][m2+1]
            # Normalize
            output[out_idx+1] = output[out_idx+1]/np.sqrt(np.sum(output[out_idx+1]**2))
            
            out_idx = out_idx + 1
            #print(out_idx, "; m=",j-idx-1,"\n", output[out_idx])

        out_idx = out_idx + 1
        j = j - 1
    return output


def getBlocks(primitive,jmax,jmin):
    '''
    Takes a block of primitives from fuse().
    Returns a dictionary of blocks indexed by "j,j1,j2".
    '''
    # Number of blocks
    ctBlk = int(jmax-jmin+1)
    output = {idx_blk+jmin:np.array([]) for idx_blk in range(ctBlk)}
    
    idx = 0
    for idx_blk in range(ctBlk):
        j = (jmax-idx_blk)
        ct = int(2*j+1)
        output[j] = np.flip(primitive[idx:idx+ct],axis=0)
        idx = idx + ct
    return output


def generateCGseries(j_unit,copies):
    '''
    Given a basic unit of spin J, generate the
    Clebsch-Gordan series required for Schur transform.
    Returned dictionary is indexed as follows:
    [1st input J][Total output J given the 1st input J and j_unit]
    '''
    output = {n*j_unit:{} for n in range(1,copies)}
    for n in range(1,copies):
        idx = n*j_unit
        primitive = fuse(n*j_unit,j_unit)
        output[idx] = getBlocks(primitive,j_unit+idx,idx-j_unit)
    return output