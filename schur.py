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


def tileDiag(mat,n):
    '''
    Takes a matrix mat, and tiles it n-times
    into a block diagonal matrix.
    '''
    height = mat.shape[0]
    width = mat.shape[1]
    output = np.zeros((n*height,n*width))
    for idx in range(n):
        output[idx*height:(idx+1)*height,idx*width:(idx+1)*width] = mat
    return output


def buildJtransorms(j_unit,copies):
    '''
    Could blow up!!!

    When it doesn't blow up, return set of unitary transformations
    between computational basis states and subspaces of definite J's.
    '''
    cgSeries = generateCGseries(j_unit,copies)
    # print(cgSeries)
    ops = []
    j_track = []
    for entry in cgSeries[j_unit]:
        op = cgSeries[j_unit][entry]
        shape = op.shape
        ops.append(op.reshape(int(2*entry+1),-1))
        j_track.append(entry)
        
    for idx2 in range(copies-2):
        newOps = []
        newJ = []
        for idx in range(len(j_track)):
            shape = ops[idx].shape
            targ = tileDiag(ops[idx],int(2*j_unit+1))
            # print("Input J: ",j_track[idx],"; Target op:\n",targ)
            if j_track[idx]<0.5:
                newOps.append(targ)
                newJ.append(j_unit)
            else:
                for entry in cgSeries[j_track[idx]]:
                    newOp = np.transpose(cgSeries[j_track[idx]][entry],axes=(0,2,1))
                    newOp = newOp.reshape(int(2*entry+1),-1)
                    # print("    Output J: ", entry, "; New op:\n", newOp)
                    newOp = np.dot(newOp,targ)
                    newOps.append(newOp)
                    newJ.append(entry)
        ops = newOps
        j_track = newJ
    
    return ops,j_track

def buildFullMat(j_unit,copies):
    '''
    Assemble J-definite transformations from above into a single
    CG-tranformation unitary matrix.
    '''
    ops,js = buildJtransorms(j_unit,copies)
    sort_args = np.argsort(js)
    ops_sorted = np.take(ops,sort_args)
    opFull = np.vstack(ops_sorted)
    return opFull


# ============================================= #
# >>> Data compression circuit construction <<< #
# ============================================= #

def genCompression(circ):
    '''
    Given a QISkit circuit, tack on a 3-qubit data-compression sequence.
    '''
    alpha = np.arccos(1/np.sqrt(3))

    # Add U-1/2
    circ.cnot(1,0)
    circ.cnot(0,1)
    circ.ry(-np.pi/4,1)
    circ.cnot(0,1)
    circ.ry(np.pi/4,1)

    # Add U-1 and U-0
    # Basis re-arrangement (qb0 is LSB, qb2 is MSB)
    circ.x(1)
    circ.x(2)
    circ.ccx(1,2,0)
    circ.x(2)
    circ.ccx(0,1,2)
    circ.x(1)
    circ.cnot(2,0)
    circ.ccx(0,2,1)
    circ.cnot(1,0)

    # Do W-block
    circ.ccx(0,1,2)
    circ.ry(2*alpha-np.pi/2,2)
    circ.ccx(0,1,2)
    circ.ry(np.pi/2-2*alpha,2)

    # Do V-block
    circ.cnot(1,2)
    circ.ry(-alpha,2)
    circ.cnot(1,2)
    circ.ry(alpha,2)

    return circ


# ============================================== #
# >>> Specht modules and irreps of Sym-group <<< #
# ============================================== #
# @jit(nopython=True)
def getShapes(n,d):
    '''
    For n copies of a d-level system, compute all shapes and multiplicities.
    Returns a Dict of np.arrays. Zeroth entry of each np.array is multiplicity.
    Rest of the np.array is a partitioning of n in weakly descending order.

    Numba's typed Dict instantiation seems problematic here.
    That's why njit-related items are commented out. Calls to 'prange' are also
    replaced with the normal Pythonic 'range'.
    '''
#     currentShape = Dict.empty(key_type=types.unicode_type,
#                               value_type=types.int64[:])
    initVal = np.zeros(d-1,dtype=np.int64)
    initVal = np.concatenate( (np.array([1,1],dtype=np.int64),initVal) )
    initKey = str(initVal[1:])
    currentShape = {initKey:initVal}
    
    for idx in range(n-1):
#         newShape = Dict.empty(key_type=types.unicode_type,
#                               value_type=types.int64[:])
        newShape = {}
        for key in currentShape:
            arrDiff = np.empty(d-1,dtype=np.int_)
            entry = currentShape[key]
            for idx2 in range(d-1):
                arrDiff[idx2] = entry[idx2+1] - entry[idx2+2]
            
            nonzeros = np.nonzero(arrDiff)[0]
            insertions = np.concatenate((np.array([-1]),nonzeros))
            
            for idx2 in range(len(insertions)):
                newArr = entry.copy()
                newArr[insertions[idx2] + 2] += 1
                newKey = str(newArr[1:])
                try:
                    newShape[newKey][0] += entry[0]
                except:
                    newShape.update({newKey:newArr})
        currentShape = newShape
        
    return currentShape


@jit(nopython=True)
def getDim(part):
    '''
    Given a shape-array (as returned by getShapes above), compute the
    dimensionality of the vector-space spanned by the irrep.
    '''
    d = len(part)-1
    tot = 1
    for i in prange(1,d):
        for j in range(i+1,d+1):
            tot *= (part[i]-i+j-part[j])/(d+1-j)

    return tot


#===========================#
# >>> Convenience funcs <<< #
#===========================#
def additiveKron(op,dim,n,onlist,op2=[]):
    '''
    Build a symmetrized sum of tensor-product terms of op.
    Args:
        op: (Mat) For each summand, insert op where index is listed in onlist
        op2:(Mat) When NOT listed in onlist, insert op2 instead.
        dim:(Int) Dimension of each operand.
        n:  (Int) Number of multiplicand in each tensor-product term.
    Returns:
        Matrix of size dim^n.
    '''
    if len(op2)==0:
        op2 = np.eye(dim,dtype=np.int_)/dim

    out = np.zeros((dim**n,dim**n),dtype=np.complex128)
    
    for entry in onlist:
        temp = np.array([1])
        for idx in range(n):
            if not (idx in entry):
                temp = np.kron(temp,op2)
            else:
                temp = np.kron(temp,op)
        out += temp
    
    return out

def genOnList(n,m):
    '''
    Support func for additiveKron above.

    Generates a nCm dimensional list
    of m-dimensional lists, enumerating
    all n-choose-m sets.
    
    Could blow up!!!
    '''
    dim = int(sps.binom(n,m))
    out = np.zeros((dim,m),dtype=np.int_)
    
    out[0] = np.arange(m,dtype=np.int_)
    for idx in range(1,dim):
        out[idx] = out[idx-1]
        for idx2 in range(m):
            limUp = 0
            if idx2==0:
                limUp = n-1
            else:
                limUp = out[idx][m-idx2]-1
                
            if not (out[idx][m-idx2-1]==limUp):
                out[idx][m-idx2-1] += 1
                limDown = out[idx][m-idx2-1]+1
                temp = np.arange(limDown,limDown+idx2,dtype=np.int_)
                out[idx][m-idx2:] = temp
                break
    
    return out