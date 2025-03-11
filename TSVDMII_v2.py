import numpy as np
import scipy as sp
from scipy import linalg
import math
from scipy.linalg import qr

def tSVDMII(A,M,nrg,*args):

    isfunc = isinstance(M, str)

    if len(args) == 0:
        orthflag = 0
    else:
        orthflag = args[0]

    if not isfunc:
        Ahat = mode3(A, M)
        
    else: 
        # Assuming M is the name of the function as a string
        function_to_call = globals()[M]
        # Call the function with the specified arguments
        Ahat = function_to_call(A,None,0)
    
    [m,p,n] = Ahat.shape

    Uah = np.zeros((m, p, min(p, n)))
    Vah = np.zeros((m, n, min(p, n)))
    Sah = np.zeros((m, min(p, n), min(p, n)))
    Delta = np.zeros((min(p, n), m))

    for i in np.arange(0, m):
        U, S, Vt = np.linalg.svd(Ahat[i, :, :], full_matrices=False)
        Uah[i, :, :] = U
        Sah[i, :, :] = np.diag(S)
        Vah[i, :, :] = Vt.T
        Delta[:, i] = S**2

    frnm2 = np.sum(Delta.ravel()) # jth col of Delta are the squared svals of jth frontal slice 
    sortedDeltas = np.sort(Delta.ravel())[::-1] # sort these globally, descending order 
    cus = np.cumsum(sortedDeltas)
    cus = cus/frnm2 # cumulative sum of the sorted deltas. These are called the energies 
    fi = np.where(cus<=nrg)[0] # find all indices where energy is less than the prescribed energy value 
    idx = fi.shape[0] # Zeros the values of energy that are larger than prescribed value 

    if len(fi) == 0:
        print('Warning nrg too small')
        cmplvl = fi
        RE = idx
        return [cmplvl, RE]

    cutoff = sortedDeltas[idx]
    fi = np.where(sortedDeltas >= cutoff)[0] # find all values in the delta array that are greater than cutoff

    Sah2 = np.zeros_like(Sah)
    F = np.where(Sah>=np.sqrt(cutoff))
    Sah2[F] = Sah[F]
    Sah = Sah2 # this is a facewise diagonal with too small svalues commented out 
    del_count = np.count_nonzero(Sah) # Count the total number of non-zeros 

    ln = m*p*n 
    ln2 = p*del_count + n*del_count
    cmplvl = ln/ln2 

    Appx = np.zeros((m, p, n))
    for i in np.arange(0,m):
        Appx[i,:,:] = Uah[i,:,:]@Sah[i,:,:]@Vah[i,:,:].T

    if not isfunc: 
        Appx = mode3i(Appx, M, 1)
    else:
        Mi = 'i' + M
        Appx = globals()[Mi]

    RE = fronorm(Appx - A) / fronorm(A)
    print("Relative error in approximation is", RE)


    return (Appx, RE, cmplvl)

def mode3(A, M):

    [m,p,n] = A.shape
    [r,q] = M.shape 

    # reshaping A
    A3 = np.reshape(A, (m,p*n), order='F')

    if m != q: 
        print('Wanring, wrong size. Quitting')
        return 0

    B = M@A3
    B = np.reshape(B, (r,p,n), order='F')

    return B

def mode3i(A, M, *args): 
    
    if len(args) == 0:
        orthoflag = 0
        print('Hello')
    else: 
        orthoflag = 1
    
    [m,p,n] = A.shape
    [r,q] = M.shape

    # reshaping A
    A3 = np.reshape(A, (m,p*n), order='F')

    if m != r: 
        print('Warning, wrong size. Quitting')
        return 0

    if orthoflag == 0:
        B = np.linalg.solve(M, A3)
    else: # Matrix is orthogonal so tranpose becomes inverse 
        B = np.transpose(M)@A3

    B = np.reshape(B, (q,p,n), order='F')

    return B

def fronorm(A):
    x = np.linalg.norm(A.flatten())
    return x

