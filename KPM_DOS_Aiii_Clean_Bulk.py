#!/usr/bin/env python

import numpy as np
import scipy.fftpack as fftp
import scipy.sparse.linalg as spla
import scipy.sparse as sps
import time
import argparse

# clean Hamiltonian (z periodic bc)
def make_aiii_spinor(Lx, Ly, Lz,m5,t):
    """
    Maintain the spinor structure. 
    15th March, 2021
    """
    #=----- Pauli matrix
    sx = sps.csc_matrix([[0., 1.], [1., 0.]])
    sy = sps.csc_matrix([[0., -1j],[1j, 0.]])
    sz = sps.csc_matrix([[1., 0.],[0., -1.]])
    s0 = sps.csc_matrix([[1,0], [0,1]]) 
    #=----- Gamma matrix
    g1 = sps.kron(sx, sx)
    g2 = sps.kron(sx, sy)
    g3 = sps.kron(sx, sz)
    g5 = sps.kron(sy, s0)
    gg = sps.kron(s0, s0)
    #=----- Hopping
    tx=0.5*(t*g5 - 1j*g1) 
    ty=0.5*(t*g5 - 1j*g2)
    tz=0.5*(t*g5 - 1j*g3)
    #=----- E0
    E0 = m5*g5
    #=----- diagonal part
    dd =sps.kron(E0, sps.eye(Lx*Ly*Lz), format='csc')
    #=----- previous matrices of hopping in 1d with bc
    xAhop = sps.diags(np.ones(Lx-1), 1, format='csc')
    xAbc  = sps.diags(np.ones(1), Lx-1, format='csc')
    yAhop = sps.diags(np.ones(Ly-1), 1, format='csc')
    yAbc  = sps.diags(np.ones(1), Ly-1, format='csc')
    zAhop = sps.diags(np.ones(Lz-1), 1, format='csc')
    zAbc  = sps.diags(np.ones(1), Lz-1, format='csc')
    Lxone  = sps.eye(Lx, format='csc')
    Lyone  = sps.eye(Ly, format='csc')
    L2oneYZ = sps.eye(Ly*Lz, format='csc')
    L2oneXZ = sps.eye(Lx*Lz, format='csc')
    ##=---------------------------------
    #
    A1  = sps.kron(tx, sps.kron(L2oneYZ, xAhop)) + \
            sps.kron(tx.conj().T, sps.kron(L2oneYZ, xAbc))
    #=---------------------------------
    A2  = sps.kron(ty, sps.kron(yAhop, L2oneXZ)) + \
            sps.kron(ty.T.conj(), sps.kron(yAbc, L2oneXZ))
    # #=---------------------------------
    A3  = sps.kron(tz, sps.kron(Lyone, sps.kron(zAhop, Lxone))) + \
            sps.kron(tz.T.conj(), sps.kron(Lyone, sps.kron(zAbc, Lxone)))
    ##= ----- Final matrix
    A3d = A1 + A2 + A3 
    A3d = A3d + A3d.T.conj() + dd
    return A3d
#@profile
def rescale(H,Hdim,eps = 0.01):
    """
    H    : For rescaling has to be complex hermitian (use eigs) or real symmetric square(use eigsh,BE).
           Finds k eigenvalues.
    LM   : Largest(in magnitude) eigenvalues.
    Hdim : dimension of H. Receives and returns in sparse form.
    """
    lmin, lmax =  spla.eigs(H, k=2, which='LM', return_eigenvectors=False)
    
    lmin = np.real(lmin)
    lmax = np.real(lmax)
    

    a=(lmax-lmin)/(2.- eps )
    b=(lmax+lmin)/2
    
    H_rescaled = (1/a)*(H- b*sps.eye(Hdim) )
    return H_rescaled, a, b
#@profile
def Jackson_Kernel(Nm):
    """
    Nm = no of moments
    retruns an array for kernel improvement  
    """
    n=np.arange(Nm)
    return ((Nm-n+1)*np.cos(np.pi*n/(Nm+1)) + np.sin(np.pi*n/(Nm+1))/np.tan(np.pi/(Nm+1)))/(Nm+1)

#@profile
def calc_mu_one(ham,r,Nm):
    """
    ham : Rescaled Hamiltonian
    r   : random vector
    Nm  : number of moments
    It calculates the moments for a given random vector.
    """
    mu=np.zeros(Nm)
    alpha=[]
    alpha.append(r)
    alpha.append(ham.dot(r))
    mu[0]= np.real( (alpha[0].T.conj()).dot(alpha[0]) )
    mu[1]= np.real( (alpha[0].T.conj()).dot(alpha[1]) )

    for i in range(1,Nm//2):
        alpha.append(2*ham.dot(alpha[i])-alpha[i-1])
        mu[2*i]  = np.real ( 2*(alpha[i].T.conj()).dot(alpha[i])-mu[0] )
        mu[2*i+1]= np.real ( 2*(alpha[i+1].T.conj()).dot(alpha[i])-mu[1] )

    return mu

#@profile
def calc_mu_all(H_rescale,v,Hdim,Nm,R):
    
    mu = np.empty ([ R , Nm ] )
    
    for i in range(R):
        mu[i,:] = calc_mu_one(H_rescale,v[i],Nm)
    
    mu = np.mean(mu, axis = 0 )
    
    return mu/Hdim   

#@profile
def random_vectors_c(R,Hdim):
    """
    R : # of vectors
    N : Dimension of each vectors
    """
    v=np.empty([R,Hdim],np.complex_)
    for i in range(R):
        v[i]=np.sqrt(3.)*np.random.rand(Hdim)*np.exp(2*np.pi*1j*np.random.rand(Hdim))
    return v

#@profile
def mu_fft(mu_improved , Nm , npts ):
    """ 
    Calculates the discrete cosine transform.
    
    
    """
    mu_dft = np.zeros( npts, dtype = 'complex')
    mu_dft[0:Nm] = mu_improved                                           
    mu_transform = fftp.dct( mu_dft,type =3)
    
    return mu_transform

#@profile
def FunctionReps( mu_transform,  npts ):
    """
    mu_improved = mu * Kernel ( mu * g)
    
    npts: No. of points at the the dunction (DOS) is to be reconstructed.
    
    npts = 2 * Nm (  npts >> Nm , npts = 2* Nm Ref. Weibe et. all)
    
    """
    k = np.arange(npts)                            
    x_rescaled = np.cos((np.pi*(k+0.5))/ npts )

    g = np.pi*(np.sqrt(1-(x_rescaled)**2))
    
    fxk_rescaled = np.divide(mu_transform,g)
    
    return x_rescaled , fxk_rescaled

#@profile
def Rescale_back(x_rescaled , fxk_rescaled , scale_a , scale_b ):
    """
    Rescale back x_k and fxk 
    
    """
    xk = x_rescaled*scale_a + scale_b 
    
    fxk = fxk_rescaled/abs(scale_a)
    
    return xk , fxk   



#@profile
def main_code(args):
    start = time.time()

    ##  KPM Paramters
    R     = args.R
    Nm    = args.Nm
    npts  = args.npts

    ##  Hamiltonian Paramters
    Lx = args.Lx
    Ly = args.Ly
    Lz = args.Lz
    m5 = args.m5
    t  = 1

    ### seed initialization 
    np.random.seed(args.sd)


    ###  Generate the Hamiltonian
    H    = make_aiii_spinor(Lx, Ly, Lz,m5,t)
    Hdim = H.shape[0] 

    ###  rescale the Hamiltonian
    H_rescale, scale_a , scale_b =rescale(H,Hdim)
    
    ###  Generate the random vectors
    v = random_vectors_c(R,Hdim)

    ###  Calculate all the R moments
    mu = calc_mu_all(H_rescale,v,Hdim,Nm,R)
    
    ##mu = func_mu_all(H_rescale,v.T,Hdim, Nm , R)

    ###  Kernel improvements
    mu_improved = mu*Jackson_Kernel(Nm)

    ###  discrete cosine transform
    mu_transform = mu_fft(mu_improved , Nm , npts )

    ###  Function reconstruction
    x_rescaled, fxk_rescaled = FunctionReps( mu_transform,  npts )

    ###  Rescaling back
    xk , fxk = Rescale_back(x_rescaled , fxk_rescaled , scale_a , scale_b )

    


    ### Saving the xk and fxk 
    np.save("xk-Lx"+str(args.Lx)+"_Ly"+str(args.Ly)+"_Lz"+str(args.Lz)+"_Nm"+str(args.Nm)+"_npts"+str(args.npts)+"_m5"+str(args.m5)+"_R"+str(args.R)+"_sd"+str(args.sd),xk)

    np.save("fxk_Lx"+str(args.Lx)+"_Ly"+str(args.Ly)+"_Lz"+str(args.Lz)+"_Nm"+str(args.Nm)+"_npts"+str(args.npts)+"_m5"+str(args.m5)+"_R"+str(args.R)+"_sd"+str(args.sd),fxk)


    ## print time in minute
    end = time.time()
    print("time[Min]=", (end-start)/60. )




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='KPM DOS calculation for Aiii Hamiltonian', prog='main', usage='%(prog)s[options]')

    parser.add_argument("-Lx", "--Lx",help="x-length", default=6,type=int,  nargs='?')

    parser.add_argument("-Ly", "--Ly",help="y-lenth", default=6,type=int,   nargs='?')

    parser.add_argument("-Lz", "--Lz",help="z-length", default=6,type=int,  nargs='?')

    parser.add_argument("-m5", "--m5",help="mass term", default=0,type=int, nargs='?')

    parser.add_argument("-Nm", "--Nm",help="No of moments", default=2048,type=int, nargs='?')

    parser.add_argument( "-R", "--R", help="No of random vectors",default=10,type=int,nargs='?')

    parser.add_argument( "-npts", "--npts", help="No of points for function reconstruction",default=4096,type=int,nargs='?')

    parser.add_argument("-sd", "--sd",help="intial seed", default=1,type=int, nargs='?')

    args=parser.parse_args()

    main_code(args)

