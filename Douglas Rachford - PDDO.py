import numpy as np
from scipy import spatial
import copy

"""
PD Enabled Learning
Ali Can Bekar
2021
"""


"""
Solution of Lasso Using Douglas-Rachford Splitting
Feature Matrix is Normalized to Have Max Value of 1
at Each Column. After the Feature Selection, Regular
Least Squares Method is Applied to Refine the Coeffs.
argmin (1/2)*||V-F*alpha||_2^2 + lam||alpha||_1
Ref: [Combettes PL, Pesquet J-C. 2011 Proximal 
splitting methods in signal processing
https://doi.org/10.1007/978-1-4419-9569-8_10]
"""
##################################################################################
##################################################################################
# Proximal Splitting for lam||alpha||_1
def Prox1(gamma, Lambda, xtilda):
    svec = np.sign(xtilda)
    x = np.multiply(np.maximum(abs(xtilda) - gamma * Lambda, 0.0), svec)
    return x
# Proximal Splitting for ||V-F*alpha||_2^2
def Prox2(gamma, FeMat, Vel, x):
    MatP = np.dot(FeMat.T, FeMat) * gamma + np.eye(len(x))
    VecP = np.dot(FeMat.T, Vel) * gamma + x
    xtilda = np.linalg.solve(MatP, VecP)
    return xtilda
# Main Algorithm for DR
def DougRach(maxit, tolerence, mu, Lambda, gamma, FeMat, Vel):
    numterm = len(FeMat[0])
    Alpha = np.zeros([numterm,])
    Alphapre = np.zeros([numterm,])
    Alphatil = np.zeros([numterm,])
    residual = tolerence * 10
    itnum = 0
    """
    alphatilda^(k+1) = (1-mu/2)alphatilda^(k)+(mu/2)*rprox_(H2)(rprox_(H1)(alphatilda^(k)))
    apha^(k+1) = prox_(H1)(alphatilda^(k))
    """
    while(residual > tolerence):
        Alphatil = Alphapre
        rprox1T = Prox1(gamma, Lambda, Alphatil) * 2.0 - Alphatil
        rprox2T = (Prox2(gamma, FeMat, Vel, rprox1T) * 2.0 - rprox1T) * mu / 2.0
        Alphatil = rprox2T + Alphatil * (1.0 - mu / 2.0)
        Alpha = Prox1(gamma, Lambda, Alphatil)
        # Residual Term to Check Convergence
        diff = Alpha - Alphapre
        residual = np.linalg.norm(diff)
       # print('Norm of the Residual for DR Algorithm :' + str(residual) + '\n')
        Alphapre = Alpha
        itnum += 1
        if itnum > maxit:
            break
    nzindx = np.where(Alpha != 0)[0]
    print('Number of Nonzero Coefficients : ' + str(len(nzindx)))
    # Regular Least Squares to Fine Tune the Inverted Coefficients
    Alpha[nzindx] = np.linalg.lstsq(FeMat[:, nzindx],Vel, rcond=-1)[0]
    return Alpha
# Normalization of Feature Matrix
def L2Normalize(FeMat):
    FeMat_N = copy.deepcopy(FeMat)
    sz = len(FeMat[0])
    regvec = np.zeros(len(FeMat[0]))
    for i in range(sz):
        regvec[i] = 1.0 / np.amax(FeMat[:,i])
        FeMat_N[:,i] *= regvec[i]
    return FeMat_N, regvec
##################################################################################
##################################################################################


"""
1D PD Derivative Matrix Generator for Given Horizon Radius.
Generates the Derivative Matrices Up to Order 4. Default
Order is 2.
Ref:[Madenci, E., Barut, A., & Futch, M. (2016). 
Peridynamic differential operator and its applications
http://dx.doi.org/10.1016/j.cma.2016.02.028]
"""
##################################################################################
##################################################################################
# G Function Generator for 1D
def InvGFunc(xi, delta, amat, order):
    GFS = np.zeros(order + 1)
    Avec = np. zeros(order + 1)
    for i in range(order + 1):
        Avec[i] = pow(xi, i)
    w = np.exp(-4.0 * (np.abs(xi) / delta) ** 2)
    for i in range(order + 1):
        C = amat[:, i] * Avec
        C *= w
        GFS[i] = np.sum(C)
    return GFS        
# A Matrix Generator
def GenAmat(xi, delta, order):
    Avec = np. zeros(order + 1)
    for i in range(order + 1):
        Avec[i] = pow(xi, i)
    w = np.exp(-4.0 * (np.abs(xi) / delta) ** 2)
    Amat = np.outer(Avec, Avec) * w
    return Amat
# b Matrix Generator
def Genbmat(order):
    bmat = np.eye(order + 1)
    for i in range(1, order + 1):
        bmat[i, i] = bmat[i - 1, i - 1] * i
    return bmat
# Derivative Operator Using PDDO
def GenDmat(dx, x, delta, order):
    lenx = len(x)
    x2 = x + max(x)
    x3 = x2 + max(x)
    x = np.concatenate((x, x2, x3), axis=None)
    points = np.c_[x.ravel()]
    tree = spatial.KDTree(points)
    Dmats = []
    for i in range(order + 1):
        Dmats.append(np.zeros([lenx, lenx]))
    for i in range(lenx):
        bmat = Genbmat(order)
        pts = np.sort(tree.query_ball_point(points[i + lenx], delta * dx))
        GFMAT = np.zeros([len(pts), order + 1])
        Amat = np.zeros([order + 1, order + 1])
        for num in points[pts]:
            xi = num - x[i + lenx]
            Amat += GenAmat(xi, delta * dx, order)
        Amat *= dx
        amat = np.linalg.solve(Amat, bmat)
        for k in pts:
            xi = x[k] - x[i + lenx]
            for derord in range(order + 1):
                Dmats[derord][i, k%lenx] = InvGFunc(xi, delta * dx, amat, order).T[derord] * dx
    return Dmats
##################################################################################
##################################################################################


"""
Peridynamic Derivative Calculations.
"""
##################################################################################
##################################################################################
def PDFieldDer(PD_Der_Mats, FieldVarSpl, Numtstp, order):
    Fders = []
    for ordc in range(order):
        Fders.append(np.zeros_like(FieldVarSpl))
    for i in range(Numtstp):
        for ordc in range(order):
            Fders[ordc][i] = np.dot(PD_Der_Mats[ordc], FieldVarSpl[i])
    return Fders
##################################################################################
##################################################################################

# Feature Matrix Creator Using Field Data and its Derivatives
def CrtFeatMat(u, ux, ux2):
    u = np.concatenate(u)
    ux = ux.flatten()
    ux2 = ux2.flatten()
    bias = np.ones_like(u)
    Fmat = np.column_stack((bias, u, ux, ux2, u * u, u * ux,
                            u * ux2, ux * ux, ux * ux2, ux2 * ux2))
    FSList = np.array(['1', 'u', 'ux', 'ux2', 'u * u', 'u * ux',
                            'u * ux2', 'ux * ux', 'ux * ux2', 'ux2 * ux2'])
    return Fmat, FSList


# Prints the Discovered Coefficients for Corresponding Terms
def CoeffPrint(Terms, Coeffs):
    print('Discovered Coefficients :')
    print('============================')
    for i, k in zip(Terms, Coeffs):
        print('{:>10} {:>10.4f}'.format(i,k))

def TrainError(FeatMat, VelVec, Alpha):
    Err = np.linalg.norm(VelVec - np.dot(FeatMat, Alpha)) / np.linalg.norm(VelVec)
    print('Training Error : ' + str(Err))
    return Err


numpt = 1025
numtstp = 100
xmin = 0.0
xmax = 1.0
dx = (xmax - xmin) / (numpt - 1)
order = 2
delta = order + 1.015
x = np.linspace(xmin, xmax, numpt)

Dmats = GenDmat(dx, x, delta, order)
Dmats.pop(0)
u = np.genfromtxt('VB_Sol.dat')
Vel = np.genfromtxt('VB_Vel.dat')
u = np.split(u, numtstp)
u_ders = PDFieldDer(Dmats, u, numtstp, order)
[Fmat, FSL] = CrtFeatMat(u, u_ders[0], u_ders[1])
[FMat_N, Norm_coeffs] = L2Normalize(Fmat)
Alpha = DougRach(1000, 1e-12, 1.2, 37.0, 0.5, FMat_N, Vel)
# Revert Normalization
Alpha = np.multiply(Alpha, Norm_coeffs)
Err = TrainError(Fmat, Vel, Alpha)
# Printing Discovered Coefficients
CoeffPrint(FSL, Alpha)
