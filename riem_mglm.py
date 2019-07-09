#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:22:57 2019

@author: jonyoung
"""

import numpy as np
import scipy as sp

def mglm_spd(X, Y, maxiter=500) :
    
#MGLM_SPD performs MGLM on SPD manifolds by interative method.
#
#   [p, V, E, Y_hat, gnorm] = MGLM_SPD(X, Y)
#   [p, V, E, Y_hat, gnorm] = MGLM_SPD(X, Y, MAXITER)
#   has optional parameter MAXITER.  
#
#   The result is in p, V, E, Y_hat.
#
#   X is dimX x N column vectors
#   Y is a stack of SPD matrices. 3D arrary 3x3xN.
#   p is a base point.
#   V is a set of tangent vectors (3 x 3 x dimX symmetric matrix).
#   E is the history of the sum of squared geodesic error.
#   Y_hat is the prediction.
#   gnorm is the history of norm of gradients.
#
#   See also WEIGHTEDSUM_MX, MGLM_LOGEUC_SPD
 
    
    # get data sizes and check them
    ndimX = np.shape(X)[0]
    ndimY = np.shape(Y)[0]
    ndata = np.shape(X)[1]
    
    if not ndata == np.shape(Y)[2] :
        
        raise ValueError('Different number of covariate X and response Y')
        
    # initialization
    p = Y[:, :, 0];
    V = np.zeros((ndimY, ndimY, ndimX));
    
    # gradient descent algorithm :
    # step size
    c1 = 1
    
    # safeguard parameter
    c2 = 1
    
    V = proj_TpM_spd(V)
    
    # errors? list
    E = []
    gnorm = []
    E.append(feval_spd(p, V, X, Y))
    step = c1
    
    # loop for the specified number of iterations
    for niter in range(maxiter) :
        
        print niter
        Y_hat = prediction_spd(p, V, X)
        J = logmap_vecs_spd(Y_hat, Y)
        err_TpM = paralleltranslateAtoB_spd(Y_hat, p, J)
        gradp = -np.sum(err_TpM, axis=2)
        
        # V projection onto tangent space
        gradV = np.zeros_like(V)
        
        # matrix multiplication
        # loop through matrices in V
        for iV in range(np.shape(V)[2]) :
            
            gradV[:, :, iV] = - weightedsum_mx(err_TpM, X[iV, :])
            
        ns = normVs(p, gradV)
        normgradv = np.sum(ns)
        
        ns = normVs(p, gradp)
        normgradp = np.sum(ns)
        
        gnorm_new = normgradp + normgradv
        if np.iscomplex(gnorm_new) :
            
            print('Numerical Error.')
            
        # Safeguard
        gradp, gradV = safeguard(gradp, gradV, p, c2)
        
        moved = 0
        for i in range(50) :
            
            step = step*0.5
            # Safeguard for gradv, gradp
            V_new = V -step*gradV;
            p_new = expmap_spd(p,-step*gradp);
            if not isspd(p_new) :
                
                p_new = proj_M_spd(p_new)
            
            V_new = paralleltranslateAtoB_spd(p,p_new,V_new)
            E_new = feval_spd(p_new, V_new, X, Y)
            
            if E[-1] > E_new :
                
                p = p_new;
                V = proj_TpM_spd(V_new)
                E.append(E_new)
                
                if np.iscomplex(gnorm_new) :
                    
                    print('Numerical error')
                    print(p)
                    print(V_new)
                    exit
                
                
                gnorm.append(gnorm_new)
                moved = 1
                step = step*2;
                break
            
        if (not moved == 1) or (gnorm[-1] < 1e-10) :
            
            break

    E.append(feval_spd(p,V,X,Y))
    Y_hat = prediction_spd(p,V,X)
    
    return p, V, E, Y_hat, gnorm
    
def normVs(p,V) :
    
    # np.shape(V)[2]  gives IndexError: tuple index out of range
    # when called with normVs(p, gradp)
    # fix : if V is 2-dimensional. add a newaxis
    if V.ndim == 2 :
        
        V = V[:, :, np.newaxis]
    # allocate memory, unlike Matlab function
    n = np.shape(V)[2]
    ns = np.zeros((n, 1))
    for i in range(n) :
        
        ns[i, 0] = norm_TpM_spd(p,V[:,:,i])
        
    return ns
        
def safeguard(gradp, gradV, p, c2) :
    
    ns = normVs(p,gradV)
    normgradv = np.sum(ns)
    ns = normVs(p,gradp)
    normgradp = np.sum(ns)
    norms = [normgradp, normgradv]
    maxnorm = max(norms)
    if maxnorm > c2 :
        
        gradV = gradV*c2/maxnorm
        gradp = gradp*c2/maxnorm
    
    return gradp, gradV

def norm_TpM_spd(p, v) :
    
#    NORM_TPM_SPD calculates the norm of tangent vector v in TpM on SPD manifolds.
#    See also DIST_M_SPD, NORM_TPM_SPD
    
    r = np.sqrt(innerprod_TpM_spd(v,v,p))
    
    return r
    
def innerprod_TpM_spd(U,V,P) :
    
#INNERPROD_TPM_SPD calculates the inner product of U and V in T_{P}M on SPD manifolds.
#
#    r  = INNERPROD_TPM_SPD(U,V,P)
#
#   See also DIST_M_SPD, NORM_TPM_SPD
    try :
        
        invP = np.linalg.inv(P)
        
    except :
        
        invP = np.linalg.inv(P)
        print('pinv')

    sqrtinvP = sp.linalg.sqrtm(invP)
    r = np.trace(np.linalg.multi_dot((sqrtinvP, U, invP, V, sqrtinvP)))
    return r

def feval_spd(p,V,X,Y) :
#FEVAL_SPD evaluates the objective function value (the sum of squared geodesic errors) of MGLM on SPD. 
#
#    E = FEVAL_SPD(p,V, X, Y)
#
#   !! make sure that X is centered if p, V are calculated by centered X !!
#
#   X is a set of column vectors. (dimX X N, where dimX = size(X,1))
#   p is the base point.
#   V is a set of symmetric matrices (dim_p x dim_p x dimX).
#   Y is a set of SPD matrices (dim_p x dim_p x N, where dim_p = size(p,1)).
#   E is the sum of squared geodesic errors.
#
#   See also LOGMAP_SPD, LOGMAP_PT2ARRAY_SPD, EXPMAP_SPD, GSQERR_SPD,
#   PREDICTION_SPD

    P_hat = prediction_spd(p,V,X)
    E = gsqerr_spd(Y, P_hat)
    return E

def prediction_spd(p,V,X) :
#PREDICTION_SPD predicts phat based on estimate p, V and covariate X.
#
#   p is a base point (SPD maxtrix). 
#   V is a set of tangent vectors (3 x 3 x dimX symmetric matrix).
#   X is a set of covariates, dimX x N column vectors.
#   p_hat is the prediction.
#
#   See also MGLM_SPD, FEVAL_SPD

    ndimX, ndata = np.shape(X)
    Yhat = np.zeros((np.shape(p)[0], np.shape(p)[1], ndata))

    for i in range(ndata) : 
        Vi = np.zeros_like(p)
        for j in range(ndimX) :
            
            Vi = Vi + V[:,:,j] * X[j, i]

        Yhat[:, :,  i] = expmap_spd(p,Vi)
        
    return Yhat

def logmap_vecs_spd(X,Y) :
#LOGMAP_VECS_SPD returns logmap(X,Y) for SPD manifolds.
#
#    V = LOGMAP_VECS_SPD(P,X)
#
#    X, Y is a set of SPD matrices (dimX x dimX x N, where dimX = size(X,1)).
#    V is a set of symmetric matrices.
#
#   See also LOGMAP_SPD, LOGMAP_PT2ARRAY_SPD, EXPMAP_SPD

    V = np.zeros_like(Y)
    if np.shape(X)[2] ==1 :
        
        for i in range(np.shape(Y)[2]) :
            
            yi = Y[:, :, i]
            V[:, :, i] = logmap_spd(X,yi)
    
    else :
        
        for i in range(np.shape(X)[2]) :
            
            xi = X[:, :, i]
            yi = Y[:, :, i]
            V[:, :, i] = logmap_spd(xi,yi)
        
    return V

def paralleltranslateAtoB_spd(a, b, w):
    
#PARALLELTRANSLATEATOB_SPD transports a set of tangent vectors w from TaM to
#TbM.
#
#   w_new = PARALLELTRANSLATEATOB_SPD(a, b, w)
#
#   a, b are points on SPD matrices. 
#   w is a set of tangent vectors.
#   w_new is a set of transported tangent vectors.
#
#   See also MGLM_SPD
    
    # np.shape(a)[2] < np.shape(b)[2] gives IndexError: tuple index out of range
    # fix : if a or b are 2-dimensional. add a newaxis
    if a.ndim == 2 :
        
        a = a[:, :, np.newaxis]
        
    if b.ndim == 2 :
        
        b = b[:, :, np.newaxis]

    if np.shape(a)[2] < np.shape(b)[2] :
        
        a = np.tile(a, (1, 1, np.shape(b)[2]))
        
    elif np.shape(a)[2] > np.shape(b)[2] :
        
        b = np.tile(b, (1, 1, np.shape(a)[2]))
    
    if not np.shape(b)[2] == np.shape(w)[2] :
        
#        a, b are fixed
#        This changes only w.
        fixab = 1
        P1 = a
        P2 = b
        
    else :
        
        fixab = 0
    
    w_new = np.zeros_like(w)
    
    for i in range(np.shape(w)[2]) :
        
        if fixab == 0 :
            
            P1 = a[:, :, i]
            P2 = b[:, :, i]
        
        # P1, P2 can have extra singleton dimension which norm function 
        # doesn't like. So squeeze them
        #if np.linalg.norm(P1-P2, 2) < 1e-20 :
        if np.linalg.norm(np.squeeze(P1)-np.squeeze(P2), 2) < 1e-20 :
            
            w_new[:, :, i] = w[:, :, i]
            continue
               
#       invP1 = inv(P1);
#       P12 = sqrtm(invP1*P2*P2*invP1);
#       T12 = P12\invP1*P2;
#       B = P1\w(:,:,i);
#       w_new(:,:,i) = P2*T12'*B*T12;
        w_new[:, :, i] = parallel(P1, P2 ,w[:, :, i])

    # symmetrization.
    for i in range(np.shape(w)[2]) :
        
        w_new[:,:,i] = (w_new[:, :, i] + np.transpose(w_new[:, :, i]))/2
        
    return w_new

def parallel(p, q, w) :
    
    # functions of matrices don't like trailing singleton dims
    p = np.squeeze(p)
    q = np.squeeze(q)
    w = np.squeeze(w)
    
    rtp = sp.linalg.sqrtm(p)
    invrtp = np.linalg.inv(rtp)
    v = logmap_spd(p,q)
    r = sp.linalg.expm(np.linalg.multi_dot((invrtp, v/2, invrtp)))
    w_new = np.linalg.multi_dot((rtp, r, invrtp, w, invrtp, r, rtp))
    return w_new

def weightedsum_mx(mx, w) :

#WEIGHTEDSUM_MX sums matrices with weight w.
#
#    S = WEIGHTEDSUM_MX(mx, w)
#    mx is d x d x N matrices.
#    w is N weights for matrices. w is a column or row vector.
#    S is the weighted sum of mx.
#
#   See also  MGLM_SPD

    w = np.reshape(w,(1, 1, len(w)))
    w = np.tile(w, (np.shape(mx)[0], np.shape(mx)[1], 1));
    S = np.sum(mx*w, axis=2);
    return S
        
def isspd(mx, c=np.finfo(np.float).eps)  :
    
#ISSPD check mx is a symmetric positive definite matrix.
#    This check whether the smallest eigen value is bigger than c.
#    Default c is epsilon.
#
#    Example:
#        T = isspd(mx)
#        T = isspd(mx,C)
#
#   See also MGLM_LOGEUC_SPD, MGLM_SPD, PROJ_M_SPD

    # Check matrices are symmetric positive definite.
    
    # np.shape(mx)[2] gives IndexError: tuple index out of range
    # fix : if mx 2-dimensional. add a newaxis
    if mx.ndim == 2 :
        
        mx = mx[:, :, np.newaxis]
    T = np.zeros((np.shape(mx)[2], 1))
    for i in range(np.shape(mx)[2]) :
    
        #T[i] = np.bitwise_and((np.sum(np.linalg.eigvals(mx[:, :, i]) <= 0+c ) == 0), issym(mx[:, :, i]))
        T[i] = (np.sum(np.linalg.eigvals(mx[:, :, i]) <= 0+c ) == 0) and (issym(mx[:, :, i])[0])
        
    return T
    
def issym(mx) :
    
    # np.shape(mx)[2] gives IndexError: tuple index out of range
    # fix : if mx 2-dimensional. add a newaxis
    if mx.ndim == 2 :
        
        mx = mx[:, :, np.newaxis]
    
    tol = 0.00001
    S = np.zeros((np.shape(mx)[2], 1))
    for i in range(np.shape(mx)[2]) :
        
        S[i] = np.sum(np.abs(mx[:, :, i] - np.transpose(mx[:, :, i])))  < tol
    
    T = all(S)
    return T, S

def proj_M_spd(X, c=np.finfo(np.float).eps) :
    #PROJ_M_SPD projects a matrix onto SPD manifolds.
#
#
#    Example:
#        p = PROJ_M_SPD(X)
#
#   p is the point on SPD manifolds.
#   X is a n x n matrix.
#
#   See also MGLM_LOGEUC_SPD, MGLM_SPD

    # Make a matrix symmetric positive definite.
    if np.linalg.norm(X - np.transpose(X)) > c :
        
        X = (X + np.transpose(X))/2
    D, V = np.linalg.eig(X)
    p = np.zeros_like(X)
    for i in range(len(D)) :
        
        if D[i] > 0 :
        
            p = p + np.multi_dot(D[i], V[:,i], np.transpose(V[:,i]))
   
#     Now X is spd
#     Make psd matrix
    if sum(D > 0+c) < len(D) :
        
        a = 1e-16
        pnew = p
        while not isspd(pnew, c) :
            
            pnew = p + a*np.eye(3)
            a = 2*a
        
        p = pnew
    
    return p
    
def gsqerr_spd(X, X_hat) :    
#GSQERR_SPD returns the sum of geodesic squared error on SPD manifolds.
#
#    gsr = GSQERR_SPD(X, X_hat)
#
#
#   X, X_hat is a set of SPD matrices (dim_X x dim_X x N, where dim_X = size(X,1)).
#
#   See also FEVAL_SPD, R2STAT_SPD

    ndata = np.shape(X)[2]
    gsr = 0;
    for idata in range(ndata) :
        
        gsr = gsr + dist_M_spd(X[:, :, idata], X_hat[:, :, idata]) ** 2
        
    return gsr

def expmap_spd(P, X) :
#EXPMAP_SPD maps tangent vector X onto SPD manifold.
#
#    exp_p_x = EXPMAP_SPD(P,X)
#
#    P, exp_p_x is a SPD matrix.
#    X is a symmetric matrix.
#    
#   See also LOGMAP_SPD, KARCHER_MEAN_SPD

    if np.linalg.norm(X) < 1e-18 :
        
        exp_p_x = P

    else :
         
        D, U = np.linalg.eig(P)
        # use diag
        # as numpy eig returns eigenvalues in a vector, rather than a diagonal
        # matrix as in matlab
        D = np.diag(D)
        g = np.linalg.multi_dot((U, np.sqrt(D)))
        invg = np.linalg.inv(g)
        Y = np.linalg.multi_dot((invg, X, np.transpose(invg)))
        S, V = np.linalg.eig(Y)
        gv = np.dot(g, V)
        # no need to use diag
        # as we take diag later anyway
        # old:
        #exp_p_x = np.linalg.multi_dot((gv, np.diag(np.exp(np.diag(S))), np.transpose(gv)))
        # new:
        exp_p_x = np.linalg.multi_dot((gv, np.diag(np.exp(S)), np.transpose(gv)))
        
#        rtP = sqrtm(P);
#        invrtP = inv(rtP);
#        exp_p_v = rtP*expm(invrtP*V*invrtP)*rtP;
    return exp_p_x
    
def logmap_spd(P,X) :
#LOGMAP_SPD maps X on SPD manifold to the tangent space at P.
#
#    v = LOGMAP_SPD(P,X)
#
#    P, X is a SPD matrix.
#    v is a symmetric matrix.
#
#   See also EXPMAP_SPD, INNERPROD_TPM_SPD, DIST_M_SPD


    if np.linalg.norm(P-X) < 1e-18 :
        
        v = np.zeros_like(P)
    
    else :

        D, U = np.linalg.eig(P)
        # use diag
        # as numpy eig returns eigenvalues in a vector, rather than a diagonal
        # matrix as in matlab
        D = np.diag(D)
        g = np.dot(U, np.sqrt(D))
        invg = np.linalg.inv(g)
        y = np.linalg.multi_dot((invg, X, np.transpose(invg)))
        S, V = np.linalg.eig(y)
        H = np.dot(g, V)
        # no need to use diag
        # as we take diag later anyway
        # old:
        #v = np.linalg.multi_dot((H, np.diag(np.log(np.diag(S))), np.transpose(H)))
        # new:
        v = np.linalg.multi_dot((H, np.diag(np.log(S)), np.transpose(H)))

    return v

def proj_TpM_spd(V) :
#PROJ_TPM_SPD projects a set of tangent V vectors onto TpM. Symmetrization.
#
#   See also MGLM_SPD

    for i in range(np.shape(V)[2]) :
        
        V[:, :, i] = (V[:, :, i] + np.transpose(V[:, :, i]))/2
    
    return V

def dist_M_spd(X, Y) :
#DIST_M_SPD returns distance between X and Y on SPD manifold.
#
#   d = DIST_M_SPD(X,Y)
#
#   See also INNERPROD_TPM_SPD, LOGMAP_SPD

    V = logmap_spd(X,Y)
    d = np.sqrt(innerprod_TpM_spd(V, V, X))
    
    return d
