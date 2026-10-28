import numpy as np
rng=np.random.default_rng(0)
n=[3,4,2]; m=5
W=[None,rng.normal(size=(4,3)),rng.normal(size=(2,4))]; b=[None,rng.normal(size=(4,1)),rng.normal(size=(2,1))]
X=rng.normal(size=(3,m)); Y=rng.integers(0,2,size=(2,m)).astype(float)
sig=lambda z:1/(1+np.exp(-z))
def fwd(W,b,X):
    A=[X];Z=[None]
    Z1=W[1]@X+b[1]@np.ones((1,m)); A1=np.tanh(Z1)
    Z2=W[2]@A1+b[2]@np.ones((1,m)); A2=sig(Z2)
    return [None,Z1,Z2],[X,A1,A2]
def loss_bce(W,b):
    Z,A=fwd(W,b,X); P=A[2]
    return -np.sum(Y*np.log(P)+(1-Y)*np.log(1-P))/m
def loss_sq(W,b):
    Z,A=fwd(W,b,X); return 0.5*np.sum((A[2]-Y)**2)/m
for name,lf in [('bce',loss_bce),('sq',loss_sq)]:
    Z,A=fwd(W,b,X); one=np.ones((m,1))
    if name=='bce': D2=(A[2]-Y)/m
    else: D2=((A[2]-Y)*A[2]*(1-A[2]))/m
    D1=(W[2].T@D2)*(1-A[1]**2)
    gW2=D2@A[1].T; gW1=D1@A[0].T; gb1=D1@one
    # outer product sum check
    s=sum(np.outer(D1[:,j],A[0][:,j]) for j in range(m)); assert np.allclose(s,gW1)
    eps=1e-6; num=np.zeros_like(W[1])
    for i in range(4):
        for k in range(3):
            Wp=[None,W[1].copy(),W[2]];Wp[1][i,k]+=eps
            Wm=[None,W[1].copy(),W[2]];Wm[1][i,k]-=eps
            num[i,k]=(lf(Wp,b)-lf(Wm,b))/(2*eps)
    numb=np.zeros_like(b[1])
    for i in range(4):
        bp=[None,b[1].copy(),b[2]];bp[1][i]+=eps;bm=[None,b[1].copy(),b[2]];bm[1][i]-=eps
        numb[i]=(lf(W,bp)-lf(W,bm))/(2*eps)
    print(name,'max|dW1 err|',np.abs(num-gW1).max(),'max|db1 err|',np.abs(numb-gb1).max())
# rank of single-sample grad
d=rng.normal(size=(4,1));a=rng.normal(size=(3,1));print('rank outer',np.linalg.matrix_rank(d@a.T))
# FLOPs
n0,n1,n2,m=784,128,10,64
g=lambda M,K,N:2*M*K*N
f1=g(n1,n0,m);f2=g(n2,n1,m)
print('fwd L1',f1,'fwd L2',f2,'fwd gemm total',f1+f2,'bias+act adds',2*n1*m+n2*m)
bwW2=g(n2,m,n1);bwA1=g(n1,n2,m);bwW1=g(n1,m,n0);bwX=g(n0,n1,m)
print('bwd dW2',bwW2,'dA1',bwA1,'dW1',bwW1,'total',bwW2+bwA1+bwW1,'(+dX would add',bwX,')')
print('params',n1*n0+n1+n2*n1+n2)
# arithmetic intensity fp32 (4B) for layer1
for mm in [1,64,256]:
    fl=g(n1,n0,mm); by=4*(n1*n0+n0*mm+n1*mm); print('m',mm,'flops',fl,'bytes',by,'FLOP/B',round(fl/by,2))
