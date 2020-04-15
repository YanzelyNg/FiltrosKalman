
import numpy as np
from numpy import matrix 
from pylab import *

CantObs=int(input('Ingrese cantidad de observaciones:  '))
CantY=2
P=np.arange(CantObs)
V=np.arange(CantObs)
matrixObs=np.ones((CantObs, 2))
BuffObsX=np.arange(CantObs)
BuffObsV=np.arange(CantObs)

BuffPredX=np.arange(CantObs)
BuffPredV=np.arange(CantObs)
BuffKalX=np.arange(CantObs)
BuffKalV=np.arange(CantObs)

# print('matrixObs1',matrixObs)
###################################
##INPUT VALUES
###################################

x=0       
for x in range (CantObs):
    print('Ingrese dato de Posición [',x,'] =')
    BuffObsX[x]=float(input())
    print('Ingrese dato de Velocidad [',x,'] =')
    BuffObsV[x]=float(input())

BuffPredX[0]=BuffObsX[0]
BuffPredV[0]=BuffObsV[0]
BuffKalX[0]=BuffObsX[0]
BuffKalV[0]=BuffObsV[0]

        

AxI=int(input('Ingrese aceleración x inicial: '))
deltat=int(input('Ingrese el delta t: '))


#Leer errores del proceso para matriz de covarianza

ErrPx=float(input('Ingrese el error en el proceso en Posición: '))
ErrVx=float(input('Ingrese el error en el proceso en Velocidad: '))

ObsErrPx=float(input('Ingrese el error de observación Posición: '))
ObsErrVx=float(input('Ingrese el error de observación Velocidad: '))

#Se define Matriz A
matrixA=np.ones((2, 2)) # Matriz de unos
matrixA[0,1]=deltat;
matrixA[1,0]=0;
print('matrixA', matrixA)

#Se define Matriz X(k-1) con los primeros valores de observación
matrixXk_1=np.ones((2, 1))

matrixXk_1[0,0]=BuffObsX[0]
matrixXk_1[1,0]=BuffObsV[0]
print('matrixXk_1', matrixXk_1)

#Se define Matriz B
matrixB=np.ones((2, 1)) # Matriz de unos
matrixB[0,0]=0.5*(deltat**2);
matrixB[1,0]=deltat;
print('matrixB', matrixB)

#Se define Matriz Uk
matrixUk=np.ones((1, 1)) # Matriz de unos
matrixUk[0,0]=AxI;
print('matrixUk', matrixUk)

####################################################
##STEP 2. INITAL PROCESS COVARIANCE MATRIX (Pk_1) - V 28
####################################################

#Se define la Initial Process Covariance
Pk_1=np.zeros((2,2))
Pk_1[0,0]=ErrPx*ErrPx
Pk_1[1,1]=ErrVx*ErrVx

print('Initial Process Covariance Matrix',Pk_1)

i=0

for i in range (CantObs-1):
    ####################################################
    ##STEP 1. PREDICTED STATE (Xkp) - V 27
    ####################################################

    #PREDICTED STATE RESULT (Xkp)
    matrixXkp=np.dot(matrixA,matrixXk_1)+np.dot(matrixB,matrixUk)
    print('\n\n\n\nmatrixA', matrixA)
    print('matrixXk_1', matrixXk_1)
    print('matrixB', matrixB)
    print('matrixXk_1', matrixXk_1)
    print('matrixUk', matrixUk)
    print('matrixXkp\n', matrixXkp)

    ####################################################
    ##STEP 3. PREDICTED PROCESS COVARIANCE MATRIX (Pkp) - V 29
    ####################################################
    print('matrixA', matrixA)
    print('Pk_1', Pk_1)
    Pkp = np.dot(matrixA,Pk_1)
    print('Predicted Process Covariance Matrix 0',Pkp)
    Pkp = np.dot(Pkp,np.transpose(matrixA))
    print('Predicted Process Covariance Matrix 1',Pkp)
    Pkp[0,1]=0
    Pkp[1,0]=0

    print('Predicted Process Covariance Matrix 2',Pkp)

    ####################################################
    ##STEP 4. KALMAN GAIN (KGain) - V 30
    ####################################################

    matrixH=np.zeros((2,2))
    matrixH[0,0]=1
    matrixH[1,1]=1
    matrixR=np.zeros((2,2))
    matrixR[0,0]=ObsErrPx*ObsErrPx
    matrixR[1,1]=ObsErrVx*ObsErrVx

    KGainNum=np.dot(Pkp,matrixH)
    KGainDen=np.dot(KGainNum,matrixH)+matrixR
    KGain = np.divide(KGainNum,KGainDen)
    KGain[0,1]=0
    KGain[1,0]=0

    print('Kalman Gain ',KGain )

    ####################################################
    ##STEP 5. THE NEW OBSERVATION (Ykm) - V:31
    ####################################################
    matrixC=np.zeros((2,2))
    matrixC[0,0]=1
    matrixC[1,1]=1
    matrixYkm=np.zeros((2,1))

    if(i<CantObs):

        matrixYkm[0,0]=BuffObsX[i+1]
        matrixYkm[1,0]=BuffObsV[i+1]
        Yk=np.dot(matrixC,matrixYkm)
        print('Yk',Yk)

        ####################################################
        ##STEP 6. THE CURRENT STATE (Xk)- V:32
        ####################################################

        Xk=Yk-np.dot(matrixH,matrixXkp)
        Xk=np.dot(KGain,Xk)

        Xk=matrixXkp+Xk
        print('Xk',Xk)

        ####################################################
        ##STEP 7. UPDATING THE PROCESS COVARIANCE MATRIX (Pk)- V:33
        ####################################################

        Pk=matrixH-np.dot(KGain,matrixH)
        Pk=np.dot(Pk,Pkp)
        print('Pk',Pk)

        BuffPredX[i+1]=matrixXkp[0,0]
        BuffPredV[i+1]=matrixXkp[1,0]
        BuffKalX[i+1]=Xk[0,0]
        BuffKalV[i+1]=Xk[1,0]
    
        #DEFINE NEW PREVIUS VALUES

        matrixXk_1=Xk
        Pk_1=Pk

print('\n\n\n\n\nResultado')
print('Valores del estado de Predicción (Posición), Xkp = ',BuffPredX)
print('Valores del estado de Predicción (Velocidad), Xkp = ',BuffPredV)
print('Valores del Estado Actual - Filtro Kalman (Posición), Xk = ',BuffKalX)
print('Valores del Estado Actual - Filtro Kalman (Velocidad), Xk = ',BuffKalV)
print('\nBuffObsX',BuffObsX)
print('BuffObsV',BuffObsV)

time=np.arange(CantObs)
print('time',time)

plot(time,BuffPredX, 'o-',  label='Estado de Predicción (Xkp)')
plot(time,BuffKalX, '+-', label='Estado Actual - Filtro Kalman (Xk)')
plot(time,BuffObsX, '*-', label='Valores Observados')
legend(loc='lower right')
xlabel('Tiempo')
ylabel('Posición')
title('Filtro de Kalman\nPosición vs Tiempo')
draw()
savefig("Filtro_Kalman_1",dpi=300)

figure()
plot(time,BuffPredV, 'o-', label='Estado de Predicción (Xkp)')
plot(time,BuffKalV, '+-', label='Estado Actual - Filtro Kalman (Xk)')
plot(time,BuffObsV, '*-', label='Valores Observados')
legend(loc='lower right')
xlabel('Tiempo')
ylabel('Velocidad')
title('Filtro de Kalman\nVelocidad vs Tiempo')
draw()
savefig("Filtro_Kalman_2",dpi=300)