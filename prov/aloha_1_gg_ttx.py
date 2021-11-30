from madflow.config import DTYPE, DTYPEINT, complex_tf, complex_me, DTYPECOMPLEX
import tensorflow as tf

VVV1P0_1_signature = [
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[], dtype=DTYPE),
tf.TensorSpec(shape=[], dtype=DTYPE),
]

@tf.function(input_signature=VVV1P0_1_signature)
def VVV1P0_1(V2,V3,COUP,M1,W1):
    cI = complex_tf(0,1)
    COUP = complex_me(COUP)
    M1 = complex_me(M1)
    W1 = complex_me(W1)
    P2 = complex_tf(tf.stack([tf.math.real(V2[0]), tf.math.real(V2[1]), tf.math.imag(V2[1]), tf.math.imag(V2[0])], axis=0), 0.)
    P3 = complex_tf(tf.stack([tf.math.real(V3[0]), tf.math.real(V3[1]), tf.math.imag(V3[1]), tf.math.imag(V3[0])], axis=0), 0.)
    V1 = [complex_tf(0,0)] * 6
    V1[0] = V2[0]+V3[0]
    V1[1] = V2[1]+V3[1]
    P1 = complex_tf(tf.stack([-tf.math.real(V1[0]), -tf.math.real(V1[1]), -tf.math.imag(V1[1]), -tf.math.imag(V1[0])], axis=0), 0.)
    TMP0 = (V3[2]*P1[0]-V3[3]*P1[1]-V3[4]*P1[2]-V3[5]*P1[3])
    TMP1 = (V3[2]*P2[0]-V3[3]*P2[1]-V3[4]*P2[2]-V3[5]*P2[3])
    TMP2 = (P1[0]*V2[2]-P1[1]*V2[3]-P1[2]*V2[4]-P1[3]*V2[5])
    TMP3 = (V2[2]*P3[0]-V2[3]*P3[1]-V2[4]*P3[2]-V2[5]*P3[3])
    TMP4 = (V3[2]*V2[2]-V3[3]*V2[3]-V3[4]*V2[4]-V3[5]*V2[5])
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -cI* W1))
    V1[2]= denom*(TMP4*(-cI*(P2[0])+cI*(P3[0]))+(V2[2]*(-cI*(TMP0)+cI*(TMP1))+V3[2]*(cI*(TMP2)-cI*(TMP3))))
    V1[3]= denom*(TMP4*(-cI*(P2[1])+cI*(P3[1]))+(V2[3]*(-cI*(TMP0)+cI*(TMP1))+V3[3]*(cI*(TMP2)-cI*(TMP3))))
    V1[4]= denom*(TMP4*(-cI*(P2[2])+cI*(P3[2]))+(V2[4]*(-cI*(TMP0)+cI*(TMP1))+V3[4]*(cI*(TMP2)-cI*(TMP3))))
    V1[5]= denom*(TMP4*(-cI*(P2[3])+cI*(P3[3]))+(V2[5]*(-cI*(TMP0)+cI*(TMP1))+V3[5]*(cI*(TMP2)-cI*(TMP3))))
    return tf.stack(V1, axis=0)


from madflow.config import DTYPE, DTYPEINT, complex_tf, complex_me, DTYPECOMPLEX
import tensorflow as tf

FFV1_0_signature = [
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX),
]

@tf.function(input_signature=FFV1_0_signature)
def FFV1_0(F1,F2,V3,COUP):
    cI = complex_tf(0,1)
    COUP = complex_me(COUP)
    TMP5 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))+(F1[3]*(F2[4]*(V3[3]-cI*(V3[4]))+F2[5]*(V3[2]-V3[5]))+(F1[4]*(F2[2]*(V3[2]-V3[5])-F2[3]*(V3[3]+cI*(V3[4])))+F1[5]*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5])))))
    vertex = COUP*-cI * TMP5
    return vertex


from madflow.config import DTYPE, DTYPEINT, complex_tf, complex_me, DTYPECOMPLEX
import tensorflow as tf

FFV1_1_signature = [
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[], dtype=DTYPE),
tf.TensorSpec(shape=[], dtype=DTYPE),
]

@tf.function(input_signature=FFV1_1_signature)
def FFV1_1(F2,V3,COUP,M1,W1):
    cI = complex_tf(0,1)
    COUP = complex_me(COUP)
    M1 = complex_me(M1)
    W1 = complex_me(W1)
    F1 = [complex_tf(0,0)] * 6
    F1[0] = F2[0]+V3[0]
    F1[1] = F2[1]+V3[1]
    P1 = complex_tf(tf.stack([-tf.math.real(F1[0]), -tf.math.real(F1[1]), -tf.math.imag(F1[1]), -tf.math.imag(F1[0])], axis=0), 0.)
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -cI* W1))
    F1[2]= denom*cI*(F2[2]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]-cI*(V3[4]))+(P1[2]*(cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-1./1.)*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))))
    F1[3]= denom*(-cI)*(F2[2]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))+P1[3]*(V3[3]-cI*(V3[4])))))+(F2[3]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P1[2]*(cI*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+M1*(F2[4]*(-V3[3]+cI*(V3[4]))+F2[5]*(-V3[2]+V3[5]))))
    F1[4]= denom*(-cI)*(F2[4]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-V3[3]+cI*(V3[4]))+(P1[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[5]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-V3[2]+V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))-P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[2]*(-V3[2]+V3[5])+F2[3]*(V3[3]+cI*(V3[4])))))
    F1[5]= denom*cI*(F2[4]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(-V3[3]+cI*(V3[4])))))+(F2[5]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]+cI*(V3[4]))+(P1[2]*(-cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+M1*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5]))))
    return tf.stack(F1, axis=0)


from madflow.config import DTYPE, DTYPEINT, complex_tf, complex_me, DTYPECOMPLEX
import tensorflow as tf

FFV1_2_signature = [
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[], dtype=DTYPE),
tf.TensorSpec(shape=[], dtype=DTYPE),
]

@tf.function(input_signature=FFV1_2_signature)
def FFV1_2(F1,V3,COUP,M2,W2):
    cI = complex_tf(0,1)
    COUP = complex_me(COUP)
    M2 = complex_me(M2)
    W2 = complex_me(W2)
    F2 = [complex_tf(0,0)] * 6
    F2[0] = F1[0]+V3[0]
    F2[1] = F1[1]+V3[1]
    P2 = complex_tf(tf.stack([-tf.math.real(F2[0]), -tf.math.real(F2[1]), -tf.math.imag(F2[1]), -tf.math.imag(F2[0])], axis=0), 0.)
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -cI* W2))
    F2[2]= denom*cI*(F1[2]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[2]*(cI*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+(F1[3]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-V3[2]+V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(-V3[3]+cI*(V3[4])))))+M2*(F1[4]*(V3[2]-V3[5])+F1[5]*(-V3[3]+cI*(V3[4])))))
    F2[3]= denom*(-cI)*(F1[2]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))-P2[3]*(V3[3]+cI*(V3[4])))))+(F1[3]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]-cI*(V3[4]))+(P2[2]*(cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+M2*(F1[4]*(V3[3]+cI*(V3[4]))-F1[5]*(V3[2]+V3[5]))))
    F2[4]= denom*(-cI)*(F1[4]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]+cI*(V3[4]))+(P2[2]*(-cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-1./1.)*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))+P2[3]*(V3[3]-cI*(V3[4])))))+M2*(F1[2]*(-1./1.)*(V3[2]+V3[5])+F1[3]*(-V3[3]+cI*(V3[4])))))
    F2[5]= denom*cI*(F1[4]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]-V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(V3[3]+cI*(V3[4])))))+(F1[5]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-V3[3]+cI*(V3[4]))+(P2[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[2]*(V3[3]+cI*(V3[4]))+F1[3]*(V3[2]-V3[5]))))
    return tf.stack(F2, axis=0)


