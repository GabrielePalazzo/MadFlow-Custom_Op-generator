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

FFV1P0_3_signature = [
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None,None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX),
tf.TensorSpec(shape=[], dtype=DTYPE),
tf.TensorSpec(shape=[], dtype=DTYPE),
]

@tf.function(input_signature=FFV1P0_3_signature)
def FFV1P0_3(F1,F2,COUP,M3,W3):
    cI = complex_tf(0,1)
    COUP = complex_me(COUP)
    M3 = complex_me(M3)
    W3 = complex_me(W3)
    V3 = [complex_tf(0,0)] * 6
    V3[0] = F1[0]+F2[0]
    V3[1] = F1[1]+F2[1]
    P3 = complex_tf(tf.stack([-tf.math.real(V3[0]), -tf.math.real(V3[1]), -tf.math.imag(V3[1]), -tf.math.imag(V3[0])], axis=0), 0.)
    denom = COUP/(P3[0]**2-P3[1]**2-P3[2]**2-P3[3]**2 - M3 * (M3 -cI* W3))
    V3[2]= denom*(-cI)*(F1[2]*F2[4]+F1[3]*F2[5]+F1[4]*F2[2]+F1[5]*F2[3])
    V3[3]= denom*(-cI)*(-F1[2]*F2[5]-F1[3]*F2[4]+F1[4]*F2[3]+F1[5]*F2[2])
    V3[4]= denom*(-cI)*(-cI*(F1[2]*F2[5]+F1[5]*F2[2])+cI*(F1[3]*F2[4]+F1[4]*F2[3]))
    V3[5]= denom*(-cI)*(-F1[2]*F2[4]-F1[5]*F2[3]+F1[3]*F2[5]+F1[4]*F2[2])
    return tf.stack(V3, axis=0)


