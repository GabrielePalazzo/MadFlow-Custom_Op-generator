
from madflow.config import (
    int_me,
    float_me,
    DTYPE,
    DTYPEINT,
    run_eager,
    complex_tf,
    complex_me
)
from madflow.wavefunctions_flow import oxxxxx, ixxxxx, vxxxxx, sxxxxx
from madflow.parameters import Model

import os
import sys
import numpy as np

import tensorflow as tf
import collections

ModelParamTupleConst = collections.namedtuple("constants", ["mdl_MT","mdl_WT"])
ModelParamTupleFunc = collections.namedtuple("functions", ["GC_11"])

root_path = '/home/palazzo/MG5_aMC_v3_1_0'
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'madgraph'))
sys.path.insert(0, os.path.join(root_path, 'aloha', 'template_files'))

import models.import_ufo as import_ufo
import models.check_param_card as param_card_reader

# import the ALOHA routines
from aloha_1_uux_ttx import *


def get_model_param(model, param_card_path):
    param_card = param_card_reader.ParamCard(param_card_path)
        # External (param_card) parameters
    aEWM1 = param_card['SMINPUTS'].get(1).value
    mdl_Gf = param_card['SMINPUTS'].get(2).value
    aS = param_card['SMINPUTS'].get(3).value
    mdl_ymb = param_card['YUKAWA'].get(5).value
    mdl_ymt = param_card['YUKAWA'].get(6).value
    mdl_ymtau = param_card['YUKAWA'].get(15).value
    mdl_MZ = param_card['MASS'].get(23).value
    mdl_MT = param_card['MASS'].get(6).value
    mdl_MB = param_card['MASS'].get(5).value
    mdl_MH = param_card['MASS'].get(25).value
    mdl_MTA = param_card['MASS'].get(15).value
    mdl_WZ = param_card['DECAY'].get(23).value
    mdl_WW = param_card['DECAY'].get(24).value
    mdl_WT = param_card['DECAY'].get(6).value
    mdl_WH = param_card['DECAY'].get(25).value

    #PS-independent parameters
    mdl_conjg__CKM3x3 = 1.0
    mdl_CKM3x3 = 1.0
    mdl_conjg__CKM1x1 = 1.0
    ZERO = 0.0
    mdl_complexi = complex(0,1)
    mdl_MZ__exp__2 = mdl_MZ**2
    mdl_MZ__exp__4 = mdl_MZ**4
    mdl_sqrt__2 =  np.sqrt(2) 
    mdl_MH__exp__2 = mdl_MH**2
    mdl_aEW = 1/aEWM1
    mdl_MW = np.sqrt(mdl_MZ__exp__2/2. + np.sqrt(mdl_MZ__exp__4/4. - (mdl_aEW*np.pi*mdl_MZ__exp__2)/(mdl_Gf*mdl_sqrt__2)))
    mdl_sqrt__aEW =  np.sqrt(mdl_aEW) 
    mdl_ee = 2*mdl_sqrt__aEW*np.sqrt(np.pi)
    mdl_MW__exp__2 = mdl_MW**2
    mdl_sw2 = 1 - mdl_MW__exp__2/mdl_MZ__exp__2
    mdl_cw = np.sqrt(1 - mdl_sw2)
    mdl_sqrt__sw2 =  np.sqrt(mdl_sw2) 
    mdl_sw = mdl_sqrt__sw2
    mdl_g1 = mdl_ee/mdl_cw
    mdl_gw = mdl_ee/mdl_sw
    mdl_vev = (2*mdl_MW*mdl_sw)/mdl_ee
    mdl_vev__exp__2 = mdl_vev**2
    mdl_lam = mdl_MH__exp__2/(2.*mdl_vev__exp__2)
    mdl_yb = (mdl_ymb*mdl_sqrt__2)/mdl_vev
    mdl_yt = (mdl_ymt*mdl_sqrt__2)/mdl_vev
    mdl_ytau = (mdl_ymtau*mdl_sqrt__2)/mdl_vev
    mdl_muH = np.sqrt(mdl_lam*mdl_vev__exp__2)
    mdl_I1x33 = mdl_yb*mdl_conjg__CKM3x3
    mdl_I2x33 = mdl_yt*mdl_conjg__CKM3x3
    mdl_I3x33 = mdl_CKM3x3*mdl_yt
    mdl_I4x33 = mdl_CKM3x3*mdl_yb
    mdl_ee__exp__2 = mdl_ee**2
    mdl_sw__exp__2 = mdl_sw**2
    mdl_cw__exp__2 = mdl_cw**2

    #PS-dependent parameters
    mdl_sqrt__aS =  np.sqrt(aS) 
    mdl_G__exp__2 = lambda G: complex_me(G**2)


    # PS-dependent couplings
    GC_11 = lambda G: complex_me(mdl_complexi*G)

    constants = ModelParamTupleConst(float_me(mdl_MT),float_me(mdl_WT))
    functions = ModelParamTupleFunc(GC_11)
    return Model(constants, functions)



smatrix_signature = [
        tf.TensorSpec(shape=[None,4,4], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX)
        ]


matrix_signature = [
        tf.TensorSpec(shape=[None,4,4], dtype=DTYPE),
        tf.TensorSpec(shape=[4], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[], dtype=DTYPE),
        tf.TensorSpec(shape=[None], dtype=DTYPECOMPLEX)
        ]


class Matrix_1_uux_ttx(object):
    nexternal = float_me(4)
    ndiags = float_me(1)
    ncomb = float_me(16)
    initial_states = [[2, -2],[4, -4],[1, -1],[3, -3]]
    mirror_initial_states = True
    helicities = float_me([ \
        [1,-1,-1,1],
        [1,-1,-1,-1],
        [1,-1,1,1],
        [1,-1,1,-1],
        [1,1,-1,1],
        [1,1,-1,-1],
        [1,1,1,1],
        [1,1,1,-1],
        [-1,-1,-1,1],
        [-1,-1,-1,-1],
        [-1,-1,1,1],
        [-1,-1,1,-1],
        [-1,1,-1,1],
        [-1,1,-1,-1],
        [-1,1,1,1],
        [-1,1,1,-1]])
    denominator = float_me(36)

    def __init__(self):
        """define the object"""
        self.clean()

    def clean(self):
        pass
        ##self.jamp = []

    def __str__(self):
        return "1_uux_ttx"

    @tf.function(input_signature=smatrix_signature)
    def smatrix(self,all_ps,mdl_MT,mdl_WT,GC_11):
        #  
        #  MadGraph5_aMC@NLO v. 3.1.0, 2021-03-30
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        # Process: u u~ > t t~ WEIGHTED<=2 @1
        # Process: c c~ > t t~ WEIGHTED<=2 @1
        # Process: d d~ > t t~ WEIGHTED<=2 @1
        # Process: s s~ > t t~ WEIGHTED<=2 @1
        #  
        # Clean additional output
        #
        ###self.clean()
        # ----------
        # BEGIN CODE
        # ----------
        nevts = tf.shape(all_ps, out_type=DTYPEINT)[0]
        ans = tf.zeros(nevts, dtype=DTYPE)
        for hel in self.helicities:
            ans += self.matrix(all_ps,hel,mdl_MT,mdl_WT,GC_11)

        return ans/self.denominator
    def cusmatrix(self,all_ps,mdl_MT,mdl_WT,GC_11):
        #  
        #  MadGraph5_aMC@NLO v. 3.1.0, 2021-03-30
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        # 
        # MadGraph5_aMC@NLO StandAlone Version
        # 
        # Returns amplitude squared summed/avg over colors
        # and helicities
        # for the point in phase space P(0:3,NEXTERNAL)
        #  
        # Process: u u~ > t t~ WEIGHTED<=2 @1
        # Process: c c~ > t t~ WEIGHTED<=2 @1
        # Process: d d~ > t t~ WEIGHTED<=2 @1
        # Process: s s~ > t t~ WEIGHTED<=2 @1
        #  
        # Clean additional output
        #
        ###self.clean()
        # ----------
        # BEGIN CODE
        # ----------
        nevts = tf.shape(all_ps, out_type=DTYPEINT)[0]
        ans = tf.zeros(nevts, dtype=DTYPE)
        matrixOp = tf.load_op_library('./matrix_uux_ttx_cu.so')
        for hel in self.helicities:
            ans += matrixOp.matrixuuxttx(all_ps,hel,mdl_MT,mdl_WT,GC_11)

        return ans/self.denominator
        return ans/self.denominator

    @tf.function(input_signature=matrix_signature)
    def matrix(self,all_ps,hel,mdl_MT,mdl_WT,GC_11):
        #  
        #  MadGraph5_aMC@NLO v. 3.1.0, 2021-03-30
        #  By the MadGraph5_aMC@NLO Development Team
        #  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
        #
        # Returns amplitude squared summed/avg over colors
        # for the point with external lines W(0:6,NEXTERNAL)
        #
        # Process: u u~ > t t~ WEIGHTED<=2 @1
        # Process: c c~ > t t~ WEIGHTED<=2 @1
        # Process: d d~ > t t~ WEIGHTED<=2 @1
        # Process: s s~ > t t~ WEIGHTED<=2 @1
        #  
        #  
        # Process parameters
        #  
        ngraphs = 1
        nwavefuncs = 5
        ncolor = 2
        ZERO = float_me(0.)
        #  
        # Color matrix
        #  
        denom = tf.constant([1,1], dtype=DTYPECOMPLEX)
        cf = tf.constant([[9,3],
                          [3,9]], dtype=DTYPECOMPLEX)
        #
        # Model parameters
        #
        # ----------
        # Begin code
        # ----------
        w0 = ixxxxx(all_ps[:,0],ZERO,hel[0],float_me(+1))
        w1 = oxxxxx(all_ps[:,1],ZERO,hel[1],float_me(-1))
        w2 = oxxxxx(all_ps[:,2],mdl_MT,hel[2],float_me(+1))
        w3 = ixxxxx(all_ps[:,3],mdl_MT,hel[3],float_me(-1))
        w4= FFV1P0_3(w0,w1,GC_11,ZERO,ZERO)
        # Amplitude(s) for diagram number 1
        amp0= FFV1_0(w3,w2,w4,GC_11)

        jamp = tf.stack([1./2.*(1./3.*amp0),+1./2.*(-amp0)], axis=0)

        ret = tf.einsum("ie, ij, je -> e", jamp, cf, tf.math.conj(jamp)/tf.reshape(denom, (ncolor, 1)))
        return tf.math.real(ret)


if __name__ == "__main__":
    import sys, pathlib
    import numpy as np

    # Get the name of the matrix in this file
    matrix_name = pathlib.Path(sys.argv[0]).stem.capitalize()
    matrix_class = globals()[matrix_name]

    # Instantiate the matrix
    matrix = matrix_class()

    # Read up the model
    model_sm = pathlib.Path(root_path) / "models/sm"
    if not model_sm.exists():
        print(f"No model sm found at {model_sm}, test cannot continue")
        sys.exit(0)
    model = import_ufo.import_model(model_sm.as_posix())
    model_params = get_model_param(model, 'Cards/param_card.dat')

    # Define th phase space
    # The structure asked by the matrix elements is
    #   (nevents, ndimensions, nparticles)
    # the 4 dimensions of the 4-momentum is expected as
    #   (E, px, py, pz)
    ndim = 4
    npar = int(matrix.nexternal)
    nevents = 2
    max_momentum = 7e3

    par_ax = 1
    dim_ax = 2

    # Now generate random outgoing particles in a com frame (last_p carries whatever momentum is needed to sum 0 )
    shape = [nevents, 0, 0]
    shape[par_ax] = npar - 3
    shape[dim_ax] = ndim - 1
    partial_out_p = tf.random.uniform(shape, minval=-max_momentum, maxval=max_momentum, dtype=DTYPE)
    last_p = -tf.reduce_sum(partial_out_p, keepdims=True, axis=par_ax)
    out_p = tf.concat([partial_out_p, last_p], axis=par_ax)

    if "mdl_MT" in dir(model_params):
        # TODO fill in the mass according to the particles
        out_m = tf.reshape((npar - 2) * [model_params.mdl_MT], (1, -1, 1))
    else:
        out_m = 0.0
    out_e = tf.sqrt(tf.reduce_sum(out_p ** 2, keepdims=True, axis=dim_ax) + out_m ** 2)
    outgoing_4m = tf.concat([out_e, out_p], axis=dim_ax)

    # Put all incoming momenta in the z axis (TODO: for now assume massless input)
    ea = tf.reduce_sum(out_e, axis=par_ax, keepdims=True) / 2
    zeros = tf.zeros_like(ea)
    inc_p1 = tf.concat([ea, zeros, zeros, ea], axis=dim_ax)
    inc_p2 = tf.concat([ea, zeros, zeros, -ea], axis=dim_ax)

    all_ps = tf.concat([inc_p1, inc_p2, outgoing_4m], axis=par_ax)

    model_params.freeze_alpha_s(0.118)
    wgt_set = matrix.smatrix(all_ps, *model_params.evaluate(None))

    print("All good!")
    for i, (p, wgt) in enumerate(zip(all_ps, wgt_set)):
        print(f"\n#{i} ME value: {wgt.numpy():.3e} for P set:\n{p.numpy()}")
