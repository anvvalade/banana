#!/usr/bin/env python3

from tensorphy import FFT3DWrapper, CloudInCell


FFT3D = FFT3DWrapper(4, 1, verbosity=3)
FFT3D.test()

CIC = CloudInCell(4, 1, .3, [-1, 0, 1], verbosity=3)
CIC.test(method_test='uniform')
