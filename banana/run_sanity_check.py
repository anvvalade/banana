#!/usr/bin/env python3

from tensorphy import FFT3DWrapper, LogServer

LogServer.cannot_save_error = False


FFT3D = FFT3DWrapper(4, 1, verbosity=3)
FFT3D.test()
