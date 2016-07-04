# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:32:00 2016

@author: jacquemont
""""

def antsRegistrationSyNQuick(fixe_image, moving_image):

        import subprocess
        import os.path as op

        image_warped = op.abspath('SyN_QuickWarped.nii.gz')
        affine_matrix = op.abspath('SyN_Quick0GenericAffine.mat')
        warp = op.abspath('SyN_Quick1Warp.nii.gz')
        inverse_warped = op.abspath('SyN_QuickInverseWarped.nii.gz')
        inverse_warp = op.abspath('SyN_Quick1InverseWarp.nii.gz')

        cmd = 'antsRegistrationSyNQuick.sh -t br -d 3 -f ' + fixe_image + ' -m ' + moving_image + ' -o SyN_Quick'
        subprocess.call([cmd], shell=True)

        return image_warped, affine_matrix, warp, inverse_warped, inverse_warp