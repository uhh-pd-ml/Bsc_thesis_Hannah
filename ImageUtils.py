#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Library for image preprocessing
"""


# =============================================================================
# I M P O R T S
# =============================================================================

import numpy as np
import math
import ctypes
import ROOT


# =============================================================================
# C O N S T A N T S
# =============================================================================

(pT_i, eta_i, phi_i) = (0, 1, 2)
jet_shape = (100, 3)


# =============================================================================
# F U N C T I O N S
# =============================================================================

def ang_dist(phi1, phi2):
    """
    Returns 'distance' between two azimuthal angles.

            Parameters:
                    phi1 (float): first azimuthal angle
                    phi2 (float): second azimuthal angle

            Returns:
                    dphi (float): angle distance
    """
    dphi = phi1 - phi2
    if(dphi < -math.pi):
        dphi += 2 * math.pi
    if(dphi > math.pi):
        dphi -= 2 * math.pi
    return dphi


def raw_moment(jet, eta_order, phi_order):
    """
    Returns raw moment.

            Parameters:
                    jet (np.ndarray): jet as numpy array in the style of
                                      [particle, particle, ...]
                    eta_order (int): order of eta
                    phi_order (int): order of phi

            Returns:
                    dphi (float): Angle distance
    """
    eta_factor = jet[:,eta_i] ** eta_order
    phi_factor = jet[:,phi_i] ** phi_order
    raw_moment = (jet[:,pT_i] * eta_factor * phi_factor).sum()
    return raw_moment


def image_center(jet):
    """
    Returns pT-weighted average of the jet.

            Parameters:
                    jet (np.ndarray): jet as numpy array in the style of
                                      [particle, particle, ...]

            Returns:
                    eta_avg (float): eta coordinate of the image center
                    phi_avg (float): phi coordinate of the image center
    """
    pt_sum = np.sum(jet[:,pT_i])
    m10 = raw_moment(jet, 1,0)
    m01 = raw_moment(jet, 0,1)

    eta_avg = m10 / pt_sum
    phi_avg = m01 / pt_sum

    return eta_avg, phi_avg


def rot_flip(jet):
    """
    Rotates and flips image such that the eigenvector with highest eigenvalue
    of the covariance matrix is parallel to phi and that the particle with
    highest pT has both positive eta and positive phi.

            Parameters:
                    jet (np.ndarray): jet as numpy array in the style of
                                      [particle, particle, ...]

            Returns:
                    new_jet (np.ndarray): jet as numpy array in the style of
                                          [particle, particle, ...]
    """
    coords = np.vstack([jet[:,eta_i], jet[:,phi_i]])
    cov = np.cov(coords, aweights=jet[:,pT_i])
    evals, evecs = np.linalg.eig(cov)

    e_max = np.argmax(evals)
    eta_v1, phi_v1 = evecs[:, e_max]

    theta = np.arctan((eta_v1)/(phi_v1))
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    transformed_mat = rotation_mat * coords
    eta_transformed, phi_transformed = transformed_mat.A

    transformed_v1 = rotation_mat * np.vstack([eta_v1, phi_v1])
    t_eta_v1, t_phi_v1 = transformed_v1.A

    argmax_pt = np.argmax(jet[:,pT_i])
    ptmax_eta = eta_transformed[argmax_pt]
    ptmax_phi = phi_transformed[argmax_pt]
    if(ptmax_eta < 0): eta_transformed *= -1
    if(ptmax_phi < 0): phi_transformed *= -1

    new_jet = np.stack((jet[:,pT_i], eta_transformed, phi_transformed)).T

    return new_jet


def pixelate(jet, npix=33, img_width=1.0, rotate = True, norm=True):
    """
    Makes a pixelated image out of particle coordinates.

            Parameters:
                    jet (np.ndarray): jet as numpy array in the style of
                                      [particle, particle, ...]
                    npix (int): the image will be created to have npix * npix
                                pixels
                    img_width (float): image size in real units of eta and phi
                    rotate (bool): should images be rotated and flipped?
                    norm (bool): should images be L1-normalized?

            Returns:
                    jet_image (np.ndarray): preprocessed image
    """
    # the image is (img_width * img_width) in size
    pix_width = img_width / npix
    jet_image = np.zeros((npix, npix), dtype = np.float16)

    # remove particles with zero pt
    jet = jet[jet[:,pT_i] > 0]

    if np.average(np.abs(jet[:, phi_i])) > 2.5:
        # jet likely wraps in phi - make all phi positive
        jet[jet[:, phi_i] < 0, phi_i] += 2 * np.pi

    # center image
    eta_avg, phi_avg = image_center(jet)
    jet[:, eta_i] -= eta_avg
    jet[:, phi_i] -= phi_avg

    if rotate and jet.shape[0] > 2:
        jet_transformed = rot_flip(jet)

    # Transformations like in Kristian's bachelor thesis are done here.
    # Don't forget to recenter and to rerotate.

    # transition to indices
    mid_pix = np.floor(npix/2)
    eta_indices = mid_pix + np.ceil(jet_transformed[:,eta_i]/pix_width - 0.5)
    phi_indices = mid_pix + np.ceil(jet_transformed[:,phi_i]/pix_width - 0.5)

    # delete elements outside of range
    mask = np.ones(eta_indices.shape).astype(bool)
    mask[eta_indices < 0] = False
    mask[phi_indices < 0] = False
    mask[eta_indices >= npix] = False
    mask[phi_indices >= npix] = False

    eta_indices = eta_indices[mask].astype(int)
    phi_indices = phi_indices[mask].astype(int)

    # construct grayscale image
    for pt,eta,phi in zip(jet_transformed[:,pT_i][mask],
                          eta_indices,
                          phi_indices): 
        jet_image[phi, eta] += pt

    # L1-normalize the pT channels of the jet image
    if norm:
        normfactor = np.sum(jet_image)
        if normfactor < 0:
            print('normfactor=',normfactor)
            print(jet)
            print(jet_transformed)
            print(eta_indices,phi_indices)
            print(jet_image)
            raise FloatingPointError('Image had no particles!')
        elif normfactor > 0: 
            jet_image /= normfactor

    return jet_image

def make_image(jet_conts, npix, img_width=1.0, rotate=True, norm=True):
    """
    Makes images.

            Parameters:
                    jet_conts (np.ndarray): jet as a list or 1D array in the
                                            shape [pT, eta, phi,
                                                   pT, eta, phi,
                                                   ...]
                    npix (int): the image will be created to have npix * npix
                                pixels
                    img_width (float): image size in real units of eta and phi
                    rotate (bool): should images be rotated and flipped?
                    norm (bool): should images be L1-normalized?

            Returns:
                    image as np.ndarray
    """
    jet_conts = jet_conts.reshape(jet_shape)
    return np.squeeze(pixelate(jet_conts,
                               npix=npix,
                               img_width=img_width,
                               rotate=rotate,
                               norm=norm))


# =============================================================================
# M A I N
# =============================================================================

if __name__ == '__main__':
    print('This doesn\'t do anything on its own. Use make_jet_images.py.')
