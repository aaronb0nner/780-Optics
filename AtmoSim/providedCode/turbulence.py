import math
import numpy as np
import csv
import numpy.linalg as la
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    frames = 10  # number of frames
    nn = 128  # image length in pixels
    d = .5  # pupil diameter
    ro = d / 10  # fried
    l1 = d * 2  # edge length of image
    deltat = 1  # time between screens
    zern_mx = 120  # number of zernikes used (max 1000)
    windx = 6  # wind speed x direction
    windy = 6  # wind speed y direction
    boil = 1  # atmospheric boil factor

    # get zernike polynomials
    zern, ch = generate_zern_polys(zern_mx, int(nn / 2), d, ro)

    # make phase screen from zernikes
    phase_screens, x_crd, zern_pad = make_screens(frames, nn, d, ro, l1, deltat, windx, windy, boil, zern, ch, zern_mx)

    # display a phase screen
    plt.imshow(phase_screens[0, :, :])
    plt.show()

# get zernike polynomials and cholesky decomposition of zernike covariance matrix
def generate_zern_polys(zern_mx, nn, d, ro):
    k = 2.2698

    # create zernike
    zern, idx = zern_poly(zern_mx + 1, nn)
    zern = zern[1: zern_mx, :, :]  # removes the piston zernike

    # transfer indices
    n = idx[:, 0]
    m = idx[:, 1]
    p = idx[:, 2]

    # calculate covariance matrix
    covar = np.zeros((zern_mx, zern_mx))
    for xx in np.arange(0, zern_mx):
        for yy in np.arange(0, zern_mx):
            test1 = m[xx] == m[yy]
            test2 = m[xx] == 0
            temp_frac = (p[xx] / 2) / np.ceil(p[xx] / 2)
            p_even = temp_frac == 1
            temp_frac = (p[yy] / 2) / np.ceil(p[yy] / 2)
            p_p_even = temp_frac == 1
            test3 = p_even == p_p_even
            test0 = test2 | test3
            if test1 and test0:
                k_zz = k * (-1) ** ((n[xx] + n[yy] - (2 * m[xx])) / 2) * np.sqrt((n[xx] + 1) * (n[yy] + 1))
                num = k_zz * math.gamma((n[xx] + n[yy] - (5 / 3)) / 2) * (d / ro) ** (5 / 3)
                dnm = math.gamma((n[xx] - n[yy] + 17 / 3) / 2) * math.gamma((n[yy] - n[xx] + 17 / 3) / 2) * \
                      math.gamma((n[xx] + n[yy] + 23 / 3) / 2)
                covar[xx, yy] = num / dnm

    # factorize covariance matrix using cholesky
    covar = covar[1:zern_mx, 1:zern_mx]
    ch = la.cholesky(covar)

    return zern, ch


# get zernike polynomial from zernike indexes
def zern_poly(i_mx, num_pts):
    # define coordinates
    del_x = (1 / num_pts) * 2
    x_crd = del_x * np.linspace(int(-num_pts / 2), int(num_pts / 2), int(num_pts))
    xm, ym = np.meshgrid(x_crd, x_crd)
    rm = np.sqrt(xm ** 2 + ym ** 2)
    thm = np.arctan2(ym, xm)

    # get zernike indexes from CSV file
    if i_mx > 1000:
        print('ERROR: TOO MANY ZERNIKE POLYNOMIALS REQUESTED')
    zern_idx = zern_indexes()
    zern_idx = zern_idx[0:i_mx, :]

    # create array of 2d zernike polynomials with zernike radial function
    zern = np.zeros((int(i_mx), int(num_pts), int(num_pts)))
    for ii in np.arange(0, i_mx):
        nn = zern_idx[ii, 0]
        mm = zern_idx[ii, 1]
        if mm == 0:
            zern[ii, :, :] = np.sqrt(nn) * zrf(nn, 0, rm)
        else:
            if np.mod(ii, 2) == 0:
                zern[ii, :, :] = np.sqrt(2 * (nn)) * zrf(nn, mm, rm) * np.cos(mm * thm)
            else:
                zern[ii, :, :] = np.sqrt(2 * (nn)) * zrf(nn, mm, rm) * np.sin(mm * thm)
        mask = (xm ** 2 + ym ** 2) <= 1
        zern[ii] = zern[ii] * mask

    return zern, zern_idx


# pull zernike indices from csv file, defaults to zern_idx.csv
def zern_indexes(filename: str = 'zern_idx.csv'):
    root = Path(__file__).parent

    # read in csv file and convert to list
    raw = csv.DictReader(open(Path(root, filename)))
    raw_list = list(raw)

    # initialize z array
    r = 1000  # number of indixes in zern.csv file
    c = 3  # x, y, i
    z = np.zeros((r, c))

    # get 'x' and 'y' column values as index 'i'
    for row in np.arange(0, r):
        row_vals = raw_list[row]
        z[row, 0] = float(row_vals['x'])
        z[row, 1] = float(row_vals['y'])
        z[row, 2] = float(row_vals['i'])

    return z


# zernike radial function
def zrf(n, m, r):
    rr = 0
    for ii in np.arange(0, (n - m + 1) / 2):
        num = (-1) ** int(ii) * math.factorial(int(n - ii))
        dnm = math.factorial(int(ii)) * math.factorial(int(((n + m) / 2) - ii)) * math.factorial(int(((n - m) / 2) - ii))
        rr = rr + (num / dnm) * r ** (n - (2 * ii))
    return rr


# make phase screens from zernike polynomials and atmosphere conditions
def make_screens(frames, nn, d, ro, l1, deltat, windx, windy, boil, zern, ch, zern_mx):
    # define variables
    x_crd = np.linspace(-l1 / 2, l1 / 2, nn)  # create coordinates

    # generate screens (True creates evolving phase screens)
    screen = zern_phase_scrn(ro, nn, zern_mx, x_crd, windx, windy, boil, deltat, frames, zern, ch, True)
    z = np.zeros((zern_mx-1, nn, nn))
    z[:, int(nn / 4):int(3 * nn / 4), int(nn / 4):int(3 * nn / 4)] = zern

    return screen, x_crd, z


# create phase screen from zernike polynomials
def zern_phase_scrn(ro, nn, zern_mx, x_crd, windx, windy, boil, deltat, frames, zern, ch, atm_flag):
    if atm_flag:
        # generate multiple phase screens over time using atmospheric vars
        screens = atmos_phase_screen(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern, ch, frames)
    else:
        # generate phase screen directly from zernike poly
        screens = general_phase_screen(nn, zern_mx, zern, ch)

    return screens


# generates phase screens randomly from zernike polynomials
def general_phase_screen(nn, zern_mx, zern, ch):
    # create random numbers to scale zernike polynomials
    rn = np.random.normal(size=(zern_mx - 1, 1))
    z_coef = np.matmul(ch, rn)

    # initialize zernike phase screen
    zern_phs = np.zeros((nn, nn))

    # summations of randomly scaled zernike polynomials
    for ii in np.arange(0, zern_mx - 1):
        zern_phs = zern_phs + z_coef[ii] * zern[ii, :, :]

    return zern_phs


# creates phase screens based on zernike polynomials and atmospheric variables
def atmos_phase_screen(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern, ch, frames):
    # increase size of the zernike polynomial spaces to allow for correlation calcs
    zern2 = np.zeros((zern_mx - 1, nn, nn))
    zern2[:, int(nn / 4):int(3 * nn / 4), int(nn / 4):int(3 * nn / 4)] = zern

    r_n, cond_var = get_evolve_stats(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern2, ch)

    # generate screens from statistics (update frame based on conditional mean and variance)
    n_vec = np.random.normal(size=(1, zern_mx - 1))
    z_record = np.zeros((zern_mx-1, frames))
    screens = np.zeros((frames, nn, nn))
    for ii in range(0, frames):
        atm_lens_phs = np.zeros((nn, nn))
        z_scale = ch @ n_vec.T
        z_record[:, ii] = np.squeeze(z_scale)
        for jj in np.arange(0, zern_mx - 1):
            atm_lens_phs = atm_lens_phs + z_scale[jj] * zern2[jj, :, :]
        screens[ii, :, :] = atm_lens_phs
        cond_mean = n_vec * r_n
        n_vec = np.sqrt(cond_var) * np.random.normal(size=(zern_mx-1)) + cond_mean

    return screens


# get statistics to evolve an atmospheric screen
def get_evolve_stats(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern2, ch):
    # get phase structure
    xm, ym = np.meshgrid(x_crd, x_crd)  # tau_x, tau_y
    phs_struct = 6.88 * (((ym + (windy + boil) * deltat) ** 2 + (xm + (windx + boil) * deltat) ** 2) ** (5 / 6) -
                      (xm ** 2 + ym ** 2) ** (5 / 6)) / ro ** (5 / 3)

    # denominator, Zernike sum of squares
    dnm = np.zeros((zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        dnm[xx] = np.sum(np.sum(zern2[xx, :, :] ** 2))

    # FFT of all zernikes
    fft_mat = np.zeros((nn, nn, zern_mx - 1)) + 0j
    for jj in np.arange(0, zern_mx - 1):
        fft_mat[:, :, jj] = np.fft.fft2(np.fft.fftshift(zern2[jj, :, :]))

    # inner double sum integral
    idsi = np.zeros((zern_mx - 1, zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        for yy in np.arange(0, zern_mx - 1):
            xcorr_fft = np.real(np.fft.fftshift(np.fft.ifft2(fft_mat[:, :, xx] * fft_mat[:, :, yy].conj())))
            idsi[xx, yy] = np.sum(np.sum(xcorr_fft * phs_struct / (dnm[xx] * dnm[yy])))

    # get n structure function from the phase structure function differences
    phi = la.inv(ch)
    d_n = np.zeros(zern_mx - 1)
    phi_out = np.zeros((zern_mx - 1, zern_mx - 1, zern_mx - 1))
    for ii in range(0, zern_mx - 1):
        phi_out[:, :, ii] = np.outer(phi[ii, :], phi[ii, :])
        d_n[ii] = np.sum(np.sum(idsi * phi_out[:, :, ii]))

    # get the n-vector, and correlation functions
    r_0 = 1
    r_n = r_0 - d_n / 2
    r_n = np.clip(r_n, a_min=0, a_max=1).reshape((1, zern_mx - 1))
    cond_var = 1 - r_n ** 2
    cond_var = cond_var.reshape((1, zern_mx - 1))

    return r_n, cond_var


if __name__ == '__main__':
    main()
