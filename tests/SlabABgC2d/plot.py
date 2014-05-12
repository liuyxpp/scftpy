import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import mpltex.acs
from scftpy import SCFTConfig
from scftpy import contourf_slab2d, list_datafile


def plot():
    skipfiles = ['test1',
                 'test5',
                 'test6-2',
                 'test14',
                 'test17',
                 'test18',
                 ]
    datafiles = list_datafile()
    for f in skipfiles:
        for df in datafiles:
            path = os.path.dirname(df)
            d = os.path.basename(path)
            if d == f:
                datafiles.remove(df)
    Fs = []
    labels = []

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)  # F vs t
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)  # F vs time
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)  # err_res vs t
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)  # err_res vs time
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)  # err_phi vs t
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)  # err_phi vs time
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)  # err_F vs t
    fig8 = plt.figure()
    ax8 = fig8.add_subplot(111)  # err_F vs time

    for dfile in datafiles:
        mat = loadmat(dfile)
        path = os.path.dirname(dfile)
        label = os.path.basename(path)
        t, time, F = mat['t'].T, mat['time'].T, mat['F'].T
        err_res, err_phi = mat['err_residual'].T, mat['err_phi'].T
        err_F = F[1:] - F[:-1]
        Fs.append(F[-1])
        labels.append(label)
        ax1.plot(t, F, label=label)
        ax2.plot(time, F, label=label)
        ax3.plot(t, err_res, label=label)
        ax4.plot(time, err_res, label=label)
        ax5.plot(t, err_res, label=label)
        ax6.plot(time, err_phi, label=label)
        ax7.plot(t[1:], np.abs(err_F), label=label)
        ax8.plot(time[1:], np.abs(err_F), label=label)

    ax3.set_yscale('log')
    ax4.set_yscale('log')
    ax5.set_yscale('log')
    ax6.set_yscale('log')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')
    ax4.legend(loc='best')
    ax5.legend(loc='best')
    ax6.legend(loc='best')
    ax7.legend(loc='best')
    ax8.legend(loc='best')
    fig1.savefig('F-t.eps', format='eps', bbox_inches='tight')
    fig2.savefig('F-time.eps', format='eps', bbox_inches='tight')
    fig3.savefig('err_res-t.eps', format='eps', bbox_inches='tight')
    fig4.savefig('err_res-time.eps', format='eps', bbox_inches='tight')
    fig5.savefig('err_phi-t.eps', format='eps', bbox_inches='tight')
    fig6.savefig('err_phi-time.eps', format='eps', bbox_inches='tight')
    fig7.savefig('err_F-t.eps', format='eps', bbox_inches='tight')
    fig8.savefig('err_F-time.eps', format='eps', bbox_inches='tight')
    plt.close('all')

    x = np.arange(len(datafiles))
    Fs = np.array(Fs)
    fig9 = plt.figure(figsize=[10, 3])
    ax9 = fig9.add_subplot(111)  # F vs test
    ax9.plot(x, Fs, 'o')
    plt.xscale('log')
    plt.xticks(x, labels)
    fig9.savefig('F-all.eps', format='eps', bbox_inches='tight')


def render():
    datafiles = list_datafile()
    for dfile in datafiles:
        mat = loadmat(dfile)
        path = os.path.dirname(dfile)
        phiA, phiB, phiC = mat['phiA'], mat['phiB'], mat['phiC']
        pfile = os.path.join(path, 'param.ini')
        config = SCFTConfig.from_file(pfile)
        Lx = config.grid.Lx
        La, Lb = config.uc.a, config.uc.b

        xp, yp, phiAp = contourf_slab2d(phiA, La, Lb)
        figfile = os.path.join(path, 'phiA.eps')
        plt.savefig(figfile, format='eps', bbox_inches='tight')
        plt.close()
        xp, yp, phiBp = contourf_slab2d(phiB, La, Lb)
        figfile = os.path.join(path, 'phiB.eps')
        plt.savefig(figfile, format='eps', bbox_inches='tight')
        plt.close()
        xp, yp, phiCp = contourf_slab2d(phiC, La, Lb)
        figfile = os.path.join(path, 'phiC.eps')
        plt.savefig(figfile, format='eps', bbox_inches='tight')
        plt.close()

        plt.plot(yp, phiAp[Lx/2], label='$\phi_A$')
        plt.plot(yp, phiBp[Lx/2], label='$\phi_B$')
        plt.plot(yp, phiCp[Lx/2], label='$\phi_C$')
        plt.legend(loc='best')
        plt.xlabel('$z$')
        plt.ylabel('$\phi(z)$')
        figfile = os.path.join(path, 'profile.eps')
        plt.savefig(figfile, format='eps', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    #render()
    plot()
