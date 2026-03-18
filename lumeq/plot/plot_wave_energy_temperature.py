from lumeq import os, sys, np
from lumeq.utils import convert_units
from lumeq.plot import plt, mcolors, ticker

def plot_wave_energy_frequency():
    # 1nm to 1mm (1e6nm)
    wave_names = {#'X-ray':    [0.01,  10],  # nm X-ray # 100 keV--100 eV
                  'Extrem-UV': [1,     100], # nm extreme UV       1
                  'Vacuum-UV': [100,   190], # nm vacuum UV
                  'Deep-UV':   [190,   280], # nm deep UV          3
                  'Mid-UV':    [280,   315], # nm mid UV
                  'Near-UV':   [315,   380], # nm near UV          5 #
                  'violet':    [380,   435], # nm violet           6
                  'blue':      [435,   500], # nm blue
                  'cyan':      [500,   520], # nm cyan
                  'green':     [520,   565], # nm green
                  'yellow':    [565,   590], # nm yellow
                  'orange':    [590,   625], # nm orange
                  'red':       [625,   780], # nm red             12 #
                  'Near-IR I': [780,  1400], # nm near infrared I 13
                  'Near-IR II':[1400, 3000], # nm near infrared II   #
                  'Mid-IR':    [3,    50  ], # um mid infrared    15
                  'Far-IR':    [50,   1000], # um far infrared    16
                  }
    wavelengths = []
    for key, [v0, v1] in wave_names.items():
        wavelengths.append(np.linspace(v0, v1, (v1-v0)//2))
    #wavelengths = np.hstack(wavelengths)
    wavelengths = [np.hstack(wavelengths[0]), np.hstack(wavelengths[1]),
                   np.hstack(wavelengths[2:12]), np.hstack(wavelengths[12:14]),
                   np.hstack(wavelengths[14]), np.hstack(wavelengths[15]),
                   ]
    #print(wavelengths.shape)

    n = len(wavelengths)
    energy, frequency, temperature, time = [0]*n, [0]*n, [0]*n, [0]*n

    units = {'l': ['nm']*n,
             'e': ['eV']*n,
             'f': ['cm-1']*n,
             'T': ['K']*n,
             't': ['fs']*n,
             }
    units['l'][-2:] = ['um']*2
    units['e'][-3]  = 'kcal'
    units['e'][-2:] = ['meV']*2
    units['t'][0]   = 'as'
    units['t'][1]   = 'au'


    for i, length in enumerate(wavelengths):
        e = convert_units(length, units['l'][i], units['e'][i])
        energy[i] = np.copy(e)
        frequency[i] = convert_units(e, units['e'][i], units['f'][i])
        temperature[i] = convert_units(e, units['e'][i], units['T'][i])
        time[i] = convert_units(e, units['e'][i], units['t'][i])



    fig_name = 'spectrum_reference'
    fig = plt.figure(figsize=(12, 6), dpi=300, layout='constrained')
    #fig = plt.figure()
    nrow, ncol = 2, 3
    colors = list(wave_names.keys())

    for i, length in enumerate(wavelengths):
        ax1 = plt.subplot(nrow, ncol, i+1)
        ax2 = ax1.twinx()
        ax3 = ax1.twiny()

        ax1.plot(length, energy[i], color='b')
        # has save curve as energy except 2, just for yticks
        ax2.plot(length, temperature[i], color='g', alpha=0)


        if i < 3:
            ax3.plot(time[i], energy[i], color='b', alpha=0) # have save curve as energy, get yticks
            ax3.set_xlabel('Time ('+units['t'][i]+')')
        else:
            ax3.plot(frequency[i], energy[i], color='g')
            ax3.invert_xaxis()
            unit = 'cm$^{-1}$' if '-1' in units['f'][i] else units['f'][i]
            ax3.set_xlabel('Frequency ('+unit+')')
            if i == 4:
                ax1.set_xticks([3, 20, 40])
            elif i == 5:
                ax1.set_xticks([50, 250, 500, 750, 1000])

        unit = '$\\mu$m' if 'um' in units['l'][i] else units['l'][i]
        ax1.set_xlabel('Wavelength ('+unit+')')
        ax1.set_ylabel('Energy ('+units['e'][i]+')')
        ax2.set_ylabel('Temperature ('+units['T'][i]+')')
        if i==2: # y2 has mismatched ticks
            yticks = convert_units(ax1.get_ylim()[-1], units['e'][i], units['T'][i])
            ax2.set_ylim([0,yticks])
            ax2.set_yticks([20000, 40000, 60000])

        alpha = .15 if i==2 else .4
        ax1.grid(ls='--', alpha=alpha)

        if i == 2:
            for key, (v0, v1) in list(wave_names.items())[5:12]:
                width = v1 - v0
                #ax1.bar(v0, hight, width, color='none', edgecolor=key, align='edge', alpha=.85, hatch='|', zorder=0)
                ax1.axvspan(v0, v1, color=key, alpha=.8)
        elif i == 3:
            keys = list(wave_names.keys())[12:14]
            (xloc0, xloc1), (_, xloc2) = list(wave_names.values())[12:14]
            print(xloc0, xloc1, xloc2 )
            ax1.vlines(xloc1, energy[i][-1], energy[i][0], ls='--')
            ax1.text((xloc0+xloc1)/2, (energy[i][0]+energy[i][-1])/2, keys[0], ha='center')
            ax1.text((xloc1+xloc2)/2, (energy[i][0]+energy[i][-1])/2, keys[1], ha='center')
        elif i < 2:
            ax1.text(np.average(length), (energy[i][0]+energy[i][-1])/2, list(wave_names.keys())[i], ha='center')
        else:
            ax1.text(np.average(length), (energy[i][0]+energy[i][-1])/2, list(wave_names.keys())[i+10], ha='center')

    plt.tight_layout()

#    plt.show()
    plt.savefig(fig_name+'.png')


if __name__ == '__main__':
    plot_wave_energy_frequency()

