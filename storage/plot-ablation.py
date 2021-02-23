import pickle
from scipy.stats import sem ,t
from scipy import mean
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import json


color = {'coor_t': [159, 67, 255],  # Purple
         'RODE': [216, 30, 54],  # Red
         'full_action_space': [204, 0, 204],  # Magenta
         'QMIX': [254, 151, 0],  # Orange
         'ROMA': [0, 0, 0],  # Black background
         'h_resac': [180, 180, 180],  # Grey board walls
         '': [0, 255, 0],  # Green apples
         '7': [255, 255, 0],  # Yellow fining beam
         '9': [159, 67, 255],  # Purple player

         # Colours for agents. R value is a unique identifier
         'RODE_full_ac': [2, 81, 154],  # Blue
         'q-q': [100, 255, 255],  # Cyan
         'cen': [83, 134, 139],  # Lavender
         'plus_v': [250, 204, 255],  # Pink
         'cen_control': [238, 223, 16],
         'social_influence': [238, 232, 170]}

color = {key: np.array(value, np.float) / 255. for key, value in color.items()}

methods = [
    'RODE',
    'RODE_full_ac'
    ]

labels = [
    'RODE',
    'RODE with Full Role Action Spaces']

length = {
    'MMM2': 204,
    '3s5z_vs_3s6z': 499,
    'corridor': 496,
    '27m_vs_30m': 499,
    '6h_vs_8z': 498
}

easy = ['2s_vs_1sc', '2s3z', '3s5z', '1c3s5z', '10m_vs_11m']
hard = ['2c_vs_64zg', 'bane_vs_bane', '5m_vs_6m', '3s_vs_5z']
vh = ['3s5z_vs_3s6z', '6h_vs_8z', '27m_vs_30m', 'MMM2', 'corridor']

envs = easy + hard + vh

# envs = ['2s_vs_1sc', '2s3z']
envs = ['corridor', '3s5z_vs_3s6z', '6h_vs_8z']
# envs = ['3s5z_vs_3s6z', '6h_vs_8z', '5m_vs_6m', '3s5z', '27m_vs_30m']

for env in easy + hard:
    length[env] = 199

# #################

alpha = 0.3
scale = 50.
confidence = 0.95
log_scale = False
font_size = 26
legend_font_size = 28 # 24
anchor = (0.5, 1.08)


def smooth(data):
    range1 = 10.0
    new_data = np.zeros_like(data)
    for i in range(int(range1), len(data)):
        new_data[i] = 1. * sum(data[i-int(range1): i]) / range1

    return new_data


# def resize(data):
#     if len(data) < max_length:
#         data += [0 for _ in range(max_length - len(data))]
#     elif len(data) > max_length:
#         data = data[:max_length]
#
#     return data


def read_data(env, type, cut, cut_length=None):
    data_n = []
    x_n = []

    if type in ['QMIX'] and env in easy + hard:
        path = "/data/smac_run_data.json"
        run_number = 3
        with open(path, 'r') as f:
            file_data = json.load(f)

            for i in range(3):
                if env == '10m_vs_11m' and i == 0:
                    continue
                if env == '2c_vs_64zg' and i == 1:
                    continue
                _data = np.array(file_data[env]['QMIX']['test_battle_won_mean']['Run_{}'.format(i + 1)])
                # if type == 'ROMA':
                #     _data[:, 1] *= 2 * np.random.rand(*_data[:, 1].shape)
                #     _data[:, 1] = np.clip(_data[:, 1], a_min=0, a_max=None)
                data_n.append(_data[:, 1])
                x_n.append(_data[:, 0])
    else:
        files = []

        epsilon_anneal_time = '500'
        if env in ['MMM2', 'corridor'] and type in ['QMIX', 'RODE', 'RODE_full_ac']:
            epsilon_anneal_time = '50'

        if type in ['QMIXR', 'QMIX']:
            path = "/data/storage/" + type + "-sc2/anneal-" + epsilon_anneal_time + "k/" + env + "/sacred"
        else:
            if env in easy + hard:
                path = type + '/' + env + '/sacred'
            else:
                path = type + '/' + env + '/action-20-' + epsilon_anneal_time + 'k/sacred'

        for r, d, f in os.walk(path):
            for file in f:
                if file == 'info.json':
                    files.append(os.path.join(r, file))

        run_number = len(files)

        for f in files:
            try:
                with open(f, 'r') as _f:
                    d = json.load(_f)

                    data_n.append(np.array(d['test_battle_won_mean']))
                    x_n.append(np.array(d['test_battle_won_mean_T']))
            except:
                print('Error:', f)
    try:
        if cut == 'min_cut':
            min_length = min([len(_x) for _x in x_n])
        elif cut == 'fix_cut':
            min_length = cut_length
        else:
            min_length = -1

        data_n = [data[:min_length] for data in data_n]
        x_n = [x[:min_length] for x in x_n]
    except:
        print('Error', env, type)

    return np.array(x_n), np.array(data_n), min_length, run_number


s_cut = 198

if __name__ == '__main__':
    # figure =
    # ######### 1
    figure = plt.figure(figsize=(32, 6))
    data = [[] for _ in methods]

    legend_elements = [Line2D([0], [0], lw=4, label=label, color=color[method]) for method, label in
                       zip(methods + ['QMIX'], labels + ['QMIX best'])]
    figure.legend(handles=legend_elements, loc='upper center', prop={'size': legend_font_size}, ncol=min(len(methods) + 1, 4) ,
                  bbox_to_anchor=(0.5, 1.12), frameon=False)

    for idx, env in enumerate(envs):
        ax= plt.subplot(1, 3, idx+1)
        ax.grid()
        # figure = plt.figure()
        # plt.grid()
        method_index = 0

        for method, label in zip(methods, labels):
            x, y, min_length, run_number = read_data(env, method, cut='fix_cut', cut_length=length[env])
            print(env, method, run_number)

            y *= 100
            y_median = smooth(np.median(y, axis=0))
            train_scores_mean = y_median
            data[method_index].append(y_median[:s_cut])
            method_index += 1

            low = smooth(np.percentile(y, 25, axis=0))
            high = smooth(np.percentile(y, 75, axis=0))

            # h = smooth(sem(y) * t.ppf((1 + confidence) / 2, min_length - 1))
            #     h = smooth(sem(data) * t.ppf((1 + confidence) / 2, max_length - 1))
            # bhos = np.linspace(1, min_length, min_length)
            bhos = x[0] / 1000000
            # if log_scale:
            #     train_scores_mean = np.log(train_scores_mean + scale) - np.log(scale)
            #     h = np.log(h + scale) - np.log(scale)
            ax.fill_between(bhos, low,
                             high, alpha=alpha,
                             color=color[method], linewidth=0)
            ax.plot(bhos, train_scores_mean, color=color[method], label=label, linewidth=4)

        # Others
        x, y, min_length, run_number = read_data(env, 'QMIX', cut='fix_cut', cut_length=length[env])
        y_median = smooth(np.median(y, axis=0))
        ax.plot(bhos, np.ones_like(bhos) * y_median.max() * 100, color=color['QMIX'], label='QMIX best', linewidth=4, linestyle='--')

        ax.tick_params('x', labelsize=font_size)
        ax.tick_params('y', labelsize=font_size)
        ax.set_xlabel('T (mil)', size=font_size)
        ax.set_ylabel('Test Win %', size=font_size)
        ax.set_title(env, size=font_size)
        ax.set_ylim(-10, 110)

    # figure.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=anchor,
    #               prop={'size': legend_font_size}, ncol=min(len(methods), 4), frameon=False)

    figure.tight_layout()
    # plt.show()

    # plt.gca().set_facecolor([248./255, 248./255, 255./255])
    figure.savefig('./RODE-ablation.pdf', bbox_inches='tight', dpi=300)  # , bbox_extra_artists=(lgd,)
    plt.close(figure)

    data = np.array(data)
    plt.figure()
    data_mean = np.mean(data, axis=1)
    bhos = bhos[:s_cut]
    for method_id, method in enumerate(methods):
        plt.plot(bhos, data_mean[method_id], color=color[method], label=labels[method_id])
    legend_elements = [Line2D([0], [0], lw=4, label=label, color=color[method]) for method, label in
                       zip(methods, labels)]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1.6, 0.5), prop={'size': font_size})
    plt.savefig('./Average.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    data_max = np.zeros([len(methods), len(bhos)])
    for env_id in range(len(envs)):
        e_data = data[:, env_id, :]
        order = np.argsort(e_data, axis=0)
        largest = np.where(order == len(methods)-1, e_data, np.zeros([len(methods), len(bhos)])).sum(axis=0)
        s_largest = np.where(order == len(methods)-2, e_data, np.zeros([len(methods), len(bhos)])).sum(axis=0)
        s = ((largest - s_largest) > 1. / 32)
        addition = np.where(order == len(methods)-1, np.ones([len(methods), len(bhos)]), np.zeros([len(methods), len(bhos)]))
        data_max += addition * s

        if (addition * s)[0, -1] == 1.:
            print(envs[env_id], data[0, env_id, s_cut-1], data[1, env_id, s_cut-1])

    for method_id, method in enumerate(methods):
        plt.plot(bhos, smooth(data_max[method_id]), color=color[method], label=labels[method_id])
    legend_elements = [Line2D([0], [0], lw=4, label=label, color=color[method]) for method, label in
                       zip(methods, labels)]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(1.6, 0.5), prop={'size': font_size})
    plt.savefig('./Max.png', bbox_inches='tight')
    plt.close()
