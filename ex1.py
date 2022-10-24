import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson
import matplotlib.animation as animation
import matplotlib.pyplot as plt
plt.style.use('ggplot')

LAMBDA = 1
DELTA = 0.2
FAST = "fast"
INTER = "intermediate"
SLOW = "slow"
ON = "always_on"


def set_params(transition_type):
    if transition_type == FAST:
        return 1, 1
    elif transition_type == INTER:
        return 0.1, 0.1
    elif transition_type == SLOW:
        return 0.01, 0.01
    elif transition_type == ON:
        return 0, 0

class State:
    def __init__(self, on, k, k_on, k_off):
        self.on = on
        self.k = k
        self.beta = k * DELTA  # rate of decreasing k
        if on:
            self.alpha = LAMBDA  # rate of increasing k
            self.gamma = k_off  # rate of switching promoter state, on/off
        else:
            self.alpha = 0
            self.gamma = k_on

    def __str__(self):
        return '({}, {})'.format(self.on, self.k)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        if self.k != other.k:
            return self.k < other.k
        return self.on < other.on


def plot_simulation(states_path, transition_type):
    states_array = np.array([[s[0], s[1].k] for s in states_path])
    temp = states_array.transpose()
    plt.plot(temp[0], temp[1], linewidth='0.3')
    plt.ylim(ymin=-1, ymax=20)
    plt.title('simulation of a run for ' + transition_type + ' transitions')
    plt.xlabel('time')
    plt.ylabel('k')
    plt.legend()
    # plt.show()
    plt.savefig('C:/Users/Ela/Documents/SemesterB/genomics/exs/ex1/run_for_' + transition_type)

    plt.clf()
    # b = states_array
    # u, idx = np.unique(b, axis=0, return_inverse=True)
    # temp = [np.where(idx == i) for i in range(len(u))]
    # for stop in range(len(temp)):
    #     label = np.array2string(temp[stop][0])
    #     # label = "{}".format(temp[stop])
    #
    #     plt.annotate(label,  # this is the text
    #                  u[stop],  # this is the point to label
    #                  textcoords="offset points",  # how to position the text
    #                  xytext=(0, 10),  # distance from text to points (x,y)
    #                  ha='center')
    # plt.xlim(xmin=-1, xmax=2)
    # plt.show()


def simulate(s0, t_end, k_on, k_off):
    cur_t = 0
    s = s0
    res = [[cur_t, s]]
    while cur_t < t_end:
        r = s.alpha + s.beta + s.gamma
        t_pass = np.random.exponential(1/r)
        cur_t += t_pass
        m = np.argwhere(np.random.multinomial(1, [s.alpha/r, s.beta/r, s.gamma/r]) == 1)[0][0]
        k = s.k
        on = s.on
        if m == 0:  # increasing k
            k += 1
        elif m == 1:  # decreasing k
            k -= 1
        else:  # m == 2, switching promoter state, on/off
            on = not on
        s = State(on, k, k_on, k_off)
        # print("s =", s, "cur_t =", cur_t)
        res.append([np.round(cur_t, 4), s])
    return np.array(res)


def collect_samples(t_end, k, k_on, k_off):
    samples = np.empty(k, dtype=type(State(True, 0, 0, 0)))
    for i in range(k):
        a = np.random.binomial(1, 0.5, 1)
        s0 = State(a == 1, 0, k_on, k_off)
        res = np.array(simulate(s0, t_end, k_on, k_off))
        samples[i] = res[-1][1]
    return samples


def collect_samples_multi_run(t_start, t_delta, k, n, k_on, k_off):
    t_end = t_start + (n-1) * t_delta
    samples = np.empty((k, n), dtype=type(State(True, 0, 0, 0)))
    for i in range(k):
        a = np.random.binomial(1, 0.5, 1)
        s0 = State(a == 1, 0, k_on, k_off)
        res = np.array(simulate(s0, t_end, k_on, k_off))
        for j in range(n):
            next_sample = res[res.transpose()[0] <= t_start + j * t_delta][-1]  # get the last t_i that it apply to
            samples[i, j] = next_sample[-1]
    return samples


def plot_hist(t_start, t_delta_end, k, n, transition_type, multirun=True):
    k_on, k_off = set_params(transition_type)
    if multirun:
        res = collect_samples_multi_run(t_start, t_delta_end, k, n, k_on, k_off)
        states_array = np.array([[int(s.on), s.k] for i in res for s in i])
    else:
        res = collect_samples(t_delta_end, k, k_on, k_off)
        states_array = np.array([[s.on, s.k] for s in res])

    prob_on = 0.5
    n_off = len(np.where(states_array.transpose()[0] == False)[0])
    n_on = len(states_array) - n_off
    u, counts = np.unique(states_array, axis=0, return_counts=True)
    x = np.array(u)
    y = np.array(counts)
    norm = np.sum(y)
    temp_x = x.transpose()[0]
    promoters = list(map(lambda x: "ON" if x else "OFF", temp_x))
    normed_y = np.array([y[i] / n_on if temp_x[i] else y[i] / n_off for i in range(len(temp_x))])
    cur_dict = {"k": x.transpose()[1],
                "count": normed_y,
                "promoter": promoters}

    cur_df = pd.DataFrame(cur_dict)
    ax = sns.barplot(x='k', y='count', hue='promoter', data=cur_df)
    plt.legend()
    if multirun:
        ax.set(title='Histogram by promoters of states from collect_samples_multi_run\n' + transition_type + ' transitions, k='
                     + str(k) + ', n=' + str(n) + ', T_delta=' + str(t_delta) + ', T_start=' + str(t_start))
        plt.savefig('C:/Users/Ela/Documents/SemesterB/genomics/exs/ex1/plots2/2promoters_multirun_' + transition_type +
                    '_k=' + str(k) + '_n=' + str(n) + '_T_delta=' + str(t_delta) + '_T_start=' + str(t_start))
    else:
        ax.set(
            title='Histogram by promoters of states from collect_samples\n' + transition_type + ' transitions, k='
                  + str(k) + ', T_end=' + str(t_delta))
        plt.savefig('C:/Users/Ela/Documents/SemesterB/genomics/exs/ex1/plots2/2promoters_' + transition_type +
                    '_k=' + str(k) + '_T_end=' + str(t_delta))
    plt.show()
    plt.clf()

    if transition_type == FAST:
        t = np.arange(0, 15, 1)
        plt.plot(t,  poisson.pmf(t, prob_on * LAMBDA / DELTA), 'bo', ms=8, label='poisson pmf')
        plt.vlines(t, 0,  poisson.pmf(t, prob_on * LAMBDA / DELTA), colors='b', lw=5, alpha=0.5)
    elif transition_type == SLOW:
        t = np.arange(1, 15, 1)
        plt.plot([0], (1 - prob_on) + prob_on * np.exp(-LAMBDA / DELTA), 'ro', ms=8, label='ZIP pmf k=0')
        plt.plot(t, prob_on * poisson.pmf(t, LAMBDA/DELTA), 'bo', ms=8, label='ZIP pmf k>0')
        plt.vlines([0], 0, (1 - prob_on) + prob_on * np.exp(-LAMBDA / DELTA), colors='r', lw=5, alpha=0.5)
        plt.vlines(t, 0, prob_on * poisson.pmf(t, LAMBDA/DELTA), colors='b', lw=5, alpha=0.5)
    else:
        t = np.arange(1, 15, 1)
        plt.plot([0], 0.5 * poisson.pmf([0], prob_on * LAMBDA / DELTA) + 0.5 * (1 - prob_on) + prob_on * np.exp(-LAMBDA / DELTA), 'ro', ms=8, label='average pmf k=0')
        plt.plot(t, 0.5 * poisson.pmf(t, prob_on * LAMBDA / DELTA) + 0.5 * prob_on * poisson.pmf(t, LAMBDA/DELTA), 'bo', ms=8, label='average pmf k>0')
        plt.vlines([0], 0, 0.5 * poisson.pmf([0], prob_on * LAMBDA / DELTA) + 0.5 * (1 - prob_on) + prob_on * np.exp(-LAMBDA / DELTA), colors='r', lw=5, alpha=0.5)
        plt.vlines(t, 0, 0.5 * poisson.pmf(t, prob_on * LAMBDA / DELTA) + 0.5 * prob_on * poisson.pmf(t, LAMBDA/DELTA), colors='b', lw=5, alpha=0.5)

    k_array = np.array(states_array).transpose()[1]
    u, counts = np.unique(k_array, return_counts=True)
    x = u
    y = np.array(counts)
    norm = np.sum(y)
    temp_y = np.zeros(max(max(x), 12) + 1)
    temp_y[list(x)] = y / norm

    ax = sns.barplot(np.arange(0, max(max(x), 12) + 1), temp_y, alpha=0.9, color='goldenrod', label='state samples')
    plt.legend()
    # if multirun:
    #     ax.set(title='Histogram of states from collect_samples_multi_run\n' + transition_type + ' transitions, k='
    #                  + str(k) + ', n=' + str(n) + ', T_delta=' + str(t_delta) + ', T_start=' + str(t_start))
    #     plt.savefig('C:/Users/Ela/Documents/SemesterB/genomics/exs/ex1/plots2/2multirun_' + transition_type +
    #                 '_k=' + str(k) + '_n=' + str(n) + '_T_delta=' + str(t_delta) + '_T_start=' + str(t_start))
    # else:
    #     ax.set(
    #         title='Histogram of states from collect_samples\n' + transition_type + ' transitions, k='
    #               + str(k) + ', T_end=' + str(t_delta))
    #     plt.savefig('C:/Users/Ela/Documents/SemesterB/genomics/exs/ex1/plots2/2' + transition_type +
    #                 '_k=' + str(k) + '_T_end=' + str(t_delta))
    # plt.show()
    plt.clf()


def animate_simulation(transition_type, t_end):
    k_on, k_off = set_params(transition_type)
    s0 = State(False, 0, k_on, k_off)
    states_path = simulate(s0, t_end, k_on, k_off)
    states_array = np.array([[0, 0], [1, 0]] + [[s.on, s.k] for s in states_path.transpose()[1]])
    temp = states_array.transpose()
    print("start animation")

    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 2), ylim=(-1, 20))
    promoters = list(map(lambda x: "ON" if x else "OFF", temp[0]))
    cur_dict = {"k": temp[1],
                "promoter": promoters}

    cur_df = pd.DataFrame(cur_dict)

    ax.set_ylabel('k')
    plt.title('Animation of simulation for 200 steps,\n' + transition_type + ' transitions, k_on='
              + str(k_on) + ', k_off=' + str(k_off))

    line1, = ax.plot('promoter', 'k', data=cur_df[:0], marker='o', color='lightcoral', label='')
    line2, = ax.plot('promoter', 'k', data=cur_df[:0], marker='o', color='maroon', label='current transition')
    ax.legend()
    ax.plot('promoter', 'k', data=cur_df[:1], marker='o', color='black')
    ax.plot('promoter', 'k', data=cur_df[1:2], marker='o', color='black')

    def init():
        return [line1, line2]

    def animate(i):
        line1.set_data(cur_df[2:i + 3]['promoter'], cur_df[2:i + 3]['k'])
        line2.set_data(cur_df[2 + i:i + 4]['promoter'], cur_df[2 + i:i + 4]['k'])
        return [line1, line2]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=200, interval=80, blit=True)
    anim.save(transition_type + '.gif', writer='imagemagick')
    plt.clf()


if __name__ == '__main__':
    # transition_type = SLOW
    # k_on, k_off = set_params(transition_type)

    # animations
    # for transition_type in [FAST, INTER, SLOW]:
    #     animate_simulation(transition_type, 200)

    # always on
    # x = np.arange(0, 15, 1)
    # plt.plot(x, poisson.pmf(x, LAMBDA/DELTA), 'bo', ms=8, label='poisson pmf')
    # plt.vlines(x, 0, poisson.pmf(x, LAMBDA/DELTA), colors='b', lw=5, alpha=0.5)
    # k=20
    # n=50
    # t_delta = 100
    # plot_hist(s0, 200, t_delta, k, n, transitions_type)

    # test mixture model
    k=30
    n=30
    t_delta = 100
    t_start = 200
    t_end = 10000
    for transition_type in [FAST, INTER, SLOW]:
    # for transition_type in [SLOW]:
    #     plot_hist(t_start, t_delta, k, n, transition_type)
        k_on, k_off = set_params(transition_type)
        plot_simulation(simulate(State(True, 0, k_on, k_off), t_end, k_on, k_off), transition_type)


    # t_end = t_start + (n-1) * t_delta
    # for transition_type in [FAST, INTER, SLOW]:
    # # # for transition_type in [INTER]:
    #     plot_hist(t_start, t_end, k*n, n, transition_type, False)

    # for n in range(10, 100, 20):
    #     for t_delta in range(100, 1000, 100):
    #         for k in range(10, 100, 20):
    #
    #
    #             plot_hist(s0, 0, t_delta, k, n, transitions_type)
    #             print("*")
    #         print("**")
    #     print("***")
    # print(states_array)

