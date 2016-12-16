import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


def gini2(p):
    return 1 - ((p ** 2) + (1 - p) ** 2)


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
gi = gini(x)
gi2 = gini2(x)
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(e) for e in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c in zip([ent, sc_ent, gi, err],
                         ['Entropy', 'Entropy Scaled', 'Gini Impurity',
                             'Misclassification Error'], ['-', '-', '--', '-.'],
                         ['black', 'lightgray',
                          'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, color=c, lw=2, linestyle=ls)

ax.legend(loc='upper center', ncol=4)
ax.axhline(y=.5, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel(r'$p(i=1)$')
plt.ylabel('Impurity Index')
plt.show()
