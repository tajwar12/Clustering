# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class LCA:
    def __init__(self, n_components=2, tol=1e-3, max_iter=100, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.tol = tol
        self.max_iter = max_iter

        # flag to indicate if converged
        self.converged_ = False

        # model parameters
        self.ll_ = [-np.inf]
        self.weight = None
        self.theta = None
        self.responsibility = None

        # bic estimation
        self.bic = None

        # verbose level
        self.verbose = 1

    def _calculate_responsibility(self, data):

        n_rows, n_cols = np.shape(data)
        r_numerator = np.zeros(shape=(n_rows, self.n_components))
        for k in range(self.n_components):
            r_numerator[:, k] = self.weight[k] * np.prod(stats.bernoulli.pmf(
                data, p=self.theta[k]), axis=1)
        r_denominator = np.sum(r_numerator, axis=1)
        print(r_denominator)
        return r_numerator / np.tile(r_denominator, (self.n_components, 1)).T

    def _do_e_step(self, data):

        self.responsibility = self._calculate_responsibility(data)

    def _do_m_step(self, data):

        n_rows, n_cols = np.shape(data)

        # pi
        for k in range(self.n_components):
            self.weight[k] = np.sum(self.responsibility[:, k]) / float(n_rows)

        # theta
        for k in range(self.n_components):
            numerator = np.zeros((n_rows, n_cols))
            for n in range(n_rows):
                numerator[n, :] = self.responsibility[n, k] * data[n, :]
            numerator = np.sum(numerator, axis=0)
            denominator = np.sum(self.responsibility[:, k])
            self.theta[k] = numerator / denominator

        # correct numerical issues
        mask = self.theta > 1.0
        self.theta[mask] = 1.0
        mask = self.theta < 0.0
        self.theta[mask] = 0.0

    def fit(self, data):

        # initialization step
        n_rows, n_cols = np.shape(data)
        if n_rows < self.n_components:
            raise ValueError(
                '''
                LCA estimation with {n_components} components, but got only
                {n_rows} samples
                '''.format(n_components=self.n_components, n_rows=n_rows))

        if self.verbose > 0:
            print('EM algorithm started')

        self.weight = stats.dirichlet.rvs(np.ones(shape=self.n_components) / 2)[0]
        self.theta = stats.dirichlet.rvs(alpha=np.ones(shape=n_cols) / 2,
                                         size=self.n_components)
        print('weights =', self.weight)
        print('weights =', self.theta)

        for i in range(self.max_iter):
            if self.verbose > 0:
                print('\tEM iteration {n_iter}'.format(n_iter=i))

            # E-step
            self._do_e_step(data)

            # M-step
            self._do_m_step(data)

            # Check for convergence
            aux = np.zeros(shape=(n_rows, self.n_components))
            for k in range(self.n_components):
                normal_prob = np.prod(stats.bernoulli.pmf(data, p=self.theta[k]), axis=1)
                aux[:, k] = self.weight[k] * normal_prob
            ll_val = np.sum(np.log(np.sum(aux, axis=1)))
            if np.abs(ll_val - self.ll_[-1]) < self.tol:
                break
            else:
                self.ll_.append(ll_val)

        # calculate bic
        self.bic = np.log(n_rows)*(sum(self.theta.shape)+len(self.weight)) - 2.0*self.ll_[-1]

    def predict(self, data):
        return np.argmax(self.predict_proba(data), axis=1)

    def predict_proba(self, data):
        return self._calculate_responsibility(data)


df = pd.read_csv('handwritten\semeion.data', header=None,  delimiter=r"\s+",)

print(df.head()) # print the header

print(df.shape)  # print the shape of the data


# a = df.iloc[1:2,:]
# fig, ax = plt.subplots(figsize=(5,5))
# plt.imshow(np.array(a.iloc[:,:256]).reshape((16,16)), cmap='gray')

X = pd.DataFrame(df)
data = X.to_numpy()
data2 = data.astype(int)


data2 = data2[:, 0:100]
from sklearn.decomposition import PCA

pca = PCA(n_components=100)
pca.fit(data2)
PCA(n_components=100)
print(pca.explained_variance_ratio_)


# label_df = pd.DataFrame(df.iloc[:, [256, 257, 258, 259, 260, 261, 262, 263, 264, 265]])
# label_df.rename(columns={256: 0, 257: 1, 258: 2, 259: 3, 260: 4, 261: 5, 262: 6, 263: 7, 264: 8, 265: 9},
#                 inplace=True)
# y = label_df


# print("yyyy = ", y.shape)


columns = ["C1", "C2", "C3", "C4"]
true_theta = [
    [0.1, 0.4, 0.9, 0.2],
    [0.5, 0.9, 0.1, 0.1],
    [0.9, 0.9, 0.5, 0.9]
]
true_weights = [0.1, 0.5, 0.4]
N = 10000
#
data = []
for tw, tt in zip(true_weights, true_theta):
    data.append(stats.bernoulli.rvs(p=tt, size=(int(tw * N), len(tt))).tolist())

data = np.concatenate(data)
# data = X.to_numpy()
lca = LCA(n_components=10, tol=10e-14, max_iter=1)
#
lca.fit(data2)
print(lca.weight)
#
#
#
#
#
#
#
#
#
#
#
