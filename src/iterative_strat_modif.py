from skmultilearn.model_selection import IterativeStratification

#Modifikasi IterativeStratification agar hasil random data tetap sama
def new_init(self, n_splits=3, order=1, sample_distribution_per_fold = None, random_state=None):

                  self.order = order
                  if random_state is not None:
                      do_shuffle = True
                  else:
                      do_shuffle = False
                  super(
                      IterativeStratification,
                      self).__init__(n_splits,
                                     shuffle=do_shuffle,
                                     random_state=random_state)
                  if sample_distribution_per_fold:
                      self.percentage_per_fold = sample_distribution_per_fold
                  else:
                      self.percentage_per_fold = [1 / float(self.n_splits) for _ in range(self.n_splits)]