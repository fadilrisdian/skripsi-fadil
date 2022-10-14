import numpy as np

def print_metric(acc_results, f1_results, prec_results, rec_results):
    print("Accuracy array:", acc_results)
    print("F1 array:", f1_results)
    print("Precision array:", prec_results)
    print("Recall array:", rec_results)
  
    print('PERFORMANCE')
    print('Accuracy    : {0:.5f}±{1:.3f}'.format(np.mean(acc_results), np.std(acc_results)))
    print('F1 Score    : {0:.5f}±{1:.3f}'.format(np.mean(f1_results), np.std(f1_results)))
    print('Precision   : {0:.5f}±{1:.3f}'.format(np.mean(prec_results), np.std(prec_results)))
    print('Recall      : {0:.5f}±{1:.3f}'.format(np.mean(rec_results), np.std(rec_results)))


