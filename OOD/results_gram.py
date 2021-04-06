import numpy as np
import utils.calculate_log as callog


def detect_mean(all_test_std, all_ood_std, gaps=None): 
    
    avg_results = dict()
    indices = list(range(len(all_test_std)))
    split = int(np.floor(0.1 * len(all_test_std))) 
    for i in range(1,11):
        np.random.seed(i)
        np.random.shuffle(indices)
        
        val_std = all_test_std[indices[:split]]
        test_std = all_test_std[indices[split:]]
        
        if gaps is not None:
            t95 = (val_std.sum(axis=0) + gaps.mean(0))
        else:
            t95 = val_std.mean(axis=0) + 10**-7
        # print (test_std)
        # print(t95)  
        # test_std = ((test_std)/t95[np.newaxis,:]).sum(axis=1)
        # ood_std = ((all_ood_std)/t95[np.newaxis,:]).sum(axis=1)

        results = callog.compute_metric(-test_std,-all_ood_std)  

        for m in results:
            avg_results[m] = avg_results.get(m,0)+results[m]
    
    for m in avg_results:
        avg_results[m] /= i
        
        
    callog.print_results(avg_results)
