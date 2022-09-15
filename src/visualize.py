import matplotlib.pyplot as plt
import numpy as np

def visual(sad_dnn, dnn, title, path):
    # Intitialize figure with two plots
    fig, ax1, = plt.subplots(1)
    # fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    fig.set_figheight(7)
    fig.set_figwidth(7)
    fig.set_facecolor('white')

    ## set bar size
    barWidth = 0.3
    sae_dnn_score = sad_dnn
    dnn_score = dnn

    ## Set position of bar on X axis
    r1 = np.arange(len(sae_dnn_score))
    r2 = [x + barWidth for x in r1]

    ## Make the plotRandom Forest
    ax1.bar(r1, sae_dnn_score, width=barWidth, edgecolor='white', label='SAE-DNN', align='edge')
    ax1.bar(r2, dnn_score, width=barWidth, edgecolor='white', label='DNN', align='edge')

    
    ## Configure x and y axis
#     ax1.set_xlabel('Metrics', fontweight='bold')
    labels = ['Accuracy', 'F1', 'Precision', 'Recall']
    ax1.set_xticks([r + (barWidth * 1) for r in range(len(sae_dnn_score))], )
    ax1.set_xticklabels(labels)
#     ax1.set_ylabel('Score', fontweight='bold')
    
    # ganti ini buat limit normal atau std
    ax1.set_ylim(0, 1) 
    
    ## Create legend & title
    ax1.set_title(title, fontsize=22, fontweight=None)
    ax1.legend()

    #save the figure
    plt.savefig(path)

    plt.show()

    
    

