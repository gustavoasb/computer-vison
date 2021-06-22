import numpy as np
import matplotlib.pyplot as plt


def get_AP(exp_id, cls, tset, draw=True):
    '''
    exp_id: nome do experimento a ser avaliado
    cls: nome da classe a ser avaliada
    draw: se verdade, desenha curva de precision-recall
    set: "val" ou "test" set
    '''
    #  load test set
    gt_ids = []
    gt = []
    with open("../VOCdevkit/VOC2007/ImageSets/Main/{}_{}.txt".format(cls, tset), "r") as handle:
        for line in handle:
            img, label = line.split()
            gt_ids.append(img)
            gt.append(int(label))

    # load results
    ids = []
    confidence = []
    with open("../VOCdevkit/results/VOC2007/Main/{}_cls_{}_{}.txt".format(exp_id, tset, cls), "r") as handle:
        for line in handle:
            img, conf = line.split()
            ids.append(img)
            confidence.append(float(conf))

    # map results to ground truth images
    out = np.ones(len(gt))*(-np.inf)
    for i in range(0, len(ids)):
        # find ground truth image
        j=gt_ids.index(ids[i])
        out[j]=confidence[i]

    # compute precision/recall
    si = np.argsort(-out)
    gt = np.array(gt)
    tp=gt[si]>0
    fp=gt[si]<0

    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    rec=tp/np.sum(gt>0)
    prec=tp/(fp+tp)

    # compute average precision
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = np.max(prec[rec >= t], initial=0.)
        ap = ap+p/11

    if draw:
        # plot precision/recall
        plt.plot(rec, prec)
        plt.title("class: {}, subset: Teste, AP= {}".format(cls, ap))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()

    return ap
