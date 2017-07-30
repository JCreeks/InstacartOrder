def mean_F_score(x,y): # 'x' and 'y' are lists of lists like [[1,2,3],[8,9,7,4,1]]
    n=len(x)
    fscore=[]
    for i in range(n):
        tp=len(set(x[i]).intersection(y[i]))
        fn=len(set(y[i]).difference(x[i]))
        fp=len(set(x[i]).difference(y[i]))
        fscore.append(2*tp/(2*tp+fn+fp))
    return sum(fscore)/n
