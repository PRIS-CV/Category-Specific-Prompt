def splitstr(aa, labels):
    '''
    input:
    aa: 从CSV文件中读取的一条原始数据
    labels: 从label.csv文件中读取的数据
    output:
    c: 按照list格式整理的可以load的数据
    '''
    # print(aa)
    a = aa.strip("[(").strip(")]").split("), (")
    c = []
    for b in a :
        d = b.strip("'").strip("'").split("', '")
        for label in labels:
            if d[1] == label[1]:
                d[1] = int(label[0])
        c.append(d)
    return c