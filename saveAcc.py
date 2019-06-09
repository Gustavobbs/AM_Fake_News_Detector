import csv

def saveAcuracia(lambda_reg, acc, strAlg):

    a = [lambda_reg, acc]
    with open('accuracy'+strAlg+'.csv','a') as fd:
        writer = csv.writer(fd)
        writer.writerow(a)
