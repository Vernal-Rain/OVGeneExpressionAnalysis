from numpy import load, array
from sklearn.neighbors import KNeighborsClassifier


def Q3_1_2(file, k, flag=0):
    with load(file) as data:
        training = data['training_data']
        training_label = data['training_label']
        testing = data['testing_data']
        testing_label = data['testing_label']
    if flag == 0:
        temp = []
        for thing in training:
            temp.append(thing[:1000])
        training = temp
        temp = []
        for thing in testing:
            temp.append(thing[:1000])
        testing = temp
    neighbors = KNeighborsClassifier(n_neighbors=k).fit(training, training_label)
    return neighbors.score(testing, testing_label)


if __name__ == '__main__':
    f = 'Data2.npz'

    accuracy = []
    accuracy.append(Q3_1_2(f, 1, 1))
    accuracy.append(Q3_1_2(f, 3, 1))
    accuracy.append(Q3_1_2(f, 5, 1))
    print('All genes:')
    print('k = [1, 3, 5]')
    print('accuracy =', accuracy)
    accuracy = []
    accuracy.append(Q3_1_2(f, 1))
    accuracy.append(Q3_1_2(f, 3))
    accuracy.append(Q3_1_2(f, 5))
    print('\nFirst 1000 genes:')
    print('k = [1, 3, 5]')
    print('accuracy =', accuracy)
