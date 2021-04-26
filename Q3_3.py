from numpy import load, array
from sklearn.svm import LinearSVC


def Q3_3(file):
    with load(file) as data:
        training = data['training_data']
        training_label = data['training_label']
        testing = data['testing_data']
        testing_label = data['testing_label']
    temp = []
    for thing in training:
        temp.append(thing[:1000])
    training = temp
    temp = []
    for thing in testing:
        temp.append(thing[:1000])
    testing = temp

    linearSVC = LinearSVC(max_iter=3000).fit(training, training_label)
    return linearSVC.score(testing, testing_label)

if __name__ == '__main__':
    f = 'Data2.npz'
    print(Q3_3(f))
