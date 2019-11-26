#checks the test size split with the best accuracy
def DecisionTree_accuracy():
    '''Docstring:
    This checks the ideal testsize to get the maximum accuracy '''

    testsize = np.round(np.linspace(0.2,0.4, 5),2)
    testAcc = np.empty(len(testsize))
    for i, n in enumerate(testsize):
        X_train, X_test, y_train, y_test = \
        train_test_split(X,y, random_state = 0, stratify = y, test_size = n)
        classifier = DecisionTreeClassifier(random_state = 0)
        classifier.fit(X_train,y_train)
        classifier.predict(X_test)
        testAcc[i] = classifier.score(X_test,y_test)
        print(f'The accuracy of when testsize is {n} is {np.round((testAcc[i]),3)}')
    plt.figure(dpi = 80, figsize = [7,5])
    plt.plot(testsize,testAcc, color = '#1fc212',  marker = 'x', ls = '--', alpha =0.9);
    plt.ylim([0.93,0.98])
    plt.xlabel('Test Size')
    plt.ylabel("Test Accuracy")
    plt.title('Test accuracies at varying test sizes');
