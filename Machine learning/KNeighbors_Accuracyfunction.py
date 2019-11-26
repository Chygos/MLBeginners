def KNeighbour_Accuracy():

    '''Docstring:
    This plots the optimum n_neighbors for the KNeighbors classification model '''
    neighbors = np.arange(1, 10)
    accuracy_train = np.empty(len(neighbors))
    accuracy_test = np.empty(len(neighbors))

    for i, n in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        accuracy_test[i] = knn.score(X_test, y_test)  # individual test_accuracy score for every n looped
        accuracy_train[i] = knn.score(X_train, y_train)  # individual train_accuracy score for every n looped

    # plt.style.use('classic')
    plt.figure(figsize=(8, 5), frameon=True)
    plt.title("KNN: Varying Number of Neigbors")
    plt.plot(neighbors, accuracy_train, 'rs-', label='Train Accuracy')
    plt.plot(neighbors, accuracy_test, 'go-', label='Test Accuracy', )
    plt.legend()
    plt.xlabel('Number of Neighbours')
    plt.ylabel('Accuracy');
