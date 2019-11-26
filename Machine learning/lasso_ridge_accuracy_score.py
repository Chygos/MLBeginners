def lasso_ridge_accuracy_score():
    '''Docstring:
    This checks the best alpha (n) value for the model'''

    alphas = np.linspace(0.1, 0.9, 9)
    ridge_accuracy = np.empty(len(alphas))
    lasso_accuracy = np.empty(len(alphas))
    for i, n in enumerate(alphas):
        room_train, room_test, medv_train, medv_test = \
            train_test_split(rm_np, MEDV_np, test_size=0.3, random_state=42)
        ridge = Ridge(alpha=n, normalize=True)
        lasso = Lasso(alpha=n, normalize=True)
        ridge.fit(room_train, medv_train)
        ridge.predict(room_test)
        lasso.fit(room_train, medv_train).predict(room_test)

        ridge_accuracy[i] = ridge.score(room_test, medv_test)
        lasso_accuracy[i] = lasso.score(room_test, medv_test)
        print(
            f'Alpha = {np.round((n), 2)}, R ={np.round((ridge_accuracy[i]), 3)}, L ={np.round((lasso_accuracy[i]), 3)}')

    plt.plot(alphas, ridge_accuracy, 'rs-', label='Ridge Accuracy')
    plt.plot(alphas, lasso_accuracy, 'go-', label='Lasso Accuracy')
    plt.xlim([0, 1])
    plt.ylim([-0.1, 0.5])
    plt.grid(True)
    plt.legend(loc=5);