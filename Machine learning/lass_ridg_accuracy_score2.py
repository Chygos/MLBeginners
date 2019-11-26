def lass_ridg_accuracy(alpha=0.1, testsize=0.3):

    '''Docstring:
        Takes alpha and test size to split'''
    try:

        room_train, room_test, medv_train, medv_test = \
            train_test_split(rm_np, MEDV_np, test_size=testsize, random_state=42)
        ridge = Ridge(alpha=alpha, normalize=True)
        lasso = Lasso(alpha=alpha, normalize=True)
        ridge.fit(room_train, medv_train)
        ridge.predict(room_test)
        lasso.fit(room_train, medv_train).predict(room_test)

        ridge_accuracy = ridge.score(room_test, medv_test)
        lasso_accuracy = lasso.score(room_test, medv_test)
        print(f'Alpha = {np.round((alpha), 2)}, R ={np.round((ridge_accuracy), 3)}, L ={np.round((lasso_accuracy), 3)}')
    except:
        print(f'Takes 2 positional arguments: alpha, testsize')