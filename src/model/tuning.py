    def tune_parameters(x_train, y_train, x_test, y_test):
            """
        Model providing function:

        Create Keras model with double curly brackets dropped-in as needed.
        Return value has to be a valid python dictionary with two customary keys:
            - loss: Specify a numeric evaluation metric to be minimized
            - status: Just use STATUS_OK and see hyperopt documentation if not feasible
        The last one is optional, though recommended, namely:
            - model: specify the model just created so that we can later use it again.
        """
        model = Sequential()

        model.add(LSTM(self.neurons[0], input_shape=(self.shape[0], self.shape[1]), return_sequences=True))
        model.add(Dropout({{uniform(0, 1)}}))

        model.add(LSTM(self.neurons[1], input_shape=(self.shape[0], self.shape[1]), return_sequences=False))
        model.add(Dropout({{uniform(0, 1)}}))

        model.add(Dense(self.neurons[2],kernel_initializer="uniform",activation='relu'))
        model.add(Dense(self.neurons[3],kernel_initializer="uniform",activation='linear'))

        #model.add(Dense(512, input_shape=(784,)))
        #model.add(Activation('relu'))
        
        #model.add(Dense({{choice([256, 512, 1024])}}))
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        model.add(Dropout({{uniform(0, 1)}}))

        # If we choose 'four', add an additional fourth layer
        if conditional({{choice(['three', 'four'])}}) == 'four':
            model.add(Dense(100))

            # We can also choose between complete sets of layers

            model.add({{choice([Dropout(0.5), Activation('linear')])}})
            model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                    optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

        model.fit(x_train, y_train,
                batch_size={{choice([64, 128])}},
                epochs=1,
                verbose=2,
                validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Test accuracy:', acc)
        return {'loss': -acc, 'status': STATUS_OK, 'model': model}