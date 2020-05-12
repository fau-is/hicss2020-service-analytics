import keras
import util
from datetime import datetime


def train(args, preprocess_manager):
    util.llprint("Loading Data starts... \n")
    X, y, sequence_max_length, num_features_all, num_features_activities = preprocess_manager.create_and_encode_training_set(
        args)
    util.llprint("Loading Data done!\n")

    print('Build model...')

    # LSTM
    if args.dnn_architecture == 0:
        main_input = keras.layers.Input(shape=(sequence_max_length, num_features_all), name='main_input')
        l1 = keras.layers.recurrent.LSTM(100, implementation=2, activation="tanh", kernel_initializer='glorot_uniform',
                                         return_sequences=False, dropout=0.2)(main_input)
        b1 = keras.layers.normalization.BatchNormalization()(l1)

    # GRU
    elif args.dnn_architecture == 1:
        main_input = keras.layers.Input(shape=(sequence_max_length, num_features_all), name='main_input')
        l1 = keras.layers.recurrent.GRU(100, implementation=2, activation="tanh", kernel_initializer='glorot_uniform',
                                        return_sequences=False, dropout=0.2)(main_input)
        b1 = keras.layers.normalization.BatchNormalization()(l1)

    # RNN
    elif args.dnn_architecture == 2:
        main_input = keras.layers.Input(shape=(sequence_max_length, num_features_all), name='main_input')
        l1 = keras.layers.recurrent.SimpleRNN(100, implementation=2, activation="tanh",
                                              kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(
            main_input)
        b1 = keras.layers.normalization.BatchNormalization()(l1)

    activity_output = keras.layers.Dense(num_features_activities + 1, activation='softmax', name='activity_output',
                                         kernel_initializer='glorot_uniform')(b1)
    model = keras.models.Model(inputs=[main_input], outputs=[activity_output])

    optimizer = keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                       schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'activity_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        '%smodel_%s.h5' % (args.checkpoint_dir, preprocess_manager.iteration_cross_validation), monitor='val_loss',
        verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
    model.summary()

    start_training_time = datetime.now()

    model.fit(X, {'activity_output': y}, validation_split=1 / args.num_folds,
              callbacks=[early_stopping, model_checkpoint, lr_reducer], verbose=1, batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    return training_time.total_seconds()
