from tensorflow import keras

def main():
    model = keras.models.load_model('logs/2021-08-08-21-48-43/best_model')
    model.summary()

if __name__ == '__main__':
    main()