import numpy as np 
import tensorflow as tf 
import keras
import matplotlib.pyplot as plt 

#getting the dataset of numbers from 1 to 10 in mnist
mnist = tf.keras.datasets.mnist

(x_train,y_train) , (x_test,y_test) = mnist.load_data()

#normalizing the data in range 0 to 1 so it can be computed faster
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

#training the network with 2 layers each of 128 neurons
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)

print(' ')

#testing the network on unknown dataset
print('Testing on unknown data:')
val_loss,val_acc = model.evaluate(x_test,y_test)
print('Testing loss is: ',val_loss,'Testing accuracy is: ',val_acc)

predictions = model.predict(x_test[:10])

count = 0
for x in range(len(predictions)):
    guess = np.argmax(predictions[x])
    actual = y_test[x]
    print('The number is :',actual)
    print('The machine predicted the number is :',guess)
    if actual==guess:
        print('Machine\'s prediction is correct')
    else:
        print('Machine\'s prediction is wrong')
        count+=1
    
    plt.imshow(x_test[x])
    plt.show()

print("The program got", count, 'wrong out of', len(x_test))
print(str(100 - ((count/len(x_test))*100)) + '% correct')            


            

