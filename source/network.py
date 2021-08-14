from source.layers import *
import numpy
import time

class NeuralNetwork:
    def __init__(self,layers):
        self.layers = layers
        self.rng = numpy.random.RandomState()

    def _setup(self, X, Y, X_val, Y_val):

        # Setup layers sequentially
        next_shape = X.shape
        for layer in self.layers:
            layer.setup(next_shape, self.rng)
            next_shape = layer.output_shape(next_shape)

        valid_shape = X_val.shape
        for layer1 in self.layers:
            layer1.setup(valid_shape, self.rng)
            valid_shape = layer1.output_shape(valid_shape)

    def fit(self, X, Y, X_val, Y_val, learning_rate=0.1, max_iter=2, batch_size=64):


        old_error = 1.1

        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size

        n_class =  5
        Y_vector=self._vectorize(Y,n_class)


        n_cla =  5
        Y_vec=self._vectorize(Y_val,n_cla)

        start_time = time.time()

        iter = 0
        while iter < max_iter :
            start_iter_time = time.time()
            print("\nIteration : ",iter)
            iter += 1
            for b in range(n_batches):
                print("Batch : ",b)
                batch_begin = b * batch_size
                batch_end = batch_begin + batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y[batch_begin:batch_end]
                Y_batch_vector = self._vectorize(Y_batch,n_class)


                # Forward Propagation
                X_next = X_batch
                for layer in self.layers:
                    X_next = layer.forward_propogation(X_next)
                Y_pred = X_next # after Log Regression


                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch_vector, Y_pred)
                for layer in reversed(self.layers[:-1]):# Except the last layer
                    next_grad = layer.backward_propogation(next_grad)


                # Update parameters
                for i,layer in enumerate(self.layers):
                    if isinstance(layer, Conv) or isinstance(layer,Linear):
                        W,b = self.layers[i].params()
                        dW,db = self.layers[i].param_incs()
                        W = W - learning_rate*dW
                        b = b - learning_rate*db
                        layer.W = W
                        layer.b = b

            end_iter_time = time.time()
            print("Iteration time : ",(end_iter_time-start_iter_time)/60,"\n")
            print("Training Prediction : ")
            Y_pred = self.predict(X)
            print(Y_pred)
            error = self._error(Y_pred, Y)
            print("Training Error : ", error)
            acc = 1 - error
            print("Training Accuracy", acc)
            loss = self._loss(X, Y_vector)
            print("Training Loss: ", loss)

            print("Validation Prediction :")
            proba = self.predict(X_val)
            print("Actual  : ", Y_val)
            print("Predicted : ", proba)
            valerror = self.val_error(proba, Y_val)
            print("Validation Error : ", valerror)
            valoss = self.val_loss(X_val, Y_vec)
            print("Validation Loss: ", valoss)
            val_acc = 1 - valerror
            print("Validation Accuracy", val_acc)

            end_time = time.time()
            print("Training Time : ", (end_time - start_time)/60,"\n")

            save_loss = self.save_loss_acc(iter, acc, loss, val_acc, valoss)




    def _vectorize(self,Y,n_class):
        # convert number to bit vector representation(one hot encoding)
        vector = numpy.zeros((Y.shape[0],n_class))

        for i,y in enumerate(Y):
            class_y = int(y[0])
            vector[i,class_y]=1

        return vector

    def _loss(self, X, Y_vector):
        print("Loss : \n")
        X_next = X
        for layer in self.layers:
            X_next = layer.forward_propogation(X_next)
        Y_pred = X_next
        return self.layers[-1].loss(Y_vector, Y_pred)

    def val_loss(self, X_val, Y_vec):
        print("Loss : \n")
        X_next = X_val
        for layer in self.layers:
            X_next = layer.forward_propogation(X_next)
        Y_pred = X_next
        return self.layers[-1].valloss(Y_vec, Y_pred)

    def _error(self, Y_pred, Y):
        """ Calculate error on the given data. """
        error = numpy.zeros(len(Y_pred))
        for i in range(len(Y_pred)):
            if (Y_pred[i] != Y[i]):
                error[i] = 1
        return numpy.mean(error)

    def val_error(self, Y_pred, Y_val):
        """ Calculate error on the given data. """
        error = numpy.zeros(len(Y_pred))
        for i in range(len(Y_pred)):
            if (Y_pred[i] != Y_val[i]):
                error[i] = 1
        return numpy.mean(error)


    def predict(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
            X_next = layer.forward_propogation(X_next)
        Y_pred = X_next
        Y_pred = numpy.argmax(X_next,axis=-1)
        return Y_pred

    def predict_proba(self, X):
        """ Calculate an output Y for the given input X. """
        X_next = X
        for layer in self.layers:
            X_next = layer.forward_propogation(X_next)
        proba = X_next
        proba = numpy.argmax(X_next,axis=-1)
        return proba



    file = open("data.txt", "w")
    file.write("epoch,acc,loss,val_acc,valoss\n")
    file.close()

    def save_loss_acc(self, iter, acc, loss, val_acc, valoss):
        fil = open("data.txt", "a")
        iter = iter - 1
        fil.write(str(iter))
        fil.write(",")
        fil.write(str(acc))
        fil.write(",")
        fil.write(str(loss))
        fil.write(",")
        fil.write(str(val_acc))
        fil.write(",")
        fil.write(str(valoss))
        fil.write("\n")

        fil.close()



