import torch
import numpy as np
from timeit import default_timer as timer
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#for every batch used in training (just for simplicity)
#store the forward pass values on each batch
#then store the backward pass values on each batch
#compute the tensor norm of the difference between the
#the estimates of the hidden state and the objective hidden state
#as each call to the forward pass is made, its iterates 
#are added to forward_batch
#then as the last backward iterate is added to backward_batch
#the norm is computed and add to the norm list
#forward_batch shape ( num_ode_blocks x time_steps x batch_size x hidden_state_shape)
#hidden_state_shape in odenet_cifar = ( 64 x 6 x 6)
forward_batch = np.empty([])
backward_batch = np.empty([])
norms = np.empty([])
training = True

time_steps = 1
batch_size = 1
epochs = 1
blocks = 1

#is the model training?
#this is set in the main method
#to pass information to the adjoint 
#process
def is_training():
    return training

#toggle training
def set_training(b):
    global training
    training = b
#set the training parameters
#initialize storage np arrays with respect to params
def init(nblocks,timesteps,batchsize,nepochs):
    global time_steps
    global batch_size
    global epochs
    global forward_batch
    global backward_batch
    global norms
    global blocks
    time_steps = timesteps
    batch_size = batchsize
    epochs = nepochs
    blocks = nblocks
    forward_batch = np.empty((0,time_steps,batch_size,64,6,6))
    backward_batch = np.empty((0,time_steps,batch_size,64,6,6))
    norms = np.empty((0,blocks,time_steps))
    print("init shapes : " + str(forward_batch.shape))
    print("init norm : " + str(norms.shape))
    blocks = nblocks

def reset_arrays():
    global forward_batch
    global backward_batch
    forward_batch = np.empty((0,time_steps,batch_size,64,6,6))
    backward_batch = np.empty((0,time_steps,batch_size,64,6,6))
    print("batch arrays reset")
    print("forward_batch size : " + str(forward_batch.shape))
    print("backward_batch size : " + str(backward_batch.shape))

#@y the first n-1 iterates of the forward pass
#don't pass the final value
# idk its being converted to a column vector ... 
def add_forward_itr(y):
    global forward_batch
    forward_batch = np.append(forward_batch,np.expand_dims(y.numpy(),axis=0),axis=0)
    #print(forward_batch.shape)

#@y the last n-1 iterates of the backward pass
def add_backward_itr(y):
    global backward_batch
    backward_batch = np.append(backward_batch,np.expand_dims(y.numpy(),axis=0),axis=0)

#Assume len(forward_batch) == len(backward_batch)
#Norm computation will be based on forward vs backward
#for individual batches, i.e. if there is k nodes
#then for each batch of data there will be k norms
#each of the norms computed will be on tensors that
# are j x k x s, where s is the shape of a single dp as
# it traverses the process
# j is the number of time steps in each ODE block
# and k is the number of ode blocks in the process
# we must compare the correct iterates for individual blocks
def compute_norm():
    global norms
    block_norms = np.empty((0,time_steps))
    start = timer()
    for x in range(len(forward_batch)):
        step_norms = np.array([])
        for y in range(time_steps):
            print(str(x) + " , " + str(y))
            diff = torch.subtract(torch.from_numpy(forward_batch[x][y]),torch.from_numpy(backward_batch[len(forward_batch) - x - 1][time_steps - y -1]))
            diffnp = diff.numpy()
            diffrs = np.reshape(diffnp, (1,diffnp.size))
            diffvec = torch.from_numpy(diffrs)
            diffnorm = torch.norm(diffvec)
            step_norms = np.append(step_norms,[diffnorm.item()])
            print("step norm shape : " + str(step_norms.shape))
            print("step norm " + str(step_norms))
        block_norms = np.append(block_norms,np.expand_dims(step_norms,axis=0),axis=0)
        print("block norms shape : " + str(block_norms.shape))
        print("block norms " + str(block_norms))

    print("norms shape : " + str(norms.shape))
    print("norms " + str(norms))
    norms = np.append(norms,np.expand_dims(block_norms,axis=0),axis=0)
    compute_time = timer() - start
    print("compute norm time : " + str(compute_time))

def get_norms():
    return norms

#pickled as tuple (time_steps,num_blocks,batch_size,norms list)
def save():
    print("saving")
    with open('analysis.pik',"wb") as f:
        sv = (time_steps,blocks,batch_size,norms)
        pickle.dump(sv,f)


#running after training
#supply pickle dump
if __name__ == "__main__":
    print("running")
    #unpickle
    with open("analysis.pik","rb") as f:
        sv = pickle.load(f)

    time = sv[0]
    blocks = sv[1]
    bs = sv[2]
    norms = sv[3]

    print("time: " + str(time) + "\n number of blocks: " + str(blocks) + "\ batch size: " + str(bs))
    print("size : " + str(norms.size))
    print("shape : " + str(norms.shape))

    #the form the datamatrix is epochs x odeblock x time_step
    #the individual batch differences are useful for analysis but a
    #graph of batch instabiltiy against time is just a random signal

    #Some suggestions for analyzing instability :
        #Total Batch Difference (1)
        #Take the sum of each difference in the epoch

        #Total Batch Difference Average (2)
        #Same as above but divide by number of datapoints dp = batch_size * odeblocks * timestesp 

        #Epoch vs ODEblock vs TimeStep vs Instability or a subset of such

    fig = plt.figure(figsize=(9,9))
    fig.suptitle('Insability')
    gs = gridspec.GridSpec(2,2)
    x = np.arange(norms.size/blocks)
    #diffs = np.ravel(norms,order='c')
    block1 = norms[:,0,:]
    block2 = norms[:,1,:]
    print('diffs size' + str(block1.size))
    print('diffs size' + str(block2.size))

    ax = plt.subplot(gs[0,:])
    ax.set_title('Instability')
    plt.plot(x,diffs)
    plt.show()





