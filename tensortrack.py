#for every batch used in training (just for simplicity)
#store the forward pass values on each batch
#then store the backward pass values on each batch
#compute the tensor norm of the difference between the
#the estimates of the hidden state and the objective hidden state

#as each call to the forward pass is made, its iterates 
#are added to forward_batch

#then as the last backward iterate is added to backward_batch
#the norm is computed and add to the norm list


forward_batch = []
backward_batch = []
norms = []


def add_debug():
    global debug #refer to the module global debug
    debug=debug+1 #mutate global
    return

#@y the first n-1 iterates of the forward pass
#don't pass the final value
def add_forward_itr(y):
    global forward_batch
    forward_batch.append(y)

#@y the last n-1 iterates of the backward pass
def add_backward_itr(y):
    global backward_batch
    backward_batch.append(y)

#Assume len(forward_batch) == len(backward_batch)
#Norm computation will be based on forward vs backward
#for individual batches, i.e. if there is k nodes
#then for each batch of data there will be k norms
#each of the norms computed will be on tensors that
# are j x k x s, where s is the shape of a single dp as
# it traverses the process
# j is the number of time steps in each ODE block
# and k is the number of ode blocks in the process
def compute_norm():
    for x in range(len(forward_batch)):
        diff = torch.subtract(forward_batch[x] - backward_batch[x])

