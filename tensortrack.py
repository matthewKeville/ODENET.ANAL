#class tensortrack:
    #class variable that holds tensor data
    #forward_initial = []
    #forward_final = []
    #backward_initial = []
    #backward_final = []

    #debug = 0 
    #@staticmethod
    #def get_debug():
        #return debug

    #@staticmethod
    #def add_debug():
        #debug=debug+1
        #return

debug = 0
forward_initial = []

#@staticmethod
def get_debug():
    return debug

def add_debug():
    global debug #refer to the module global debug
    debug=debug+1 #mutate global
    return

def get_forward_initial():
    return forward_initial

def add_forward_initial_itr(y):
    global forward_initial
    forward_initial.append(y)



