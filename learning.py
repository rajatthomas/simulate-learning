# -*- coding: utf-8 -*-
# Simulations to compare different learning rules and architectures
# in Learning theory. Primary paradigms are Hebbian & the Associative
# learning theory
#
# References:
#  1. R. P. Cooper, C. Catmur, C. Heyes ⁄ Cognitive Science 37 (2013)
#  2. R.P. Cooper et al. / Neuroscience Letters 540 (2013) 28–36


# version 0.0 19-11-2013

import itertools
import random

import numpy as np
import matplotlib.pyplot as plt


class neurons:
    
    def __init__(self, **kwargs):
        self.vars  = kwargs;
        self._val=[];
        self._val.append(self.vars.get('value',0.0)); # list storing activation-timeseries
        self._type = self.vars.get('type',None);
        self._ID = self.vars.get('ID',None);
        self._counter = 0; # keep count of total activity iteration
        self._respcount = 0; # keep count of response
        # Parameters set using :
        # 1. R. P. Cooper, C. Catmur, C. Heyes ⁄ Cognitive Science 37 (2013)
        # 2. R.P. Cooper et al. / Neuroscience Letters 540 (2013) 28–36
        
        if self._type == 'sensory':
            self._persistence = 0.960;
            self._bias = -2.0;
            self._habitThreshold = 0.90; # Habituation threshold
            self._Ei = 5.0; # Ei keeps original value
            self._exciteInput = self._Ei; # Ei is eq (2)ref 2., from ref. 1 Table-1
            self._noisestd = 2.0;
            
        if self._type == 'context':
            self._persistence = 0.960;
            self._bias = -2.0;
            self._habitThreshold = 0.90;
            self._Ei = 5.0; # Ei keeps original value
            self._exciteInput = self._Ei; # for C_imp and C1 and C2 various 
            self._noisestd = 2.0;
            
        if self._type == 'imperative':
            self._persistence = 0.990;
            self._bias = -2.0;
            self._Ei = 5.0; # Ei keeps original value
            self._exciteInput = self._Ei;
            self._noisestd = 2.0;
            
        if self._type == 'motor':
            self._persistence = 0.990;
            self._bias = -6.0;
            self._exciteInput = 0.0;  # No external drive for o/p nodes
            self._respThreshold = 0.80; # response threshold
            self._noisestd = 2.0;
            self._resptime = [np.inf]*72*6; # Response time 72 trials per block (6 blocks)
    
    def increment_counter(self):
        self._counter += 1;
             
    def set_var(self,k,v):
        self.vars[k] = v;
        # set all possible things that might be modified
        if k=='value':
            self._val.append(self.vars['value']);
            self._counter += 1
            
        self._type = self.vars['type'];
        
        
    def get_var(self,k):
        return self.vars.get(k,None)
               
        

def get_var_allneurons(neurons,k):
    
    result = [];
    if k=='val':
        for i in range(len(neurons)):
            result.append(neurons[i]._val[neurons[i]._counter-1]);
                # check eq (2) of Ref (2) for why the -1
             
    return np.array(result)
     
   
def initialstate_network():
    
    # Set network according to Cooper et al.,2012 (Mini Review) fig.2

    num_inputs  = 8; 
    # split as 
    num_sensory = 2;
    num_context = 4;
    num_imperative = 2;
    
    num_outputs = 2;
    
    # parameters nodes
    
    input_nodes = [];
    motor_nodes = [];
    
    # Initial weights of sensory nodes ( open & close in figure )
    input_nodes.append(neurons(type='sensory', value=0.15, ID=0));
    input_nodes.append(neurons(type='sensory', value=0.13, ID=1));

    
    # Initial context nodes (1, c1, c2, 2)
    for i in range(num_context):
        input_nodes.append(neurons(type='context', value=0.12, ID=i+num_sensory));
    
    # Initial imperative nodes (1,2)
    for i in range(num_imperative):
        input_nodes.append(neurons(type='imperative', value=0.12, ID=i+(num_sensory+num_context)));
                   
                                                 
    # Initial motor nodes ( open and close ) 
    for i in range(num_outputs):
        motor_nodes.append(neurons(type='motor', value=0.0, ID=i));
        
    #---------------------------------------------------------------------------        
    # Initialize the values of the nodes based on their biases
    for neuron in input_nodes:
        neuron.set_var('value',math_sigmoid(neuron._bias));
    
    for neuron in motor_nodes:
        neuron.set_var('value',math_sigmoid(neuron._bias));
    #---------------------------------------------------------------------------  
      
    # setup weights of the network
    weights = np.zeros( (num_inputs, num_outputs) );        
        
    # Initial weights for weights between the nodes
    # From inputs to motor neuron 'open'
    weights[0][0] = 4.0;
    weights[1][0] = 0.0;
    weights[2][0] = 0.0;
    weights[3][0] = 0.0;
    weights[4][0] = 0.0;
    weights[5][0] = 0.0;
    weights[6][0] = 10.0;
    weights[7][0] = 0.0;

    # From inputs to motor neuron 'close'
    weights[0][1] = 0.0;
    weights[1][1] = 4.0;
    weights[2][1] = 0.0;
    weights[3][1] = 0.0;
    weights[4][1] = 0.0;
    weights[5][1] = 0.0;
    weights[6][1] = 0.0;
    weights[7][1] = 10.0;
    
    return input_nodes,motor_nodes,weights
    
    
def burnin_network(ineurons=[], oneurons=[], w = np.array([]), samples=500):
    
    
    for i in range(samples):
        # Calculate activation of neuron ( Eq.1 )
        [ineurons,oneurons] = neuron_update(ineurons,oneurons,w,'burnin');
        # Check conditions of threshold
        ineurons = neuron_habituations(ineurons);
    
    return ineurons,oneurons,w
    
def train_network(ineurons=[], oneurons=[], w = np.array([]), max_sample=1000, \
    learning_rule = 'hebbian', group_ID = 'contingent', params=[0.002,10.0,8.0],\
    blocks=6, trialsperblock=72):
    
    rate      = params[0];
    alpha_tau = params[1];
    contingent_trials = [1]*blocks*trialsperblock;
    non_or_sig_trials = [0]*blocks*trialsperblock;
    
    if group_ID == 'contingent':
        all_trials = contingent_trials;
    else:
        # flatten array
        all_trials = list(itertools.chain([contingent_trials, non_or_sig_trials]));
    
    # randomize trials but keep number of contingent trials constant
    random.shuffle(all_trials);  # 1s and 0s will be random now
   
    for trial in all_trials:

        reinitialize_Ei(ineurons);

        if trial == 1:
           [stimulus, response] = pattern_gen('contingent'); # every gets equal contingents
        else:
           [stimulus, response] = pattern_gen(group_ID); # mixed in with others for rest

        #w = weight_update(w, stimulus, response,learning_rule,rate,alpha_tau);
        w = weight_update(w, stimulus, response, ineurons, oneurons, learning_rule,rate,alpha_tau);
        iterations = 0;
        while (response_trigger(oneurons,iterations)) == False and (iterations < max_sample): 
            [ineurons,oneurons] = neuron_update(ineurons,oneurons,w,'training',stimulus,params[2]);
            ineurons = neuron_habituations(ineurons);   
            iterations += 1;
                    
    
    return ineurons, oneurons, w
    
def reinitialize_Ei(ineurons):
    
    for x in ineurons:
        x._exciteInput = x._Ei;
        
    return ineurons    
    
def response_trigger(oneurons, iterations):
    
    for x in oneurons:
        if x._val[x._counter] > x._respThreshold:
            #x._respcounter.append(iterations);
            return True
        else:
            return False
        
    
# Simple func to return patterns corresponding to contingent, non-contingent & Signalled    
def pattern_gen(train_type='contingent'):

    # stimulus pattern      = [So, Sc, 1, c1, c2, 2, Io, Ic]; i/p nodes of fig (2)
    # response pattern      = [open, close]
    
    # info about context nodes 1, c1, c2, 2:
    # c1 active for standard trial ( real hand )
    # c2 active for signalled color hand
    # 1 active with value Cimp if Io is active
    # 2 active with value Cimp if Ic is active
    
    if train_type == 'contingent':
        if random.random() < 0.5: # coin toss
           # open hand stimulus (image); # close hand imperative -> close hand response
           stimulus = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
           response = np.array([0.0, 1.0]);  
        else :
           # close hand stimulus (image); # open hand imperative -> open hand response
           stimulus = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
           response = np.array([1.0, 0.0]);
        
    if train_type == 'non_contingent':
        if random.random() < 0.5: # coin toss
           # neutral real hand stimulus (image); # close hand imperative -> close hand response
           stimulus = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
           response = np.array([0.0, 1.0]);  
        else :
           # neutral real hand stimulus (image); # open hand imperative -> open hand response
           stimulus = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
           response = np.array([1.0, 0.0]);  
    
    if train_type == 'signalled':
        if random.random() < 0.5: # coin toss
           # neutral color hand stimulus (image); # close hand imperative -> close hand response
           stimulus = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
           response = np.array([0.0, 1.0]);  
        else :
           # neutral color hand stimulus (image); # open hand imperative -> open hand response
           stimulus = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
           response = np.array([1.0, 0.0]);  
           
    return stimulus, response

def neuron_habituations(ineurons=[]):
    
    for x in ineurons:
        if (x._type == 'sensory') or (x._type == 'context'):
            if x._val[x._counter] >= x._habitThreshold:
                x._exciteInput = 0.0;
            
    return ineurons

def neuron_update(ineurons=[], oneurons=[], w = np.array([]),phase='burnin',stimulus=np.array([]),c2=0):
    
    # update input neurons
    for i in range(len(ineurons)):
        x = ineurons[i]; # short hand only
        net_input = sum_input(1,i,ineurons,oneurons,w,phase,stimulus,c2)
        new_val = x._persistence*x._val[x._counter] +(1-x._persistence)*math_sigmoid(net_input);
        ineurons[i]._val.append(new_val);
        #Update counter ( counter roughly translates to time in ms )
        x.increment_counter()
    
    # update output neurons
    for i in range(len(oneurons)):
        x = oneurons[i]; # short hand only
        net_input = sum_input(0,i,ineurons,oneurons,w,phase,stimulus,c2)
        new_val = x._persistence*x._val[x._counter] +(1-x._persistence)*math_sigmoid(net_input);
        oneurons[i]._val.append(new_val);
        x.increment_counter()
        
    return ineurons,oneurons
    

def sum_input(update_input, index,ineurons, oneurons, weights= np.array([]), \
            phase='burnin', stimulus=np.array([]), C2=0):
    
    Ij = 0;
    
    # only for the output node
    if update_input == 0: # i.e., update output
        Ij = np.sum(weights[:,index]*get_var_allneurons(ineurons,'val'));
        Ij += oneurons[index]._bias + np.random.normal(0.0,oneurons[index]._noisestd);
    
    else:       
        activeStrength = 1.0;
        if phase == 'burnin':
            activeStrength = 0.0;
        
        if (phase == 'training'):
            activeStrength = stimulus[index]; 
            if (ineurons[index]._ID==4):
                activeStrength = C2*stimulus[index]; #._ID == 4 is the C2 node (same  as stim[4] )
        
        Ij += ineurons[index]._bias + activeStrength*ineurons[index]._exciteInput + \
            np.random.normal(0.0,ineurons[index]._noisestd);
    
    return Ij
    
def math_sigmoid(x):
    
    return 1.0/(1.0 + np.exp(-x) );
    

def weight_update(w, stimulus, response, ineurons, oneurons, learning_rule = 'hebbian',rate=0,alpha_tau=0):
    
    if learning_rule == 'hebbian' :
        w = learning_hebbian(w,stimulus, response,rate,alpha_tau);    
    
    if learning_rule == 'quasihebbian' :
        w = learning_quasihebb(w,stimulus, response,rate,alpha_tau); 
        
    if learning_rule == 'RW' :
        w = learning_RW(w,ineurons,rate,alpha_tau); 
        
    if learning_rule == 'covhebbian' :
        w = learning_covhebb(w,ineurons, oneurons,rate); 
        
    return w
    

def learning_hebbian(w, stimulus, response, rate, alpha):
    
    
    ins,outs = w.shape;
    
    # alpha -> weight asymptote 
    
    for i in range(ins):
        for j in range(outs):
            dw = (stimulus[i]*response[j])*(alpha - w[i,j])/alpha;
            # Note: (stimulus[i]*response[j]) = 1 only if both are ON 
                     
            w[i,j] = w[i,j] + rate*dw;
 
    return w
    
def learning_quasihebb(w, stimulus, response, rate, alpha):
    
    # alpha -> weight asymptote
    
    ins,outs = w.shape;
    
    for i in range(ins):
        for j in range(outs):
            dw = 0;
            
            if (stimulus[i] != 0) and (response[j] !=0):
                dw = (alpha - w[i,j])/alpha;
                
            if (stimulus[i] != 0) and (response[j] == 0):    
                dw = -(alpha - w[i,j])/alpha;
                 
            w[i,j] = w[i,j] + rate*dw;
            
    return w
    
def learning_RW(w, ineurons, rate, tau):
    
    # tau -> target input to drive a node to its maximum
    
    ins, outs = w.shape;
        
    for i in range(outs):
        epsj = tau -  np.sum(w[:,i]*get_var_allneurons(ineurons,'val'));
        
        for j in range(ins):
            dw = epsj*ineurons[j]._val[ineurons[j]._counter];
            w[j,i] = w[j,i] + rate*dw;
    
    return w
    
def learning_covhebb(w, ineurons, oneurons, rate):
    
    ins, outs = w.shape;
    
    print(get_var_allneurons(ineurons,'val'))
    print(get_var_allneurons(oneurons,'val'))
    raw_input()
    for i in range(ins):
        for j in range(outs):
            
            dw = np.corrcoef(ineurons[i]._val,oneurons[j]._val);
            
            w[i,j] = w[i,j] + rate*dw[0,1]; # dw
            
    return w    

def record_responsetime():
    
    pass
    

def plot_node(ineurons, oneurons, w, phase='burnin'):
    
    plt.close("all");
    
    font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 16,
        }

#    t = np.linspace(0.0, 500.0, len(oneurons[0]._val))

#    t = np.arange(len(oneurons[0]._val))
    p1, = plt.plot(oneurons[0]._val)
    p2, = plt.plot(oneurons[1]._val)
    
    plt.legend([p1,p2], ['$\mathcal{M}_0(t)$','$\mathcal{M}_1(t)$']);
    
    plt.axis([0, max(len(oneurons[0]._val),len(oneurons[1]._val)), 0, 1.0])
    
    if phase == 'burnin':
        plt.title('Burn-in (Motor neurons) ', fontdict=font)
    
    if phase == 'training':
        plt.title('Training (Motor neurons) ', fontdict=font)
    
    plt.text(200, 0.5, r'$\mathcal{M}(t)$', fontdict=font)
    plt.xlabel('time (~ms)', fontdict=font)
    plt.ylabel('Activity', fontdict=font)

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


    # Plot the input neurons
   
    legs1 = []
    legs2 = []
    legs3 = []
    
    f, axarr = plt.subplots(3, sharex=True)
    
    axarr[0].set_title(' Sensory Neurons ')
    axarr[1].set_title(' Context Neurons ')
    axarr[2].set_title(' Imperative Neurons ')
    
    max_x = 0;
    for i in range(0,2):
        p, = axarr[0].plot(ineurons[i]._val);
        max_x = max(max_x,len(ineurons[i]._val));
        legs1.append(p)
        
    axarr[0].legend(legs1, ['$\mathcal{S}_0(t)$','$\mathcal{S}_1(t)$'],loc=0)
    
    for i in range(2,6):
        p, = axarr[1].plot(ineurons[i]._val);
        max_x = max(max_x,len(ineurons[i]._val));
        legs2.append(p)
        
    axarr[1].legend(legs2, ['$1(t)$','$\mathcal{C}_1(t)$','$\mathcal{C}_2(t)$','$2(t)$'],ncol=4,loc=0)
    axarr[1].set_ylabel('Activation', fontdict=font)

    for i in range(6,8):
        p, = axarr[2].plot(ineurons[i]._val);
        max_x = max(max_x,len(ineurons[i]._val));
        legs3.append(p)
    
    axarr[2].legend(legs3, ['$\mathcal{I}_1(t)$','$\mathcal{I}_2(t)$'], loc=0);
    
    axarr[0].axis([0, max_x, 0, 1.0])
    axarr[1].axis([0, max_x, 0, 1.0])
    axarr[2].axis([0, max_x, 0, 1.0])
    
    
    plt.xlabel('time (~ms)', fontdict=font)
    

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

    # Plot the weights 
    #plt.figure()
    #hinton(w)
    #plt.show()
   
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x,y),w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
 
    
def main():
    
    print '\n'
    print """ Simulation of Learning Rules """
  
    # Learning rate ( see Ref.2 for values chosen)
    #lrate = np.arange(0.0002, 0.0060, 0.0002);
    
    # weight asymptote 
    #alpha_tau = np.range(2,20,2);
    
    # Signalling C2 values 
    #c2 = np.range(4,12,2);
    
  
    
    lrate = .0044; 
    alpha_tau = 2.0;
    c2 = 12.0;
    
    print """ The Hebbian Learning paradigm """

    print """ -------------------------------------"""  
    print """ The CONTINGENT case """
    print """ -------------------------------------"""

    print '\n Initializing network ' 
    # Set network with initial weights and node status
    [input_neurons, motor_neurons, weights] = initialstate_network()

    print '\n Running the current network to steady-state (typically 500 ms)'    
    [input_neurons, motor_neurons, weights] = burnin_network(input_neurons, motor_neurons, weights, samples=500)
    
    # plot the burnin results
    plot_node(input_neurons, motor_neurons, weights);

    # Just a pause .. press enter to continue    
    pause = raw_input('press enter to continue ... ');    

        
    print '\n Training network -- Contingent Group ' 
    [cont_input_neurons, cont_motor_neurons, cont_weights]  = \
        train_network(input_neurons, motor_neurons, weights, max_sample=1000, \
        learning_rule = 'hebbian', group_ID = 'contingent', params=[lrate,alpha_tau,c2],\
        blocks=1, trialsperblock=1);
        
        
    plot_node(cont_input_neurons, cont_motor_neurons, cont_weights,'training');
       
        
    print '\n Response times being calculated -- Testing phase ' 
    
    # Begin testing and calculate response times
    
    # plot results
    return True

if __name__ == '__main__': 
    main()
    