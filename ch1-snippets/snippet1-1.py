# first element in vector x must be 1
# length of w & x must be n+1 for neuron w/ n inputs

def compute_output(w,x):

    z = 0.0
    for i in range(len(w)):

        z += x[i] * w[i] # compute sum of weighted inputs
        

    if z < 0: # apply sign function
        return -1
    else:
        return 1

def main():
    # single-neuron perceptron w/ 3 inputs
    x = [1,2,3] # inputs
    w = [4,5,-6] # weights

    y0 = compute_output(w, x)

    print(y0)

if __name__ == "__main__":
    main()

