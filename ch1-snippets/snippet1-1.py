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
    # single-neuron perceptron w/ 3 inputs (2 inputs + bias input)

    # 1st combo
    x1 = [1,-1,-1] # inputs (bias input must be 1)
    w1 = [0.9,-0.6,-0.5] # weights

    y1 = compute_output(w1, x1)

    print(y1)

    # 2nd combo
    x2 = [1,1,-1] # inputs (bias input must be 1)
    w2 = [0.9,-0.6,-0.5] # weights

    y2 = compute_output(w2, x2)

    print(y2)

    # 3rd combo
    x3 = [1,-1,1] # inputs (bias input must be 1)
    w3 = [0.9,-0.6,-0.5] # weights

    y3 = compute_output(w3, x3)

    print(y3)

    # 4th combo
    x4 = [1,1,1] # inputs (bias input must be 1)
    w4 = [0.9,-0.6,-0.5] # weights

    y4 = compute_output(w4, x4)

    print(y4)

if __name__ == "__main__":
    main()

