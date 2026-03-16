Basics
What is NNUE?
NNUE (ƎUИИ Efficiently Updatable Neural Network) is, broadly speaking, a neural network architecture that takes advantage of having minimal changes in the network inputs between subsequent evaluations. It was invented for Shogi byYu Nasu, integrated into YaneuraOu developed by Motohiro Isozaki in May 2018, and later ported to chess for use in Stockfish byHisayori Nodain June 2019, but is applicable to many other board games and perhaps even in other domains. NNUE operates on the following principles:

The network should have relatively low amount of non-zero inputs.
The inputs should change as little as possible between subsequent evaluations.
The network should be simple enough to facilitate low-precision inference in integer domain.
Following the 1st principle means that when the network is scaled in size the inputs must become sparse. Current best architectures have input sparsity in the order of 0.1%. Small amount of non-zero inputs places a low upper bound on the time required to evaluate the network in cases where it has to be evaluated in its entirety. This is the primary reason why NNUE networks can be large while still being very fast to evaluate.

Following the 2nd principle (provided the first is being followed) creates a way to efficiently update the network (or at least a costly part of it) instead of reevaluating it in its entirety. This takes advantage of the fact that a single move changes the board state only slightly. This is of lower importance than the first principle and completely optional for the implementations to take advantage of, but nevertheless gives a measurable improvement in implementations that do care to utilize this.

Following the 3rd principle allows achieving maximum performance on common hardware and makes the model especially suited for low-latency CPU inference which is necessary for conventional chess engines.

Overall the NNUE principles are applicable also to expensive deep networks, but they shine in fast shallow networks, which are suitable for low-latency CPU inference without the need for batching and accelerators. The target performance is million(s) of evaluations per second per thread. This is an extreme use case that requires extreme solutions, and most importantly quantization.

Quantization 101 and its importance
Quantization is the process of changing the domain of the neural network model from floating point to integer. NNUE networks are designed to be evaluated fast in low-precision integer domain, and can utilize available int8/int16 performance of modern CPUs to the fullest extent. Floating point is not an option for achieving maximum engine strength as it sacrifices too much speed for too little accuracy gains (though floating point representation is used by some engines due to its simplicity). Quantization inevitably introduces error that accumulates more the deeper the network is, however in the case of NNUE networks, which are relatively shallow, this error is negligible. Quantization will be described in more detail later in this document. Until then this document will be using floats instead of ints, it won't be important until we get to actual code optimization. The purpose of this interjection is to make the reader aware of the ultimate goal of NNUE, as it is the biggest factor that shapes the NNUE models and dictates what is possible and what is not.

What layers are useful in NNUE?
NNUE relies on simple layers that can be implemented in low-precision environments using simple arithmetic. This means Linear (fully connected, basically matrix multiplication) and ClippedReLU (clamp(0, 1)) layers are particularly suitable for it. Pooling layers (mul/avg/max) or approximations of more complex activation functions (like sigmoid) are also suitable but not commonly used.

Usually, such networks are kept shallow (2-4 layers), because most knowledge is kept in the first layer (which takes advantage of input sparsity to remain performant) and after that first layer the network needs to sharply reduce its width (the benefits of a deeper section in the later parts of the net would be dominated by the impact of the large first layers) to maintain performance requirements.

Linear layer
A linear (fully connected) layer is just a simple matrix multiplication. It can be implemented efficiently, supports sparse inputs, and provides good capacity. It takes as an input in_features values, and produces out_features values. The operation is y = Ax+b, where:

x - the input column vector of size in_features

A - the weight matrix of size (out_features, in_features)

b - the bias column vector of size out_features

y - the output column vector of size out_features

Matrix vector multiplication

Linear layer with sparse inputs
The multiplication Ax can be conceptually simplified to "if x[i] is not zero then take column i from A, multiply it by x[i] and add it to the result". Now it should be obvious that whenever an element of the input is zero we can skip processing the whole column of the weight matrix. This means that we have to only process as many columns of A as there are non-zero values in the input vector. Even though there may be tens of thousands of columns in the weight matrix, we're only concerned about a few of them for each position! That's why the first layer can be so large.

Matrix and sparse vector multiplication

Clipped ReLU layer
This is an activation function based on normal ReLU, with the difference that it is bounded both from below and above. The formula is y = min(max(x, 0), 1).

ClippedReLU

The purpose of this layer is to add non-linearity to the network. If it was just linear layers they could all be collapsed into one, because the matrices could be just multiplied together.

ClippedReLU would ideally be replaced with ReLU, but aggressive quantization requires reducing the dynamic range of hidden layer inputs, so capping the values at 1 becomes important for performance.

Sigmoid
This is an activation function that, contrary to [clipped] ReLU, is smooth. The formula is y = 1/(1+e^-kx), where k is a parameter that determines how "stretched" the shape is.

Sigmoid

There are two main differences compared to clipped ReLU:

sigmoid is smooth, meaning that it is differentiable everywhere, meaning that there are no situations (realistically speaking) where the gradient disappears.
sigmoid is nonlinear, the output saturates towards 0 or 1 but never reaches it
While this function generally allows the network to learn more than ReLU it is costly and unsuitable for evaluation in the integer domain. It is however a good starting point for improvements...

Quantmoid4
With sigmoid being too costly we need to look for alternatives. One such alternative is to use an approximation. And it just so happens that sigmoid(4x) (scaled to integer domain in a particular way) can be fairly well approximated by a simple piece-wise quadratic function that needs just addition, multiplication, and bit-shifts. Since the primary purpose for this approximation is to be used in a quantized implementation directly we will present a specific variant that outputs values in range [0, 126] (and with input scaled accordingly). The reason for the choice of the upper range being defined as 126 is that this is the largest even 8-bit integer, and we want an even one to allow the value for x=0 to be exactly in the middle. The equation is as follows:

Quantmoid4 Equation

Note, that the equation for both positive and negative x is almost identical. The similarity allows for a branchless implementation even though there are two cases.

And the resulting graph is the following (with a scaled sigmoid(4x) for comparison):

Quantmoid4

The disadvantage is that it loses the smoothness, and the output rounds to 0/1 quite early. This however doesn't appear to be an issue in practice, the actual error from this "rounding" is negligible.

More cool stuff will happen once we implement and optimize it, so we will get back to this layer in the optimized quantized implementation section.

Pooling layers
Sometimes it is desirable to reduce the input dimensionality to make the size of the layer more approachable. For example instead of having a 1024->8 layer, which has a very narrow output, one may prefer 512->16. Pooling layers can provide some flexibility by reducing the dimensionality.

Pooling layers work by applying a function F over non-overlapping spans of the input, where F has more inputs than outputs. So for example one may have F take 2 consecutive inputs and produce one output, effectively halving the number of neurons.

The following types of pooling layers can be considered:

Average Pooling - outputs the average of inputs. Works well with any number of inputs.
Max Pooling - outputs the maximum of inputs. Works well with any number of inputs.
Product Pooling - outputs the product of inputs. Introduced by Stockfish, not common in machine learning in general. Only works well with 2 inputs. This one also appears to have similar benefits to sigmoid (quantmoid4); it increases the network's capacity, while other pooling layers only allow reducing dimensionality.
A simple input feature set.
For the purpose of illustration we will consider a simple set of inputs based on piece placement. We will call it "A" features, because they will represent "All pieces".

There are 64 squares on the board, 6 piece types (pawn, knight, bishop, rook, queen, king), and 2 colors (white, black). What we want to encode as inputs are the positions of pieces, so each input will correspond to some (square, piece_type, color) tuple. There are 64*6*2=768 such tuples. If there is a piece P of color C on the square S we set the input (S, P, C) to 1, otherwise, we set it to 0. Even though the total number of inputs is 768 there can only be 32 non-zero inputs in any given legal chess position, because there are only at most 32 pieces on the board. Moreover, any move can only change at most 4 inputs (castling), and the average should be below 3.

The binary and sparse nature of the inputs is utilized when passing the features to the neural network - the input is simply the list of features (indices), there's no need for a full input vector as other positions have value 0 and we know that each active feature has a value 1 associated with it.

Let's look at an example position 1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1.



On the board above we have 4 active features: (A1, king, white), (C3, pawn, white), (B8, king, black), (D4, rook, black).

Now let's consider the move c4 - the only feature that became invalid is the (C3, pawn, white), it needs to be replaced with (C4, pawn, white).

Now let's consider the move cxd4 - the pawn moved, so like before we remove (C3, pawn, white) and add (D4, pawn, white). But also the rook got removed from the board, so we have to remove (D4, rook, black) too. This is still less work than recreating the inputs from scratch!

A simple NNUE network
We will use our "A" feature set from the previous paragraph, so we have 768 inputs. The layers for the purpose of this illustration will be the 3 linear layers, 768->8, 8->8, 8->1. All layers are linear, and all hidden neurons use ClippedReLU activation function. The image below illustrates the architecture:

A[768]->8->8->1 architecture diagram

The flow is from the left to the right. The first layer is a large fully connected layer with 768 inputs, but only a small fraction of them is non-zero for each position - sparse matrix-vector multiplication can be utilized. Hidden layers are much smaller and always computed with dense matrix-vector multiplication. At the end, we get 1 output, which is usually trained to be the centipawn evaluation of the position (or proportional to it).

Consideration of networks size and cost.
Choosing the right architecture is tricky as it's an accuracy/performance trade-off. Large networks provide more accurate evaluation, but the speed impact might completely negate the gains in real play. For example Stockfish slowly transitioned from 256x2->32->32->1 to 1024x2->8->32->1.

Feature set
When choosing a feature set it might be tempting to go into complicated domain-specific knowledge, but the costs associated make simpler solutions more attractive. HalfKP, explained in detail later, is very simple, fast, and good enough. More sophisticated feature sets have been tried but they usually cannot combat the hit on performance. HalfKP features are easy to calculate, and change little from position to position.

Size also has to be considered. For the 256x2->32->32->1 architecture HalfKP inputs require about 10 million parameters in the first layer, which amounts to 20MB after quantization. For some users it might not be an issue to have a very large set of features, with possibly hundreds of millions of parameters, but for a typical user it's inconvenient. Moreover, increasing the feature set size may reduce the training speed for some implementations, and certainly will require more time to converge.

First set of hidden neurons
The number of outputs in the first layer is the most crucial parameter, and also has the highest impact on speed and size. The costs associated with this parameter are two-fold. For one, it increases the number of operations required when updating the accumulator. Second, for optimized implementations, one must consider the number of available registers - in Stockfish going past 256 neurons requires multiple passes over the feature indices as AVX2 doesn't have enough registers. It also partially determines the size of the first dense linear layer, which also greatly contributes to the total cost.

Further layers
Unlike in typical networks considered in machine learning here most of the knowledge is stored in the first layer, and because of that adding further small layers near the output adds little to accuracy, and may even be harmful if quantization is employed due to error accumulation. NNUE networks are kept unusually shallow, and keeping the size of the later layers small increases performance.

Accumulator
Even though we observed that few inputs change from position to position we have yet to take advantage of that. Recall that a linear layer is just adding some weight matrix columns together. Instead of recomputing the first set of hidden neurons for each position we can keep them as part of the position's state, and update it on each move based on what features (columns) were added or removed! We have to handle only two simple cases:

the feature i was removed from the input (1 -> 0) - subtract column i of the weight matrix from the accumulator
the feature i was added to the input (0 -> 1) - add column i of the weight matrix to the accumulator
For a single move, it's trivial to find which "A" features changed - we know what piece we're moving, from where, and where to. Captures and promotions can be considered as a piece disappearing or appearing from nowhere.

However, care must be taken when using floating point values. Repeatedly adding and subtracting floats results in error that accumulates with each move. It requires careful evaluation of whether the error is small enough for the net to still produce good results. Thankfully, it is best implemented such that the accumulator is not updated when undoing a move. Instead, it is simply stored on the search stack, so the error is bounded by O(MAX_DEPTH) and can mostly be ignored.

When using quantization this is no longer a problem, the incremental implementation is consistent, but now there is a possibility of overflowing the accumulator (regardless of whether incremental updates are used or not). The quantization scheme must be chosen such that no combination of possible active features can exceed the maximum value.

HalfKP
HalfKP is the most common feature set and other successful ones build on top of it. It fits in a sweet spot of being just the right size, and requiring very few updates per move on average. Each feature is a tuple (our_king_square, piece_square, piece_type, piece_color), where piece_type is not a king (in HalfKA feature set kings are included). This means that for each king position there is a set of features P, which are (piece_square, piece_type, piece_color). This allows the net to better understand the pieces in relation to the king. The total number of features is 64*64*5*2=40960. (Note that there is a leftover from Shogi in the current Stockfish implementation and there are 64 additional features that are unused, but we will disregard them in this document). The feature index can be calculated as


p_idx = piece_type * 2 + piece_color
halfkp_idx = piece_square + (p_idx + king_square * 10) * 64
The one special case that needs to be handled is when the king moves, because it is tied to all the features. All features are changed, so an accumulator refresh is executed. This makes king moves more costly but on average it still keeps the number of updates per evaluation low.

Now, you might ask, "but which king?!". The answer is both...

Multiple perspectives, multiple accumulators
This is where we need to start accounting for the features of both sides separately. The white side will keep its own accumulator, and the black side its own accumulator too. Effectively, it means that the maximum active number of features is twice as high as for a simple feature set with only one perspective. There will be twice as many updates and the accumulator will be twice as large in total, but overall this tradeoff between speed and accuracy is worth it. This approach inevitably creates some problems, options, and choices with regard to the exact model topology. Let's go through them one by one.

How to combine multiple accumulator perspectives?
Since we now have two accumulators, we need to somehow combine them into one vector that gets passed further into the network. This can be solved in two (three) ways. Let's denote the accumulator for white as A_w, and the accumulator for black as A_b.

concatenate the A_w and A_b, placing A_w first and A_b second. This is the simplest option. The output in this case is always relative to the white's perspective.
concatenate the A_w and A_b, placing A_w first if it's white to move, otherwise A_b first, and the other accumulator second. This approach has the advantage that the net can learn tempo. It now knows whose turn it is, which is an important factor in chess and can have a huge impact on evaluation of some positions. The output in this case is always relative to the side to move perspective.
Either 1 or 2, but instead of concatenating interleave. So A_w[0], A_b[0], A_w[1], A_b[1], .... This might be advantageous in some exotic architectures where not always the whole combined accumulator is used, in which case interleaving means that the slice used always contains the same number of outputs from white's and from black's perspectives. This might become useful, for example when employing structured sparsity to the first hidden layer, which ultimately works on the subset of the accumulator.
Which set of weights to use for each perspective?
So we compute the features for white and black the same, are their weights related? They can be, but it's not required. Engines differ in the handling of this.

Same weights for both perspectives. This means the board state needs to somehow be oriented. Otherwise white king on E1 would produce a different subset of features than a black king on E8, and white king on G4 would produce the same subset of features as a black king on G4. That's bad. The solution is to mirror the position and swap the color of the pieces for black's perspective; then the piece placement to feature mapping is logical for both. White king on E1 from white's perspective should be the same as a black king on E8 from black's perspective. Now you may think that flip is the way to go, but while chess has vertical symmetry, Shogi has rotational symmetry. The initial implementation of HalfKP in Stockfish used rotation to change the perspective, which is arguably incorrect for chess, but it worked surprisingly well.
Different weights for different perspectives. Is the white king on E1 actually equal to black king on E8? What about other pieces? Arguably one plays the game differently as black compared to as white, and it seems it makes sense to use different features for these perspectives. This is how some engines do it, and there's nothing wrong with this. The only downsides are larger size and slightly longer training time, but other than that it might even be better! It also completely removes the discussion about flip or rotate, which makes the implementation simpler.
HalfKP example and network diagram
Similar to the diagram above for the "A" feature set, here is the diagram for the same network but with HalfKP feature set, with combined weights. With a change that both accumulators are of size 4, so the network is in the end HalfKP[40960]->4x2->8->1

Let's look at the same example position as before: 1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1.



Now we have two perspectives, and will list the features for both of them separately. Remember the features are (our_king_square, piece_square, piece_type, piece_color) and we use flip to orient the squares for black and the colors are reversed! (One can think of the "color" as "us" or "them")

White's perspective: (A1, C3, pawn, white), (A1, D4, rook, black)

Blacks's perspective: (B1, C6, pawn, black), (B1, D5, rook, white)

The network diagram looks more interesting now.

HalfKP[40960]->4x2->8->1

Forward pass implementation
In this part, we will look at model inference as it could be implemented in a simple chess engine. We will work with floating point values for simplicity here. Input generation is outside of the scope of this implementation.

Example network
We will take a more generally defined network, with architecture FeatureSet[N]->M*2->K->1. The layers will therefore be:

L_0: Linear N->M
C_0: Clipped ReLU of size M*2
L_1: Linear M*2->K
C_1: Clipped ReLU of size K
L_2: Linear K->1
Layer parameters
Linear layers have 2 parameters - weights and biases. We will refer to them as L_0.weight and L_0.bias respectively. The layers also contain the number of inputs and outputs, in L_0.num_inputs and L_0.num_outputs respectively.

Here something important has to be said about the layout of the weight matrix. For sparse multiplication, the column-major (a column is contiguous in memory) layout is favorable, as we're adding columns, but for dense multiplication this is not so clear and a row-major layout may be preferable. For now we will stick to the column-major layout, but we may revisit the row-major one when it comes to quantization and optimization. For now, we assume L_0.weight allows access to the individual elements in the following form: L_0.weight[column_index][row_index].

The code presented is very close to C++ but technicalities might be omitted.

Accumulator
The accumulator can be represented by an array that is stored along other position state information on the search stack.


struct NnueAccumulator {
    // Two vectors of size M. v[0] for white's, and v[1] for black's perspectives.
    float v[2][M];

    // This will be utilised in later code snippets to make the access less verbose
    float* operator[](Color perspective) {
        return v[perspective];
    }
};
The accumulator can either be updated lazily on evaluation, or on each move. It doesn't matter here, but it has to be updated somehow. Whether it's better to update lazily or eagerly depends on the number of evaluations done during search. For updates, there are two cases, as laid out before:

The accumulator has to be recomputed from scratch.
The previous accumulator is reused and just updated with changed features
Refreshing the accumulator

void refresh_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // storage for the result
    const std::vector<int>& active_features,  // the indices of features that are active for this position
    Color                   perspective       // the perspective to refresh
) {
    // First we copy the layer bias, that's our starting point
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = layer.bias[i];
    }

    // Then we just accumulate all the columns for the active features. That's what accumulators do!
    for (int a : active_features) {
        for (int i = 0; i < M; ++i) {
            new_acc[perspective][i] += layer.weight[a][i];
        }
    }
}
Updating the accumulator

void update_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // it's nice to have already provided storage for
                                              // the new accumulator. Relevant parts will be overwritten
    const NNueAccumulator&  prev_acc,         // the previous accumulator, the one we're reusing
    const std::vector<int>& removed_features, // the indices of features that were removed
    const std::vector<int>& added_features,   // the indices of features that were added
    Color                   perspective       // the perspective to update, remember we have two,
                                              // they have separate feature lists, and it even may happen
                                              // that one is updated while the other needs a full refresh
) {
    // First we copy the previous values, that's our starting point
    for (int i = 0; i < M; ++i) {
        new_acc[perspective][i] = prev_acc[perspective][i];
    }

    // Then we subtract the weights of the removed features
    for (int r : removed_features) {
        for (int i = 0; i < M; ++i) {
            // Just subtract r-th column
            new_acc[perspective][i] -= layer.weight[r][i];
        }
    }

    // Similar for the added features, but add instead of subtracting
    for (int a : added_features) {
        for (int i = 0; i < M; ++i) {
            new_acc[perspective][i] += layer.weight[a][i];
        }
    }
}
And that's it! Pretty simple, isn't it?

Linear layer
This is simple vector-matrix multiplication, what could be complicated about it you ask? Nothing for now, but it will get complicated once optimization starts. Right now we won't optimize, but we will at least write a version that uses the fact that the weight matrix has a column-major layout.


float* linear(
    const LinearLayer& layer,  // the layer to use. We have two: L_1, L_2
    float*             output, // the already allocated storage for the result
    const float*       input   // the input, which is the output of the previous ClippedReLU layer
) {
    // First copy the biases to the output. We will be adding columns on top of it.
    for (int i = 0; i < layer.num_outputs; ++i) {
        output[i] = layer.bias[i];
    }

    // Remember that rainbowy diagram long time ago? This is it.
    // We're adding columns one by one, scaled by the input values.
    for (int i = 0; i < layer.num_inputs; ++i) {
        for (int j = 0; j < layer.num_outputs; ++j) {
            output[j] += input[i] * layer.weight[i][j];
        }
    }

    // Let the caller know where the used buffer ends.
    return output + layer.num_outputs;
}
ClippedReLU

float* crelu(,
    int          size,   // no need to have any layer structure, we just need the number of elements
    float*       output, // the already allocated storage for the result
    const float* input   // the input, which is the output of the previous linear layer
) {
    for (int i = 0; i < size; ++i) {
        output[i] = min(max(input[i], 0), 1);
    }

    return output + size;
}
Putting it together
In a crude pseudo code. The feature index generation is left as an exercise for the reader.


void Position::do_move(...) {
    ... // do the movey stuff

    for (Color perspective : { WHITE, BLACK }) {
        if (needs_refresh[perspective]) {
            refresh_accumulator(
                L_0,
                this->accumulator,
                this->get_active_features(perspective),
                perspective
            );
        } else {
            update_accumulator(
                L_0,
                this->accumulator,
                this->get_previous_position()->accumulator,
                this->get_removed_features(perspective),
                this->get_added_features(perspective),
                perspective
            );
        }
    }
}

float nnue_evaluate(const Position& pos) {
    float buffer[...]; // allocate enough space for the results

    // We need to prepare the input first! We will put the accumulator for
    // the side to move first, and the other second.
    float input[2*M];
    Color stm = pos.side_to_move;
    for (int i = 0; i < M; ++i) {
        input[  i] = pos.accumulator[ stm][i];
        input[M+i] = pos.accumulator[!stm][i];
    }

    float* curr_output = buffer;
    float* curr_input = input;
    float* next_output;

    // Evaluate one layer and move both input and output forward.
    // The last output becomes the next input.
    next_output = crelu(2 * L_0.num_outputs, curr_output, curr_input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = linear(L_1, curr_output, curr_input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = crelu(L_1.num_outputs, curr_output, curr_input);
    curr_input = curr_output;
    curr_output = next_output;

    next_output = linear(L_2, curr_output, curr_input);

    // We're done. The last layer should have put 1 value out under *curr_output.
    return *curr_output;
}
And that's it! That's the whole network. What do you mean you can't use it?! OH RIGHT, you don't have a net trained, what a bummer.

Training a net with pytorch
This will be very brief, as this is on the nnue-pytorch repo after all so you can just look up the code! We will not explain how pytorch works, but we will, however, explain some of the basics, and the quirks needed to accommodate this exotic use case.

Let's continue using the architecture from the forward pass implementation.

Model specification
Pytorch has built-in types for linear layers, so defining the model is pretty simple.


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()

        self.ft = nn.Linear(NUM_FEATURES, M)
        self.l1 = nn.Linear(2 * M, N)
        self.l2 = nn.Linear(N, K)

    # The inputs are a whole batch!
    # `stm` indicates whether white is the side to move. 1 = true, 0 = false.
    def forward(self, white_features, black_features, stm):
        w = self.ft(white_features) # white's perspective
        b = self.ft(black_features) # black's perspective

        # Remember that we order the accumulators for 2 perspectives based on who is to move.
        # So we blend two possible orderings by interpolating between `stm` and `1-stm` tensors.
        accumulator = (stm * torch.cat([w, b], dim=1)) + ((1 - stm) * torch.cat([b, w], dim=1))

        # Run the linear layers and use clamp_ as ClippedReLU
        l1_x = torch.clamp(accumulator, 0.0, 1.0)
        l2_x = torch.clamp(self.l1(l1_x), 0.0, 1.0)
        return self.l2(l2_x)
Thankfully, Pytorch handles backpropagation automatically through automatic differentiation. Neat! The hard bit now is, maybe surprisingly, feeding the data.

Preparing the inputs
There are two main bottlenecks in this part.

Parsing the training data sets
Preparing the tensor inputs
Parsing the training data sets and moving them to the python side
You might be tempted to implement this in python. It would work, but sadly, it would be orders of magnitude too slow. What we did in nnue-pytorch is we created a shared library in C++ that implements a very fast training data parser, and provides the data in a form that can be quickly turned into the input tensors.

We will use Ctypes for interoperation between C and Python. Seer's trainer uses pybind11 for example if you want more examples. In practice, anything that provides a way to pass pointers and call C functions from Python will work. Other languages can be used too, but keep in mind that only C has a stable ABI, which makes things easier and more portable. So for example, if you want to use C++ (like we will here) it's important to mark exported functions as extern "C".

The data reader is passed a file on creation, and then it spawns the requested number of worker threads that chew through the data and prepare whole batches asynchronously. The batches are then passed to the python side and turned into PyTorch tensors. Going one sample at a time by one is not a viable option, corners need to be cut by producing whole batches. You may ask why. PyTorch can turn multiple tensors into a batch so what's the problem? Let's see...

Remember how the input is sparse? Now let's say our batch size is 8192. What would happen if we sent 8192 sparse tensors and tried to form a batch from them? Well, pytorch doesn't like doing that by itself, we need to help it. And the best way is to form one big 2D sparse input tensor that encompasses the whole batch. It has 2 sparse dimensions and the indices are (position_index, feature_index), pretty simple, has great performance, and no need to create temporary tensors! The fact that we're forming whole batches from the start also means that we can reduce the number of allocations and use a better memory layout for the batch parts.

Because of that we also cannot simply use the PyTorch's DataLoader, instead we need to use it as a mere wrapper. But this effort is worth it. One worker thread can usually saturate even a high-end GPU without any issues.

Training batch structure and communication
The minimum that's needed are the features (from both perspectives), the side to move (for accumulator slice ordering), and the position evaluation (the score). Let's see how we can represent such a batch.


struct SparseBatch {
    SparseBatch(const std::vector<TrainingDataEntry>& entries) {

        // The number of positions in the batch
        size = entries.size();

        // The total number of white/black active features in the whole batch.
        num_active_white_features = 0;
        num_active_black_features = 0;

        // The side to move for each position. 1 for white, 0 for black.
        // Required for ordering the accumulator slices in the forward pass.
        stm = new float[size];

        // The score for each position. This is the value that we will be teaching the network.
        score = new float[size];

        // The indices of the active features.
        // Why is the size * 2?! The answer is that the indices are 2 dimensional
        // (position_index, feature_index). It's effectively a matrix of size
        // (num_active_*_features, 2).
        // IMPORTANT: We must make sure that the indices are in ascending order.
        // That is first comes the first position, then second, then third,
        // and so on. And within features for one position the feature indices
        // are also in ascending order. Why this is needed will be apparent later.
        white_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];
        black_features_indices = new int[size * MAX_ACTIVE_FEATURES * 2];

        fill(entries);
    }

    void fill(const std::vector<TrainingDataEntry>& entries) {
        ...
    }

    int size;
    int num_active_white_features;
    int num_active_black_features;

    float* stm;
    float* score;
    int* white_features_indices;
    int* black_features_indices;

    ~SparseBatch()
    {
        // RAII! Or use std::unique_ptr<T[]>, but remember that only raw pointers should
        // be passed through language boundaries as std::unique_ptr doesn't have stable ABI
        delete[] stm;
        delete[] score;
        delete[] white_features_indices;
        delete[] black_features_indices;
    }
};
and in python


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('size', ctypes.c_int),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('stm', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('white_features_indices', ctypes.POINTER(ctypes.c_int)),
        ('black_features_indices', ctypes.POINTER(ctypes.c_int))
    ]

    def get_tensors(self, device):
        # This is illustrative. In reality you might need to transfer these
        # to the GPU. You can also do it asynchronously, but remember to make
        # sure the source lives long enough for the copy to finish.
        # See torch.tensor.to(...) for more info.

        # This is a nice way to convert a pointer to a pytorch tensor.
        # Shape needs to be passed, remember we're forming the whole batch, the first
        # dimension is always the batch size.
        stm_t = torch.from_numpy(
            np.ctypeslib.as_array(self.stm, shape=(self.size, 1)))
        score_t = torch.from_numpy(
            np.ctypeslib.as_array(self.score, shape=(self.size, 1)))

        # As we said, the index tensor needs to be transposed (not the whole sparse tensor!).
        # This is just how pytorch stores indices in sparse tensors.
        # It also requires the indices to be 64-bit ints.
        white_features_indices_t = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.white_features_indices, shape=(self.num_active_white_features, 2))
            ), 0, 1).long()
        black_features_indices_t = torch.transpose(
            torch.from_numpy(
                np.ctypeslib.as_array(self.black_features_indices, shape=(self.num_active_white_features, 2))
            ), 0, 1).long()

        # The values are all ones, so we can create these tensors in place easily.
        # No need to go through a copy.
        white_features_values_t = torch.ones(self.num_active_white_features)
        black_features_values_t = torch.ones(self.num_active_black_features)

        # Now the magic. We construct a sparse tensor by giving the indices of
        # non-zero values (active feature indices) and the values themselves (all ones!).
        # The size of the tensor is batch_size*NUM_FEATURES, which would
        # normally be insanely large, but since the density is ~0.1% it takes
        # very little space and allows for faster forward pass.
        # For maximum performance we do cheat somewhat though. Normally pytorch
        # checks the correctness, which is an expensive O(n) operation.
        # By using _sparse_coo_tensor_unsafe we avoid that.
        white_features_t = torch._sparse_coo_tensor_unsafe(
            white_features_indices_t, white_features_values_t, (self.size, NUM_FEATURES))
        black_features_t = torch._sparse_coo_tensor_unsafe(
            black_features_indices_t, black_features_values_t, (self.size, NUM_FEATURES))

        # What is coalescing?! It makes sure the indices are unique and ordered.
        # Now you probably see why we said the inputs must be ordered from the start.
        # This is normally a O(n log n) operation and takes a significant amount of
        # time. But here we **know** that the tensor is already in a coalesced form,
        # therefore we can just tell pytorch that it can use that assumption.
        white_features_t._coalesced_(True)
        black_features_t._coalesced_(True)

        # Now this is what the forward() required!
        return white_features_t, black_features_t, stm_t, score_t

# Let's also tell ctypes how to understand this type.
SparseBatchPtr = ctypes.POINTER(SparseBatch)
Feature factorization
Let's focus on the features again. We will take a closer look at the HalfKP feature set. Recall, that HalfKP features are indexed by tuples of form (king_square, piece_square, piece_type, piece_color), where piece_type != KING.

The HalfKP feature set was formed by specialization of the P feature set for every single king square on the board. This in turn increased the feature set size, and caused the accesses to become much more sparse. This sparsity directly impacts how much each feature is seen during training, and that negatively impacts the learning of weights.

Feature factorization effectively, and efficiently, relates features together during training, so that more features are affected during each step of training. This is particularly important during early stages of training, because it results in even the rarest of feature weights being populated quickly with reasonable values.

Feature factorization works by introducing a "virtual" feature set (as opposed to the "real" feature set, here HalfKP) that contains denser features, each being directly related to (and, importantly, redundant with) one or more "real" features. These "virtual" features are present only during the training process, and will learn the common factor for all "real" features they relate to. Let's see how it works in case of HalfKP.

HalfKP is just P taken 64 times, once for each king square, as mentioned previously. Each P feature is therefore related to 64 HalfKP features, and will learn the common factor for a (piece_square, piece_type, piece_color) feature for all possible king positions.

Because "virtual" features are redundant with the "real" features their weights can be coalesced into the "real" features weights after the training is finished. The way to coalesce them follows from the computation performed in the network layer (the feature transformer).

Virtual feature coalescing
So how can we coalesce them? Let's look at how matrix and vector multiplication is done again. Consider the example position from before (1k6/8/8/8/3r4/2P5/8/K7 w - - 0 1).

:

Let's focus on the feature (A1, C3, pawn, white). Now, we're also gonna add the corresponding P feature (C3, pawn, white). What happens when the input goes through the first layer?


accumulator += weights[(A1, C3, pawn, white)];
accumulator += weights[(C3, pawn, white)];
which is equivalent to


accumulator += weights[(A1, C3, pawn, white)] + weights[(C3, pawn, white)];
So the relation is very simple. We just need to add the weights of each P feature to all the related HalfKP feature weights!

Other factors
Sometimes it's possible to add even more factors. It should be noted, however, that just adding more factors doesn't necessarily improve the training and may even cause it to regress. In general, whether using some factors helps or not depends on the training setup and the net being trained. It's always good to experiment with this stuff. With that said, however, we can consider for example the following factors for HalfKP.

"K" factors
The king position, 64 features. This one requires some careful handling as a single position has this feature multiple times - equal to the number of pieces on the board. This virtual feature set is needed purely because with HalfKP the king position feature is not encoded anywhere. HalfKA doesn't need it for example because it specifically has the feature for the king's position. In general, handling this is tricky, it may even require reducing the gradient for these features (otherwise the gradient is input*weight, but input is large compared to others).

"HalfRelativeKP" factors
In HalfKP we use the absolute piece position, but what if we encoded the position as relative to the king? There are 15x15 such relative positions possible, and most of them correspond 1:many to some HalfKP feature. The HalfRelativeKP feature index could be calculated for example like this:


int get_half_relative_kp_index(Color perspective, Square king_sq, Square piece_sq, Piece piece)
{
    const int p_idx = static_cast<int>(piece.type()) * 2 + (piece.color() != perspective);
    const Square oriented_king_sq = orient_flip(perspective, king_sq);
    const Square oriented_piece_sq = orient_flip(perspective, piece_sq);
    // The file/rank difference is always in range -7..7, and we need to map it to 0..15
    const int relative_file = oriented_piece_sq.file() - oriented_king_sq.file() + 7;
    const int relative_rank = oriented_piece_sq.rank() - oriented_king_sq.rank() + 7;
    return (p_idx * 15 * 15) + (relative_file * 15) + relative_rank;
}
Real effect of the factorizer
While the factorizer helps the net to generalize, it seems to only be relevant in the early stages, that is when the net doesn't really know anything yet. It accelerates the early stages of training and reduces the sparsity of the input (some inputs are very rare otherwise). But it quickly becomes unimportant and in later stages of the training can be removed to gain some training speed (after all it can add a lot of active features).

Loss functions and how to apply them
The Goal
Training a network is really just minimizing a loss function, which needs to be smooth and have a minimum at the "optimal" evaluation (the training target). For the purpose of NNUE, this is done by gradient descent through usual machine learning methods (there are also non-gradient methods that are not described here).

Converting the evaluation from CP-space to WDL-space
By CP-space we mean the centipawn scale (or something proportional, like engine's internal units). By WDL-space we mean 0=loss, 0.5=draw, 1=win.

It's of course possible to apply the loss function directly on the evaluation value (in CP-space), but this can lead to large gradients (or a lot of hyperparameter tuning), restricts the set of loss functions available, and doesn't allow using results for loss. We will focus on evaluation in WDL-space. But how to convert between these spaces? Usually, the evaluation to performance correspondence can be well-fitted by a sigmoid. For example, in some data generated by Stockfish we have:



so in the code we may do the following:


scaling_factor = 410 # this depends on the engine, and maybe even on the data
wdl_space_eval = torch.sigmoid(cp_space_eval / scaling_factor)
This transformation also has the nice effect that large evaluations become "closer" together, which aligns well with the real play, where large evaluations don't need to be that precise.

Using results along the evaluation
With the values for which we will compute loss being in WDL-space, we may now interpolate them with game results. We will introduce a lambda_ parameter that governs the interpolation.


# game_result is in WDL-space
wdl_value = lambda_ * wdl_space_eval + (1 - lambda_) * game_result
The interpolation can also be applied to the loss.


loss_eval = ... # loss between model eval and position eval
loss_result = ... # loss between model eval and game result
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
Which way works better depends on your case 😃

Mean Squared Error (MSE)
Now we know what we're trying to fit; let's look at how we will fit them.

This is a very simple loss function that just takes a square of the difference between the predicted value and the target. This results in a nice linear gradient.

With interpolation applied before:


scaling = ... # depends on the engine and data. Determines the shape of
              # the sigmoid that transforms the evaluation to WDL space
              # Stockfish uses values around 400
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)
wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result
loss = (wdl_eval_model - wdl_value_target)**2
With interpolation applied after:


scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)
loss_eval   = (wdl_eval_model - wdl_eval_target)**2
loss_result = (wdl_eval_model - game_result)**2
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
Note: in practice, the exponent can be >2. Higher exponents give more weight towards precision at a cost of accuracy. Stockfish networks had good training results with an exponent of 2.6 for example.

loss


grad


Cross entropy
This loss function is usually used for continuous classification problems, and our use case could be considered one.

Care must be taken around domain boundaries. Usually, a very small value (epsilon) is added such that the values never reach 0 under the logarithm.

With interpolation applied before:


epsilon = 1e-12 # to prevent log(0)
scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)
wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result

# The first term in the loss has 0 gradient, because we always
# differentiate with respect to `wdl_eval_model`, but it makes the loss nice
# in the sense that 0 is the minimum.
loss = (wdl_value_target * log(wdl_value_target + epsilon) + (1 - wdl_value_target) * log(1 - wdl_value_target + epsilon))
      -(wdl_value_target * log(wdl_eval_model   + epsilon) + (1 - wdl_value_target) * log(1 - wdl_eval_model   + epsilon))
With interpolation applied after:


epsilon = 1e-12 # to prevent log(0)
scaling = ...
wdl_eval_model = sigmoid(model(...) / scaling)
wdl_eval_target = sigmoid(target / scaling)

# The first term in the loss has 0 gradient, because we always
# differentiate with respect to `wdl_eval_model`, but it makes the loss nice
# in the sense that 0 is the minimum.
loss_eval   = (wdl_eval_target * log(wdl_eval_target + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_target + epsilon))
             -(wdl_eval_target * log(wdl_eval_model  + epsilon) + (1 - wdl_eval_target) * log(1 - wdl_eval_model  + epsilon))
loss_result = (game_result     * log(wdl_eval_target + epsilon) + (1 - game_result)     * log(1 - wdl_eval_target + epsilon))
             -(game_result     * log(wdl_eval_model  + epsilon) + (1 - game_result)     * log(1 - wdl_eval_model  + epsilon))
loss = lambda_ * loss_eval + (1 - lambda_) * loss_result
loss


grad


Quantization
At the start of this document, it was briefly mentioned what quantization is and that it will be important. Now it's time to understand it properly. The goal is that we want to use the smallest possible integers everywhere. Most CPU architectures provide instructions that can work on 8, 16, 32, or even 64 int8 values at a time, and we should take advantage of that. That means we need to use int8 values, with range -128..127, for weights and inputs; or int16, with range -32768..32767, where int8 is not possible.

Coming up with the right quantization scheme is not easy, so first we'll present the one currently used by Stockfish, and then we'll explain how to get there, how to code it, and finally how to optimize it.

Stockfish quantization scheme
Feature Transformer
Let's start with the feature transformer. Recall that its purpose is to accumulate between 0 to 30 (for HalfKP) rows of weights. We want to have int8 values as inputs to the later layers, with the activation range (ClippedReLU) being 0..127, but that means that using int8 integers for the accumulator doesn't provide enough space as the values would go beyond the range of int8 before applying the ClippedReLU... so we use int16 for the accumulator and then convert to int8 when doing the ClippedReLU.

Linear layer
We wanted int8 inputs and we can get them without losing too much precision. The nature of matrix-purposed SIMD instructions is that, thankfully, the accumulation happens in int32. So we don't experience the same issue as in the feature transformer where we're manually adding rows, and we can utilize the int8 multiplication with int32 accumulation to the fullest extent, and only later go back to int8 in the ClippedReLU layer. We will add the biases after the accumulation has happened, so they should be stored in int32.

ClippedReLU
Nothing special going on in here. Since the inputs are not being scaled, this is simply the same operation but in a different domain. Instead of clamping to 0..1 we clamp to 0..127. The input type is usually different than the output type as inputs will be either int32 or int16, and the output we want is int8. The values won't change but the conversion needs to be applied.

The math of quantization and how to make it fit
To quantize the network we need to multiply the weights and biases by some constant to translate them to a different range of values. This poses a problem when confronted with multiplication during network inference - (a*x) * (a*w) = a*a*x*w, and we have to sometimes scale back the outputs too. But each layer is still independent so let's go through them one by one again.

Feature Transformer
Remember we want our activation range to change from 0..1 to 0..127. Since the feature transformer is a purely additive process, it's enough that we multiply the weights and biases by 127. Both weights and biases are stored as int16. We could divide the output by some factor a to get more precision, in which case the weights and biases would have to be multiplied by a*127 instead, but in practice, it increases the accuracy only by a little bit.

Linear layer
To arrive at int8 weights we have to apply some scaling factor. This scaling factor ultimately depends on how much precision needs to be preserved, but cannot be too large because then the weights will be limited in magnitude. For example, if we took the scaling factor to be 64 (used in Stockfish), then the maximum weight in the floating point space is 127/64=1.984375. This is enough to have good nets, but care needs to be taken to clamp the weights during training so that they don't go outside that range. The scaling factor of 64 can also be understood as the smallest weight step that can be represented being 1/64=0.015625.

A linear layer is just matrix multiplication, so we're multiplying inputs and weights, but now both are scaled relative to the float version. Let's denote the input scaling factor (activation range scaling) as s_A, and the weight scaling factor by s_W. x is the unquantized input, w is the unquantized weight, 'b' is the unquantized bias, and y is the unquantized output. So we have:


x * w + b = y
((s_A * x) * (s_W * w)) + (b * s_A * s_W) = (y * s_A) * s_W
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) / s_W = (y * s_A)
From that we learn that we need to scale the bias by (s_A * s_W), weights by s_W, and divide output by s_W to get the desired (y * s_A), which is correctly scaled to the activation range.

Now, this applies only when the next layer is the ClippedReLU layer. For the last layer, the output range is very different and the quantization will also be different. In Stockfish we want the last layer to output values in range -10000..10000 while still keeping int8 weights. This can be achieved without any additional scaling factors, but it's easiest to do and understand with an additional scaling factor.

We'll introduce a new scaling factor, s_O. This scaling factor, unlike others, needs to be applied to the output both during training (for loss calculation against the actual evaluation) and inference. The purpose of it is to scale the float output of the network to match the range of the integer evaluation used by Stockfish. Basically, it means that 1 in the float space is equal to s_O internal evaluation units. It has an additional advantage that it allows us to have the layer weights be similar in magnitude to the previous layers.

So the math is now:


x * w + b = y
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O = ((y * s_A) * s_W) * s_O
(((s_A * x) * (s_W * w)) + (b * s_A * s_W)) * s_O / s_A / s_W = (y * s_O)
(((s_A * x) * (s_W / s_A * w)) + (b * s_A * s_W / s_A)) * s_O / s_W = (y * s_O)
(((s_A * x) * (s_W * s_O / s_A * w)) + (b * s_W * s_O)) / s_W = (y * s_O)
From that we learn that we need to scale the bias by s_W * s_O, weights by s_W * s_O / s_A, and divide the output by s_W to get the desired (y * s_O).

Implementation
For the unoptimized implementation, not much changes. One just has to remember to change the data types to integers with desired size, scale weights on input, and divide the output from linear layers by s_W. s_W is usually chosen to be a power of two, so that this operation is a simple bitwise right shift, as there are no SIMD division instructions for integers and even if there were it would be slow.

Optimized implementation
For simplicity, we will focus on optimization only for the AVX2 extension of the x86-64 instruction set.

Feature Transformer
The benefit of SIMD for the feature transformer is two-fold:

multiple additions per instruction can be performed
large total register size means we don't need to write to memory as often
Our accumulation structure doesn't change much, we just change float to int16:


// We now also make sure that the accumulator structure is aligned to the cache line.
// This is not strictly required by AVX2 instructions but may improve performance.
struct alignas(64) NnueAccumulator {
    // Two vectors of size N. v[0] for white's, and v[1] for black's perspectives.
    int16_t v[2][N];

    // This will be utilised in later code snippets to make the access less verbose
    int16_t* operator[](Color perspective) {
        return v[perspective];
    }
};
Now let's look at the refresh function. For simplicity, we will assume that there are enough registers so that spills don't happen, but in reality (M > 256) it is required to do multiple passes over the active features, each time considering a part of the accumulator only. A single AVX2 register can fit 16 int16 values and there are 16 AVX2 registers (32 since AVX-512).


void refresh_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // storage for the result
    const std::vector<int>& active_features,  // the indices of features that are active for this position
    Color                   perspective       // the perspective to refresh
) {
    // The compiler should use one register per value, and hopefully
    // won't spill anything. Always check the assembly generated to be sure!
    constexpr int register_width = 256 / 16;
    static_assert(M % register_width == 0, "We're processing 16 elements at a time");
    constexpr int num_chunks = M / register_width;
    __m256i regs[num_chunks];

    // Load bias to registers and operate on registers only.
    for (int i = 0; i < num_chunks; ++i) {
        regs[i] = _mm256_load_si256(&layer.bias[i * register_width]);
    }

    for (int a : active_features) {
        for (int i = 0; i < num_chunks; ++i) {
            // Now we do 1 memory operation instead of 2 per loop iteration.
            regs[i] = _mm256_add_epi16(regs[i], _mm256_load_si256(&layer.weight[a][i * register_width]));
        }
    }

    // Only after all the accumulation is done do the write.
    for (int i = 0; i < num_chunks; ++i) {
        _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]);
    }
}
similarly for the update:


void update_accumulator(
    const LinearLayer&      layer,            // this will always be L_0
    NnueAccumulator&        new_acc,          // it's nice to have already provided storage for
                                              // the new accumulator. Relevant parts will be overwritten
    const NNueAccumulator&  prev_acc,         // the previous accumulator, the one we're reusing
    const std::vector<int>& removed_features, // the indices of features that were removed
    const std::vector<int>& added_features,   // the indices of features that were added
    Color                   perspective       // the perspective to update, remember we have two,
                                              // they have separate feature lists, and it even may happen
                                              // that one is updated while the other needs a full refresh
) {
    // The compiler should use one register per value, and hopefully
    // won't spill anything. Always check the assembly generated to be sure!
    constexpr int register_width = 256 / 16;
    static_assert(M % register_width == 0, "We're processing 16 elements at a time");
    constexpr int num_chunks = M / register_width;
    __m256i regs[num_chunks];

    // Load the previous values to registers and operate on registers only.
    for (int i = 0; i < num_chunks; ++i) {
        regs[i] = _mm256_load_si256(&prev_acc[perspective][i * register_width]);
    }

    // Then we subtract the weights of the removed features
    for (int r : removed_features) {
        for (int i = 0; i < num_chunks; ++i) {
            regs[i] = _mm256_sub_epi16(regs[i], _mm256_load_si256(&layer.weight[r][i * register_width]));
        }
    }

    // Similar for the added features, but add instead of subtracting
    for (int a : added_features) {
        for (int i = 0; i < num_chunks; ++i) {
            regs[i] = _mm256_add_epi16(regs[i], _mm256_load_si256(&layer.weight[a][i * register_width]));
        }
    }

    // Only after all the accumulation is done do the write.
    for (int i = 0; i < num_chunks; ++i) {
        _mm256_store_si256(&new_acc[perspective][i * register_width], regs[i]);
    }
}
Linear layer
Matrix multiplication is hard to optimize in general, and there are many approaches depending on the size of the matrices. Since we expect the layers to be small, we will not delve into any fancy blocked algorithms. And just rely on manual unrolling and trying to process multiple values at a time. This is not optimal, but it's simple and very close. We will only describe the case where the number of outputs is divisible by 4. The output layer has 1 output but it's also very small and doesn't require anything clever. We will also require the input size to be a multiple of 32, otherwise adding 0 padding is required.


int32_t* linear(
    const LinearLayer& layer,  // the layer to use. We have two: L_1, L_2
    int32_t*           output, // the already allocated storage for the result
    const int8_t*      input   // the input, which is the output of the previous ClippedReLU layer
) {
    constexpr int register_width = 256 / 8;
    assert(layer.num_inputs % register_width == 0, "We're processing 32 elements at a time");
    assert(layer.num_outputs % 4 == 0, "We unroll by 4");
    const int num_in_chunks = layer.num_inputs / register_width;
    const int num_out_chunks = layer.num_outputs / 4;

    for (int i = 0; i < num_out_chunks; ++i) {
        // Prepare weight offsets. One offset for one row of weights.
        // This is a simple index into a 2D array.
        const int offset0 = (i * 4 + 0) * layer.num_inputs;
        const int offset1 = (i * 4 + 1) * layer.num_inputs;
        const int offset2 = (i * 4 + 2) * layer.num_inputs;
        const int offset3 = (i * 4 + 3) * layer.num_inputs;

        // Accumulation starts from 0, we add the bias only at the end.
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
        for (int j = 0; j < num_in_chunks; ++j) {
            // We unroll by 4 so that we can reuse this value, reducing the number of
            // memory operations required.
            const __m256i in = _mm256_load_si256(&input[j * register_width]);

            // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
            // For definition see below.
            m256_add_dpbusd_epi32(sum0, in, _mm256_load_si256(&layer.weights[offset0 + j * register_width]));
            m256_add_dpbusd_epi32(sum1, in, _mm256_load_si256(&layer.weights[offset1 + j * register_width]));
            m256_add_dpbusd_epi32(sum2, in, _mm256_load_si256(&layer.weights[offset2 + j * register_width]));
            m256_add_dpbusd_epi32(sum3, in, _mm256_load_si256(&layer.weights[offset3 + j * register_width]));
        }

        const __m128i bias = _mm_load_si128(&layer.bias[i * 4]);
        // This function adds horizontally 8 values from each sum together, producing 4 int32 values.
        // For the definition see below.
        __m128i outval = m256_haddx4(sum0, sum1, sum2, sum3, bias);
        // Here we account for the weights scaling.
        outval = _mm_srai_epi32(outval, log2_weight_scale);
        _mm_store_si128(&output[i * 4], outval);
    }

    return output + layer.num_outputs;
}
m256_add_dpbusd_epi32


The output needs to be horizontally accumulated further, but it's faster to do it with 4 sums (sum0, sum1, sum2, sum3) later.

This function can benefit from VNNI extension, here controlled by USE_VNNI.


void m256_add_dpbusd_epi32(__m256i& acc, __m256i a, __m256i b) {
#if defined (USE_VNNI)

    // This does exactly the same thing as explained below but in one instruction.
    acc = _mm256_dpbusd_epi32(acc, a, b);

#else

    // Multiply a * b and accumulate neighbouring outputs into int16 values
    __m256i product0 = _mm256_maddubs_epi16(a, b);

    // Multiply product0 by 1 (idempotent) and accumulate neighbouring outputs into int32 values
    __m256i one = _mm256_set1_epi16(1);
    product0 = _mm256_madd_epi16(product0, one);

    // Add to the main int32 accumulator.
    acc = _mm256_add_epi32(acc, product0);

#endif
};