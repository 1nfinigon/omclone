```
$ cargo run
$ cargo run benchmark
$ cargo run check


$ export LIBTORCH_USE_PYTORCH=1
$ cargo run solver
$ cargo run train
```

# Done

-   Simple MCTS solver (without transpositions), choosing actions for a single
    arm at a time in order
-   ResNet network architecture and basic training loop, with win-probability
    and policy heads
    -   Input features use sparse tensors for evaluation and training, to save
        bandwidth/space
-   Seed-solver that plays games starting from a human "seed solution",
    following human moves until a few moves from the end, and then uses the MCTS
    solver to play out the rest
-   Training data generator from partially-seeded games, generating two types of 
    input: human and MCTS
    -    Initially I used human inputs to train policy and MCTS inputs to train
         win-rate. This did okay but ran into two problems: not enough diverse
         training data and overfitting to blind repetitions of arm patterns
         ("fard", "faar", see observations below)

# In progress

-   Generate synthetic random problems by iterating backwards
-   Probe/visualise the feature inputs and network to make sure everything's
    working as I expected

# Experiment 0 (2024-09-27): just playing around

-   Training data: epoch-v1-n seeded ~100-1000hum:1mcts, later w/ split loss
    weight
-   Optimizer: AdamW lr between 4e-5 and 3e-3
-   Batch size: 128
-   Model: model v1 7 layer ch=96 (~2000k params)

Running the seed solver training on samples derived from MCTS where we start the
search from only a small number (x) of moves from the end, then training a new
network on the result; rinse and repeat. Increasing x gradually as the network
gets better. No systematic analysis or probing, just manually black-box
inspecting solver output and generated solutions to gain a bit of intuition and
spot obvious issues.

Notes:

-   The network very quickly overspecializes to spam "Drop"
    -   This was just because I forgot to clear the input between timesteps,
        meaning the input contained very little usable data, which meant the
        best the network could do is to blindly spam the most likely instruction
    -   In response to this observation I also tweaked the loss-weight for
        samples that come from human solutions (which ~only sees policy loss) vs
        those that come from MCTS (which ~only sees value loss)
-   Training both value and policy on a mix of human solutions (weight) +
    weak seeded MCTS solutions, I attempted to weight the loss scale to
    significantly (x0.00001) shrink the impact of value loss for human
    solutions. This has caused the value head to become extremely sharp (always
    outputting 0 or 1, no in-between) for MCTS solutions, even though the
    contribution of the value loss to the overall loss is very small (0.006 vs
    2.0).
    -   Actually, why _is_ the value loss so low here? Is it just predicting
        "cycles left until cycle limit > some small number"? I think so...
    -   Proposed solution: Generate solutions that have higher cycles-left but
        still succeed, and lower cycles-left but still fail
    -   After I increased the cycle-window for seeded solutions, and started
        generating solutions that have higher cycles-left but still succeed, the
        value loss was still extremely small. Why??
    -   It's because I have 100x the number of human-solution samples than MCTS
        samples (for a given seed game, the ratio is currently approx 1k-2k
        human samples vs 10 MCTS samples)
    -   I could compensate for this by increasing the importance weighting by a
        factor of ~100x, but that effectively 100x's my learning rate (TODO:
        apparently this is only true for SGD optimizer, but not quite true for
        AdamW optimizer? Figure out why), which my previous rudimentary
        experiments around lr has already shown to completely throw off the
        training stability.
    -   So, I just need to generate more, and more diverse data...
        -   Right now I have 1 GPU for training and 1 GPU for game-playing; it's
            looking like I want a ratio of 1:10 instead. Which is also approx
            the ratio the KataGo paper used, interestingly.
        -   As for more diverse data... working on generating synthetic random
            problems by iterating backwards from the final state.
-   Starting to see forms of overfitting, from training on limited numbers of
    human solutions.  In a particular solution to OM2021_W0, an arm has learnt
    to "fard" repeatedly, and another "faar", all the way to the end, even
    though this isn't actually the shortest/most optimal sequence -- it's just
    learned behaviour from the fact that this instruction-count solution repeats
    those motifs a lot.
    -   Proposed solution: taper off the human training and generate more MCTS
        data

# Experiment 1 (2024-09-28): effect of varying #channels

-   Training/validation data:
    -   epoch-v1-1 (seeded ~100hum:1mcts)
    -   w/ split loss weight (hum -> policy loss only, mcts -> value loss only)
    -   831k samples from 2.3k games
    -   75/25 training/validation split
-   Optimizer: AdamW lr=4e-4
-   Batch size: 128
-   Model A: model96_20240928_112658 v1 7 layer ch=96 (2048k params)
-   Model B: model48_20240928_112943 v1 7 layer ch=48 (764k params)

Notes:

-   This training data set still suffers from value loss being small
    relative to policy loss, this time it's more like a factor of 100x
-   The randomly-initialized Model A had double the L2 penalty of Model B. This
    makes sense, given that L2 norm scales with # params (if each param is drawn
    from the same distribution to start with).
    -   It seems that this means L2 penalty loss (and hence the total loss) is
        not directly comparable between different-sized networks. The other loss
        terms do not scale with # params, and are directly comparable. Although
        with the current tiny loss coeff on L2 penalty this is a negligible
        effect.
    -   I think I want to conclude that Model B is _not_ being regularised
        less than A, and that this is just an initialization effect. Here are
        two different ways of thinking about this:
        -   Consider that Model B can be embedded entirely inside Model A, by
            making the other 48 channels in A "zero channels", which incur zero
            L2 loss. So, two models of different sizes can have "the same
            functionality" with the same L2 loss.
        -   Consider an extremely simplified model of a neural network layer, as
            a function R^n -> R^n which is a composition of a linear function L
            and a nonlinear function N. Assume the input vector is either zero
            or of unit length, and assume that N preserves this property
            property (this is not true for ReLU, but whatever). I think we'd
            like L to preserve this property as well, which means that L should
            be a rotation matrix. The Frobenius norm for a rotation matrix is n,
            the number of dimensions. However, we _can_ still reduce the norm
            whilst still preserving the property, by reducing the number of
            dimensions being rotated (and setting the other dimensions to zero)!
            (Such dimensionality reduction is akin to reducing the number of
            channels. Which is kinda what the previous bullet point was
            illustrating by way of explicit construction.) Handwaving even more,
            this suggests that the L2 penalty loss is pressuring the network
            overall to stray from the above property at most k*l2_loss_coeff
            times, for some constant k. And if we assume that the number of
            times to "stray" is a function of the problem to be solved (and how
            "badly behaved" the nonlinear functions are, summed across the whole
            of the network evaluation), then indeed the loss coeff should be
            fixed even as the number of channels changes (and we'd expect the
            L2 number itself to be comparable eventually as well, assuming the
            two networks are doing equivalent work). This is a horrible
            hand-wavey analogy but intuitively it feels to have somewhat
            satisfying explaining power.
        -   What about varying number of layers, rather than number of channels?
            For ResNets the answer is approximately the same as above. But
            interestingly, for non-ResNet architectures, it is _not_ possible to
            embed a shallow model inside a deep model whilst keeping L2 loss the
            same, as the extra layers still need identity matrices simply to
            propagate c channels of data (with Frobenius norm equal to c). So,
            for non-ResNets I think I do expect L2 norm to increase
            proportionally with depth, and I would want to adjust the L2 loss
            penalty coefficient accordingly (but, not when varying #channels)!
-   Both networks train at the same speed, indicating the python dataloader or
    disk->sparse->dense->GPU or loss-computation overheads dominate. The GPU is
    not hitting power usage limits (avg 100W for A and 50W for B), further
    indicating that some time spent optimizing pytorch/train.py would be
    fruitful.
-   I neglected to set a seed, so the random training/validation split is
    different, making validation results hard to compare between the two. This
    is fixed for future experiments.

# Experiment 2 (2024-09-28): effect of varying lr

-   Model: v1 7 layer ch=48 (764k params)
-   Run A: model48_20240928_112943
    -   LR = 4e-4
-   Run B: model48_20240928_200924
    -   LR = 8e-4
-   Other details same as experiment 1

Notes:

-   The L2 norm for run B grows ~2x as fast. Looks like same trajectory as
    experiment 1 run A, interestingly -- that run had double the channels but
    half the lr. Coincidence or no? Not sure.
-   Loss curve for both track identically trendwise (but higher variance) until
    I killed the run after 15 epochs (75k batches).
-   Not sure I learnt anything from this. Still TODO to learn how AdamW actually
    works / what lr rate means there.

# Experiment 3 (2024-09-29): AdamW vs SGD

-   Model: v1 7 layer ch=48 (764k params)
-   Run A: model48_20240928_112943
    -   Optimizer: AdamW lr=4e-4
-   Run B: model48_20240929_002831
    -   Optimizer: SGD lr=4e-4 momentum=0.9
-   Run C: model48_20240929_011754
    -   Optimizer: SGD lr=8e-4 momentum=0.9
-   Run D: model48_20240929_012537
    -   Optimizer: SGD lr=2e-5 momentum=0.9
-   Other details same as experiment 1

Notes:

-   Run B hit a plateau in loss very quickly, at batch 3k
-   Run C hit a plateau in loss even more quickly, at batch 2k
-   Run D loss never changed. I then tried again with only 32 samples, and even
    after several epochs the loss did not change.
-   Something's going wrong, needs more investigation.

# Experiment 4 (2024-09-29): gradient logging

-   Run A: model48_20240929_024745
    -   Optimizer: AdamW lr=4e-4
    -   Batch size: 128
-   Run B: model48_20240929_024923
    -   Optimizer: SGD lr=4e-4 momentum=0.9
    -   Batch size: 128
-   Run C: model48_20240929_025707
    -   Optimizer: SGD lr=4e-4 momentum=0.0
    -   Batch size: 16
-   Run D: model48_20240929_031121
    -   Optimizer: SGD lr=4e-5 momentum=0.0
    -   Batch size: 16
-   Run E: model48_20240929_032818
    -   Optimizer: SGD lr=4e-4 momentum=0.999
    -   Batch size: 128
-   Run F: model48_20240929_034011
    -   Optimizer: SGD lr=8e-4 momentum=0.999
    -   Batch size: 128
-   Run G: model48_20240929_040550
    -   Optimizer: SGD lr=2e-4 momentum=0.999
    -   Batch size: 128

Notes:

-   Run A is a rerun of experiment 1 run A, this time with gradient logging
-   SGD runs with 0.0 and 0.9 momentum failed to make any progress at all
-   The AdamW run had increasing L2 norm in weights immediately, but no change
    in L2 norm in the failed SGD runs. I think this points towards bad weight
    initializations.
    -   https://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    -   This suggests that 0.999 momentum is reasonable to try
-   TODO: rerun 0.9 momentum experiment after I have better weights init.
-   Run F clearly had too high of an lr rate; at batch 9k diverged to a spot
    with zero gradient and then couldn't get out (What would cause this?  Dead
    ReLU neuron? Not digging into this for now.)

# Future work

Technical/impl work:

-   Seed-solver: Generate solutions that have low cycles-left but still fail
    -   ...because the layout is bad
    -   ...because the layout is good but there's just not enough cycles left
-   Seed-solver: Randomly generate some training samples that have no history
-   Seed-solver: Evaluate and monitor
    -   proportion of successful solves (where it had to find _n_ moves,
        as a function of _n_?)
    -   how well does it solve with no history?
-   Train a timesteps-left head on existing solutions
-   Search: Parallelise MCTS, batch NN evaluations
-   Make net/eval/training board size agnostic
-   use symmetries to generate more data
-   Network/code for generating and estimating winrate for layouts
-   Solvers: Better logging, saving intermediate layer outputs / residuals
-   Policy network visualisation

Science-y investigation work:

-   Explicit weights init
-   Lean on existing solutions more, but be careful mixing them in with
    MCTS-generated solutions. They are from different domains; try having them
    train on non-overlapping outputs, to avoid the model overspecializing on
    "source"
-   weight and neuron activation visualization
    -   mean/stdev/activation-proportion per layer / per history
    -   max abs gradient
-   find/monitor dead neurons
-   keep more notes/observations for experiments, be more scientific about this
-   run same NN several times to get an idea of consistency
-   Batch size? https://arxiv.org/pdf/1812.06162

# idk

-   Does allowing action selection for any arm (rather than the next one in order)
    improve strength?
