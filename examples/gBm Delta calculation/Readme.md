# Data of Quantum-enhanced analysis of discrete stochastic processes

This is the data of the publication "Quantum-enhanced analysis of discrete 
stochastic processes" to be found at [NPJ QI](https://www.nature.com/articles/s41534-021-00459-2).

# Usage

In the following, we explain how to replicate our figures and findings.

## Figure 4
The idea is to use Donsker's invariance principle to use random walks to model the
geometric Brownian motion. Then the Delta is easily computed.

![](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41534-021-00459-2/MediaObjects/41534_2021_459_Fig4_HTML.png?as=webp)

### Left

The Delta of an European call option is calculated for various 
strike prices. The underlying asset is defined with
μ = 0, σ = 0.02, r = 0.02, S0 = 100, t = 1, 
and the time of maturity T = 10. The dotted black line is the true 
evaluation of the Delta. The black (blue) solid line is the real 
(imaginary) part of the theoretical Delta calculation obtained by a 
Fourier approximation with P = 100 and L = 100, which serves 
as the reference for the experimental validation. The black crosses 
(blue tri-downs) are the real (imaginary) part of the Delta calculated 
with the experiment on the IBM quantum computer with error mitigation 
applied. The black dots (blue triangles) are the simulation with 
noise model provided in qiskit with the same error mitigation applied. 

### Right

This plot shows the characteristic function calculated in
theory ( × ), by simulation with noise and error mitigation 
(dot) and the IBM quantum experiment with error mitigation (tri-down) 
for the example strike price of K = 110.

### Replication

The input data is encoded in `gBm_delta_data.py`:

    S_0 = 100
    r = 0.02
    time_evaluation = 1
    time_of_maturity = 10
    time_to_maturity = time_of_maturity - time_evaluation
    mu = 0.00
    sigma = 0.02
    K = np.round(S_0 * np.exp(r * time_to_maturity), decimals=4)

The experiment has several steps:

* Calculate the "true" delta by high-approximation Fourier method using
`gBm_delta_theory_fourier.py` with n=10, P=10000, L=10000.
*  Calculate the "approximation" delta by Fourier method using
`gBm_delta_theory_fourier.py` with n=4, P=100, L=100.
* Use the Quantum Computer to calculate the approximation (n=4, P=100, L=100)

The last step is technologically involved as it needs many round trips to the
quantum processor. This is done in the following way:

To start the calculation, use two methods:

1) The very slow, break-prone all-in-one script `gBm_delta_donsker_qc-simulation_fourier.py`
2) The very stable but hard to use automated-execution engine using the scripts
   1. To create the experiments (but *not* run them): `gBm_delta_donsker_qc-simulation_fourier_rest_start.py`
   2. To start/run the execution of the experiments: `gBm_delta_donsker_qc-simulation_fourier_rest_run.py`
   3. To gather result data from finished runs: `gBm_delta_donsker_qc-simulation_fourier_rest_done.py`
   4. To gather the details data: `gBm_delta_donsker_qc-simulation_fourier_detail_analysis.py`

The script (easy-to-learn but hard to carry-through) `gBm_delta_donsker_qc-simulation_fourier.py` creates 38
separate set of experiments for a separate strike-price. Each set of experiments consists of one `PreparedExperiment`
for the calculation and one `PreparedExperiment` for the error mitigation. Each such prepared experiments consists of
many `QasmQobj` that each consists of some qiskit `QasmQobjExperiment` that is to be executed on the 
quantum computer.

The more complicated version 2. uses a (to-be-published) "server" that handles the execution. It is a job manager. 

### Plotting

We have the experiment data stored here. One way to replicate the images, which are also part of this
example, is to execute the two scripts:

1) `gBm_delta_plot.py`
2) `gBm_delta_plot-details.py`

That's it, enjoy!
