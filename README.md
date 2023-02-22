# Cerebros AutoML

![assets/0-cerebros-logo.png](assets/0-cerebros-logo.png)

The Cerebros package is an ultra-precise Neural Architecture Search (NAS) / AutoML that is intended to much more closely mimic biological neurons than conventional Multi Layer Perceptron based neural network architecture search strategies.

## Cerebros Community Edition and Cerebros Enterprise

The Cerebros community edition provides an open-source minimum viable single parameter set NAS and also also provides an example manifest for an exhaustive Neural Architecture Search to run on Kubeflow/Katib. This is licensed for free use provided that the use is consistent with the ethical use provisions in the license described at the bottom of this page. You can easily reproduce this with the Jupyter notebook in the directory `/kubeflow-pipeline`, using the Kale Jupyter notebook extension. For a robust managed neural architecture search experience hosted on Google Cloud Platform and supported by our SLA, we recommend Cerebros Enterprise, our commercial version. Soon you will be able to sign up and immediately start using it at `https://www.cerebros.one`. In the meantime, we can set up your own Cerbros managed neural architecture search pipeline for you with a one business day turnaround. We offer consulting, demos, full service machine learning service and can provision you with your own full neural architecture search pipeline complete with automated Bayesian hyperparameter search. Contact David Thrower:`david@cerebros.one` or call us at (US country code 1) `(650) 789-4375`. Additionally, we can complete machine learning tasks for your organization. Give us a call.



## In summary what is it and what is different:

A biological brain looks like this:

![assets/brain.png](assets/brain.png)

Multi layer perceptrons look like this:

![assets/mpl.png](assets/mlp.png)

If the goal of MLPs was to mimic how a biological neuron works, why do we still build neural networks that are structurally similar to the first prototypes from 1989? At the time, it was the closest we could get, but both hardware and software have changed since.

In a biological brain, neurons connect in a multi-dimensional lattice of vertical and lateral connections, which may repeat. Why don't we try to mimic this? In recent years, we got a step closer to this by using single skip connections, but why not simply randomize the connectivity to numerous levels in the network's structure altogether and add lateral connections that overlap like a biological brain? (We presume God knew what He was doing, so why re-invent the wheel.)

That is what we did here. We built a neural architecture search that connects Dense layers in this manner.

What if we made a multi-layer pereceptron that looks like this: (Green triangles are Keras Input layers. Blue Squares are Keras Concatenate layers. The Pink stretched ovals are Keras Dense layers. The one stretched red oval is the network's Output layer. It is presumed that there is a batch normaliation layer between each Concatenate layer and the Dense layer it feeds into.)

![assets/Brain-lookalike1.png](assets/Brain-lookalike1.png)

... or what if we made one like this:

![assets/Brain-lookalike2.png](assets/Brain-lookalike2.png)

and like this

![assets/Neuron-lookalike6.png](assets/Neuron-lookalike6.png)

What if we made a single-layer perceptron that looks like this:

![assets/Neuron-lookalike1.png](assets/Neuron-lookalike1.png)

The deeper technical details can be found here:

![documentation/cerebros-technical-details.md](documentation/cerebros-technical-details.md)

## Use example: Try it for yourself:

shell:

Clone the repo
`git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git`

cd into it
`cd cerebros-core-algorithm-alpha`

install all required packages
```
pip3 install -r requirements.txt
```
Run the Ames housing data example:
```
python3 regression-example-ames-no-preproc.py
```

## Example output from this task:

```
... # lots of summaries of training trials
...

Best result this trial was: 169.04592895507812
Type of best result: <class 'float'>
Best model name: 2023_01_12_23_42_cerebros_auto_ml_test_meta_0/models/tr_0000000000000006_subtrial_0000000000000000
Best model: (May need to re-initialize weights, and retrain with early stopping callback)
Model: "NeuralNetworkFuture_0000000000000nan_tr_6_nn_materialized"

```

## Summary of Results

- Ames housing data set, not pre-processed or scaled:
- House sell price predictions, val_rmse $169.04592895507812.
- The mean sale price in the data was $180,796.06.
- Val set RMSE was 0.0935% the mean sale price. Yes, you are reading it right. Less than 0.1% off on average.
- There was no pre-trained base model. The data in [ames.csv](ames.csv) is the only data any of the model's weights has ever seen.


Run the Neural Architecture Search and get results back.
```python3
result = cerebros.run_random_search()

print("Best model: (May need to re-initialize weights, and retrain with early stopping callback)")
best_model_found = cerebros.get_best_model()
print(best_model_found.summary())

print("result extracted from cerebros")
print(f"Final result was (val_root_mean_squared_error): {result}")

```

For further details, see ![documentation/examples/use-example-detailed.md](documentation/examples/use-example-detailed.md)



## Documentation

Classes: (Meant for direct use)

```
SimpleCerebrosRandomSearch
    - Args:
                unit_type: Unit,
                                  The type of units.units.___ object that the
                                  in the body of the neural networks created
                                  will consist of. Usually will be a
                                  DenseUnit object.
                 input_shapes: list,
                                  List of tuples representing the shape
                                  of input(s) to the network.
                 minimum_levels: int,
                                  Basically the minimum deprh of the neural
                                  networks to be tried.
                 maximum_levels: int,
                                  Basically the maximum deprh of the neural
                                  networks to be tried.
        21-add-a-way-to-get-the-best-model
                 minimum_neurons_per_unit: int,
                                  This is the mininmum number of neurons each
                                  Dense unit (Dense layer) nested in any Level
                                  may consist of.
                 maximum_neurons_per_unit: int,
                                  This is the mininmum number of neurons each
                                  Dense unit (Dense layer) nested in any Level
                                  may consist of.
                 activation: str: defaults to 'elu',
                                  keras activation for dense unit
                 number_of_architecture_moities_to_try: int: defaults to 7,
                                  The auto-ml will select this many
                                  permutations of architectural moities, in
                                  other words (number of levels, number of
                                  Units or Dense Layers per level for each
                                  level.
                 minimum_skip_connection_depth: int: defaults to 1,
                                  The DenseLevels will be randomly connected
                                  to others in Levels below it. These
                                  connections will not be made exclusively to
                                  DenseUnits in its immediate successor Level.
                                  Some may be 2,3,4,5, ... Levels down. This
                                  parameter controls the minimum depth of
                                  connection allowed for connections not at
                                  the immediate successor Level.
                 maximum_skip_connection_depth: int: defaults to 7,
                                  The DenseLevels will be randomly connected
                                  to others in Levels below it. These
                                  connections will not be made exclusively to
                                  DenseUnits in its immediate successor Level.
                                  Some may be 2,3,4,5, ... Levels down. This
                                  parameter controls the minimum depth of
                                  connection allowed for connections not at
                                  the immediate successor Level.
                 predecessor_level_connection_affinity_factor_first: int: defaults to 5,
                                  The number of randomly selected Input Unit
                                  connestions which the first Dense Level's
                                  Units will make to randomly selected Input
                                  units per input unit. In other words if
                                  there are n input units, each first Level
                                  Dense unit will select predecessor_level_connection_affinity_factor_first * n
                                  InputUnits to connect to (WITH REPLACEMENT).
                                  Hence this may be > 1, as a given
                                  InputUnit may be chosen multiple times.
                                  For example, with this set to 5, and where
                                  were 2 connections, then each Dense
                                  unit will make 10 connections to randomly
                                  selected Input Units. If there were
                                  10 inputs, and this were set to 0.5, then
                                  the each Dense unit on the first layer would
                                  connect to 5 randomly selected InputUnits.
                 predecessor_level_connection_affinity_factor_first_rounding_rule: str: defaults to 'ceil',
                                  Since the numnber of randomly selected
                                  connections to an InputUnit that a given
                                  Dense unit on the first layer makes is
                                  [number of InputUnits] *  predecessor_level_connection_affinity_factor_first,
                                  then this will usually come out to a floating
                                  point number. The number of upstream
                                  connections to make is a discrete variable,
                                  so this float must be cast as an integer.
                                  This supports "floor" and "ceil" as the
                                  options with "ceil" as the recommended
                                  option, because it can never ruturn 0,
                                  which would throw and error
                                  (It would create a disjointed graph).
                 predecessor_level_connection_affinity_factor_main: float: defaults to 0.7,
                                  For the second and subsequent DenseLevels,
                                  if its immediate predecessor level has
                                  n DenseUnits, the DenseUnits on this layer will randomly select
                                  predecessor_level_connection_affinity_factor_main * n
                                  DenseUnits form its immediate predecessor
                                  Level to connect to. The selection is
                                  WITH REPLACEMENT and a given DenseUnit may
                                  be selected multiple times.
                 predecessor_level_connection_affinity_factor_main_rounding_rule: str: defaults to 'ceil',
                                 The number of connections calculated by
                                 predecessor_level_connection_affinity_factor_main * n
                                 will usually come out to a floating point
   21-add-a-way-to-get-the-best-model                              the number of DenseUnits  in its predecessor
                                 Level to connect to is a discrete variable,
                                 hence this must be cast to an integer. The
                                 options to do this are "floor" and "ceil",
                                 with "ceil" being the generally recommended
                                 option, as it will never return 0. 0 will
                                 create a disjointed graph and throw an error.
                 predecessor_level_connection_affinity_factor_decay_main: object: defaults to zero_7_exp_decay,
                                 It is likely that the connections with the
                                 imemdiate predecessor, grandparent, and
                                 great - grandparent predecessors will be more
                                 (or less) infuential than more distant
                                 predecessors. This parameter allows any
                                 function that accepts an integer and returns
                                 a floating point or integer to be used to
                                 decrease (or increase) the number of
                                 connections made between the k th layer and
                                 its nthe predecessor, where n = 1 is its
                                 GRANDPARENT predecesor level, and n-2 is its great - grandparent level. What this does exactly: The kth layer will make
                                 predecessor_level_connection_affinity_factor_main
                 seed: int: defaults to 8675309,
                 max_consecutive_lateral_connections: int: defaults to 7,
                                 The maximum number of consecutive Units on a Level that make
                                 a lateral connection. Setting this too high can cause exploding /
                                 vanishing gradients and internal covariate shift. Setting this too
                                 low may miss some patterns the network may otherwise pick up.
                 gate_after_n_lateral_connections: int: defaults to 3,
                                 After this many kth consecutive lateral connecgtions, gate the
                                 output of the k + 1 th DenseUnit before creating a subsequent
                                 lateral connection.
                 gate_activation_function: object: defaults to simple_sigmoid,
                                 Which activation function to gate the output of one DenseUnit
                                 before making a lateral connection.
                 p_lateral_connection: float: defaults to 0.97,
                                 The probability of the first given DenseUnit on a level making a
                                 lateral connection with the second DenseUnit.
                 p_lateral_connection_decay: object: defaults to zero_95_exp_decay,
                                 A function that descreases or increases the probability of a
                                 lateral connection being made with a subsequent DenseUnit.
                                 Accepts an unsigned integer x. returns a floating point number that
                                 will be multiplied by p_lateral_connection where x is the number
                                 of subsequent connections after the first
                num_lateral_connection_tries_per_unit: int: defaults to 1
                learning_rate: Keras API optimizer parameter
                loss: Keras API parameter
                metrics: Keras API parameter
                epochs: int: Keras API parameter
                patience: int Keras API parameter for early stopping callback
                project_name: str: An arbitrary name for the projet. Must be a vaild POSIX file name.
                model_graphs='model_graphs': str: prefix for model graph image files
                batch_size: int: Keras param
                meta_trial_number: int: Arbitraty trial number that adds uniqueness to files created suring distributed tuning of tunable hyperparams.

    Methods:
        run_random_search:
            Runs a random search over the parametrs chosen and returns the best metric found.

            Params:
                None
            Returns:
                int: Best metric
        get_best_model:
            Params:
                None
            Returns:
                keras.Model: Best model found.

```


## How this all works:

We start with some basic structural components:

The SimpleCerebrosRandomSearch: The core auto-ML that recursively creates the neural networks, vicariously through the NeuralNetworkFuture object:

NeuralNetworkFuture:
  - A data structure that is essentially a wrapper around the whole neural network system.
  - You will note the word "Future" in the name of this data structure. This is for a reason. The solution to the problem of recursively parsing neural networks having topologies similar to the connectivity between biological neurons involves a chicken before the egg problem. Specifically this was that randomly assigning neural connectivity will create some errors and disjointed graphs, which once created is impractical correct without starting over. Additionally, I have found that for some reason, graphs having some dead ends, but not completely disjointed train very slowly.
  - The probability of a given graph passing on the first try is very low, especially if you are building a complex network, nonetheless, the randomized vertical and lateral, and potentially repeating connections are the key to making this all work.
  - The solution was to create a Futures objects that first planned how many levels of Dense layers would exist in the network, how many Dense layers each level would consist of, and how many neurons each Dense layer would consist of, allowing the random connections to be tentatively selected, but not materialized. Then it applies a protocol to detect and resolve any inconsistencies, disjointed connections, etc in the planned conductivities before any actual neural network components are actually materialized.
  - A list of errors is compiled and another protocol appends the planned connectivity with additional connections to fix the breaks in the connectivity.
  - Lastly, once the network's connectivity has been validated, it then materializes the Dense layers of the neural network per the conductivities planned, resulting in a neural network ready to be trained.

Level:
  - A data structure adding a new layer of abstraction above the concept of a Dense layer, which is a wrapper consisting of multiple instances of a future for what we historically called a Dense layer in a neural network. A level will consist of multiple Dense Units, which each will materialize to a Dense Layer. Since we are making both vertical and lateral connections, the term "Layer" loses relevance it has in the traditional sequential MLP context. A NeuralNetworkFuture has many Levels, and a Level belongs to a NeuralNetworkFuture. A Level has many Units, and a Unit belongs to a Level.

Unit:
  - A data structure which is a future for a single Dense layer. A Level has many Units, and a Unit belongs to a Level.

![assets/Cerebros.png](assets/Cerebros.png)

Here are the steps to the process:

0. Some nomenclature:
  1. k refers to the Level number being immediately discussed.
  2. l refers to the number of DenseUnits the kth Level has.
  3. k-1 refers to the immediate predecessor Level (Parent Level) number of the kth level of the level being discussed.
  4. n refers to the number of DenseUnits the kth Level's parent predecessor has.
1. SimpleCerebrosRandomSearch.run_random_search().
    1. This calls SimpleCerebrosRandomSearch.parse_neural_network_structural_spec_random() which chooses the following random unsigned integers:
        1. How many Levels, the archtecture will consist of;
        2. For each Level, how many Units the levl will consist of;
        3. For each unit, how many neurons the Dense layer it will materialize will consist of.
        4. This is parsed into a dictionary as a high-level specification for the nodes, but not edges called a neural_network_spec.
        5. This will instantiate a NeuralNetworkFuture of the selected specification for number of Levels, Units per Level, and neurons per Unit, and the NeuralNetworkFuture takes the neural_network_spec as an argument.
        6. This entire logic will repeat once for each number in range of: number_of_architecture_moities_to_try.
        7. Step 5 will repea multiple times, for each same neural_network_spec, once for each number in range: number_of_tries_per_architecture_moity.
        8. All replictions are done as separate Python multiprocessing proces (multiple workers in parallel on separate processor cores).
2. **(This is a top down operation starting with InputLevel and proceeding to the last hidden layer in the network)** In each NeuralNetworkFuture, the neural_network_spec will be iterated through, instantiating a DenseLevel object for each element in the dictionary , which will be passed as the argument level_prototype. Each will be linked to the last, and each will maintain access to the same chain of Levels as the list predecessor_levels (The whole thing is essentially like a list, having many necessary nested elements).
3. A dictionary of possible predecessor connections will also be parsed. This is a symbolic representation of the levels and units above it that is faster to iterate through than the actual Levels and Units objects themselves.
4. SimpleCerebrosRandomSearch calls each NeuralNetworkFuture.materialize() method which calls NeuralNetworkFuture.parse_units(), method, which recursively calls the .parse_units() belonging to each Levels object. Within each Levels object, this will iterate through its level_prototype list and will instantiate a DenseUnits object for each item and append DenseLevel.parallel_units with it.
5. NeuralNetworkFuture.materialize() calls each NeuralNetworkFuture object's NeuralNetworkFuture.set_connectivity_future_prototype() method. ... **(This is a bottom - up operation starting with the last hidden layer and and proceeding to the InputLevel)** This will recursively call each Levels object's determine_upstream_random_connectivity(). Which will trigger each layer to recursively call each of its constituent DenseUnit objects' set_connectivity_future_prototype() method. Each DenseUnit will calculate how many connections to make to DenseUnits in each predecessor Level and will select that many units from its possible_predecessor_connections dictionary, from each of those levels, appending each to its __predecessor_connections_future list.
7. Once the random predecessor connections have been selected, now, the system will then need to validate the DOWNSTREAM network connectivity of each Dense unit and repair any breaks that would cause a disjointed graph.  (verify that each DenseUnit has at least one connection to a SUCCESSOR Layer's DenseUnit). Here is why: If the random connections were chosen bottom - up, (which is the lesser of two evils) AND each DenseUnit will always select at least one PREDECESSOR connection (I ether validate this to throw an error if the number of connections to the immediate predecessor layer is less than 1 OR it just coerce it to 1 if 0 is calculated, with this said, then upstream connectivity is not possible to break, however, it is inherently possible that at least one predecessor unit was NOT selected by any of its successor's randomly selected connections. (especially if a low value is selected for the predecessor_level_connection_affinity_factor_main), something, I speculate may be advantageous to do, as some research has indicate that making sparse connections can outperform Dense connections, which is what we aim to do. We want not all connections to be made to the immediate successor Level. We want some to connect 2 Levels downstream, 3, ...  4, ... 5, Levels downstream ... The trouble is that the random connections that facilitate this can "leave out" a DenseUnit in a predecessor Level as not being picked at all by any DenseUnit in a SUCCESSOR level, leaving us with a dead end in the network, hence a disjointed graph. There are 3 rules this has to follow:
    1. **Rule 1:** Each DenseUnit must connect to SOMETHING upstream (PREDECESSORS) WITHIN max_skip_connection_depth layers of its immediate predecessor. (Random bottom - up assignment can't accidentally violate this rule if it always selects at least one selection, so nothing to worry about or validate here).
    2. **Rule 2:** Everything must connect to something something DOWNSTREAM (successors) within max_skip_connection_depth Levels of its level_number. **(Random bottom - up assignment will frequently leave violations of this rule behind. Where this happens, these should either be connected to a randomly selected DenseUnit max_skip_connection_depth layers below | **or a randomly selected DenseUnit residing a randomly chosen number of layers below in the range of [minimum_skip_connection_depth, maximum_skip_connection_depth] below when possible | and if necessary, to the last hidden DenseLevel or the output level)**.
    3. Now the third rule **Rule 3:** The connectivity must flow in only one direction vertically and one direction laterally (on a layer - by - layer basis). In other words, a kth Level's DenseUnit can't take its own output as one of its inputs. Nor can a kth Level's DenseUnit take its k+[any number]th successor's output as one of its inputs (because it is also a function of the kth Level's DenseUnit's own output). This would obviously be a contradiction, like filling the tank of an empty fire truck using the fire hose that draws water from its own tank.. or an empty fuel pump at a fuel station filling its own empty tank using the hose that goes in a car's gas tank, ... and gets that fuel from its own tank... Fortunately, both the logic setting the vertical connectivity and the logic setting the lateral connectivity both can't create this inconsistency, so there is nothing to worry about or validate here either. We only have to screen for and fix violations of rule 2, "Every DenseUnit must connect to some (DOWNSTREAM / SUCCESSOR) DenseUnit within max_skip_connection_depth of itself. Here is the logic for this validation and rectification:   
8. The test to determine whether there are violations of rule 2 desctibed above is done by DenseLevel objects:
  1. Scenario 1: **(If the kth DenseLayer is the last Layer)**:
    1. For each layer having a  **layer_number >= k - maximum_skip_connection_depth** (look at possible_predecessor_connections):
    2. For each DenseUnit in said layer, check each successor layers' Dense units' __predecessor_connections_future list for whether the DenseUnit making the check has been selected. If at least one connection is found: The check passes, else, the kth layer will add itself to the __predecessor_connections_future belonging to a successor meeting the constraint in 7.1.1.
    3. Scenario 2: **(If the kth DenseLayer is not the last Layer)** Do the same as scenario 1, EXCEPT, only check the layer where **its layer number == (k - maximum_skip_connection_depth) for a pre-existing selection for connection.**.    
9. Lateral connectivity: For each DesnseLayer:
  7.1 For each DenseUnit:
  7.2 Calculate the number of lateral connections to make.
  7.3 Randomly select a number of units from level_prototype where unit is > the the current unit's unit id and the connection doesn't violate the max_consecutive_lateral_connections setting. If there has been > gate_after_n_lateral_connections, then apply the gating function and restart the count for this rule. There is nothing to validate here. If this is set uo to only allow right OR left connections, then this can't create a disjointed graph or contradiction. Now all connectivities are planned.
10. Materialize Dense layers. **(This is a top down operation starting with InputLevel and proceeding to the last hidden layer in the network)**
11. Create output layer.
12.Compile model.
13. Fit the model, save the oracles, and save the model as a saved Keras model.
14. Iterate through the results and find best the model ad metrics.  


## Open source license:

[license.md](license.md)

**Licnse terms may be amended at any time as deemed necessry at Cerebros sole discretion.**

## Acknowledgements:

1. My Jennifer and my step-kids who have chosen to stay around and have rode out quite a storm because of my career in science.
2. My son Aidyn, daughter Jenna, and my collaborators Max Morganbesser and Andres Espinosa.
3. Mingxing Tan, Quoc V. Le for EfficientNet (recommeded image embedding base model).
4. My colleagues who I work with every day.
5. Tensorflow, Keras, Kubeflow, Kale, Optuna, Keras Tuner, and Ray open source communities and contributors.
6. Google Cloud Platform, Arikto, Canonical, and Paperspace and their support staff for the commercial compute and ML OPS platforms used.
7. Microk8s, minikube,and the core Kubernetes communities and associated projects.
8. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018. Base embedding usee for text classification tests.
9. Andrew Howard1, Mark Sandler1, Grace Chu1, Liang-Chieh Chen1, Bo Chen1, Mingxing Tan2, Weijun Wang1, Yukun Zhu1, Ruoming Pang2, Vijay Vasudevan2, Quoc V. Le2, Hartwig Ada MobileNet image embedding used for CICD tests.

## Legal disclaimers:

1. Cerebros is an independent initiative. Nothing published herein, nor any predictions made by models developed by the Cerebros algorithm should be construed as an opinion of any Cerebros maintainer or contributor or community member nor any of such community member's, clients, or employer, whether private companies, academic institutions, or government agencies.
2. Although Cerebros may produce astoundingly accurate models from a relatively minuscule amount of data as the example above depicts, past performance does not constitute a promise of similar results on your data set or even that such results would bear relevance in your business use case. Numerous variables will determine the outcome of your experiments and models used in production developed therefrom, including but not limited to:
    1. The characteristics, distribution, and scale of your data
    2. Sampling methods used
    3. How data was trained - test split (hint, if samples with identical data is a possibility, random selection is usually not the best way, hashing each sample then modulus division by a constant, and placing samples where the result of this is <= train set proportion, is better. This will force all occurrences of a given set of identical samples on the same side of the train, test split),
    4. Hyperparameter selection and tuning algorithm chosen
    5. Feature selection practices and features available in your use case
    6. Model drift, changes in the patterns in data, trends over time, climate change, social changes over time, evolution, etc.
3. Users are responsible for validating one's own models and the suitability for their use case. Cerebros does not make predictions. Cerebros parses neural networks (models) that your data will train, and these models will make predictions based on your data whether or not it is correct, sampled in a sensible way, or otherwise unbiased and useful. Cerebros does a partial validation, solely by metrics such as 'val_root_mean_squared_error'. This is a preliminary metric of how the model is performing, assuming numerous logical and ethical assumptions that only humans with subject matter expertise can validate (think spurious associations and correlations), in addition to statistical parameters such as valid sampling of the training data and that the distribution of the data is not skewed.
4. The mechanism by which Cerebros works, gives it an ability to deduce and extrapolate intermediate variables which are not in your training data. This is in theory how it is able to make such accurate predictions in data sets which seem to not have enough features to make such accurate predictions. With this said, care should be taken to avoid including proxy variables that can be used to extract variables which are unethical to consider in decision making in your use case. An example would be an insurance company including a variable closely correlated with race and or disability status, such as residential postal code in a model development task which will be used to build models that determine insurance premium pricing. This is unethical, and using Cerebros or any derivitive work to facilitate such is prohibited and will be litigated without notice or opportunity to voluntarily settle, if discovered by Cerebros maintainers.    
5. Furthermore, an association however strong it may be does not imply causality, nor implies that it is ethical to apply the knowledge of such association in your business case. You are encouraged to use as conservative of judgment as possible in such, and if necessary consulting with the right subject matter experts to assist in making these determinations. Failure to do so is a violation of the license agreement.   
