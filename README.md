# Cerebros AutoML

![assets/0-cerebros-logo.png](assets/0-cerebros-logo.png)

The Cerebros package is an ultra-precise Neural Architecture Search (NAS) / AutoML that is intended to much more closely mimic biological neurons than conventional Multi Layer Perceptron based neural network architecture search strategies.

## Cerebros Community Edition and Cerebros Enterprise

This branch is a Cerebros Research and Development branch that may not have the same structure, efficiency or certain features as the Enterprise branch. Some features from here, such as case study examples, will be migrated to the Community branch in due time.  

For a robust managed neural architecture search experience hosted on Google Cloud Platform and supported by our SLA, we recommend Cerebros Enterprise, our commercial version. Soon you will be able to sign up and immediately start using it at `https://www.cerebros.one`. In the meantime, we can set up your own Cerebros managed neural architecture search pipeline for you with a one business day turnaround. We offer consulting, demos, full service machine learning service and can provision you with your own full neural architecture search pipeline complete with automated Bayesian hyperparameter search. Contact David Thrower:`david@cerebros.one` or call us at (US country code 1) `(650) 789-4375`. Additionally, we can complete machine learning tasks for your organization. Give us a call.

## In summary what is it and what is different:

A biological brain looks like this:

![assets/brain.png](assets/brain.png)

Multi layer perceptrons look like this:

![assets/mpl.png](assets/mlp.png)

If the goal of MLPs was to mimic how a biological neuron works, why do we still build neural networks that are structurally similar to the first prototypes from 1989? At the time, it was the closest we could get, but both hardware and software have changed since.

In a biological brain, neurons connect in a multi-dimensional lattice of vertical and lateral connections, which may repeat. Why don't we try to mimic this? In recent years, we got a step closer to this by using single skip connections, but why not simply randomize the connectivity to numerous levels in the network's structure altogether and add lateral connections that overlap like a biological brain? (We presume God knew what He was doing, so why re-invent the wheel.)

That is what we did here. We built a neural architecture search that connects Dense layers in this manner.

What if we made a multi-layer perceptron that looks like this: (Green triangles are Keras Input layers. Blue Squares are Keras Concatenate layers. The Pink stretched ovals are Keras Dense layers. The one stretched red oval is the network's Output layer. It is presumed that there is a Batch Normalization layer between each Concatenate layer and the Dense layer it feeds into.)

![assets/Brain-lookalike1.png](assets/Brain-lookalike1.png)

... or what if we made one like this:

![assets/Brain-lookalike2.png](assets/Brain-lookalike2.png)

and like this

![assets/Neuron-lookalike6.png](assets/Neuron-lookalike6.png)

What if we made a single-layer perceptron that looks like this:

![assets/Neuron-lookalike1.png](assets/Neuron-lookalike1.png)

The deeper technical details can be found here:

![documentation/cerebros-technical-details.md](documentation/cerebros-technical-details.md)

## Use examples

Use examples are available in `use-examples`, while documentation is available in `documentation/examples`. 

Clone the repo

`git clone https://github.com/david-thrower/cerebros-core-algorithm-alpha.git`

cd into it: `cd cerebros-core-algorithm-alpha`

install all required packages: `pip3 install -r requirements.txt`

cd into it: `test-cases/ames-housing-price-prediction`

Run the Ames housing data example:

`python3 ames_housing_pred.py`

You can also access a Jupyter notebook version of it as `ames_housing_pred.ipynb`

## Summary of Results

- Ames housing data set, not pre-processed or scaled, non-numerical columns dropped:
- House sell price predictions, val_rmse $169.04592895507812.
- The mean sale price in the data was $180,796.06.
- Val set RMSE was 0.0935% of the mean sale price. In other words, on average, the model predicted the sale price accurate to less than 0.1% of the actual sale price. Yes, you are reading it right. Less than 1/10 of a percent off on average.
- There was no pre-trained base model used. The data in [ames.csv](ames.csv) which was selected for training is the only data any of the model's weights have ever seen.

For further details, see ![documentation/examples/use-example-detailed.md](documentation/examples/use-example-detailed.md)

# Documentation

![documentation/api-docs/summary-and-navigation.md](documentation/api-docs/summary-and-navigation.md)

## Open source license:

[license.md](license.md)

**License terms may be amended at any time as deemed necessary at Cerebros sole discretion.**

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
4. The mechanism by which Cerebros works, gives it an ability to deduce and extrapolate intermediate variables which are not in your training data. This is in theory how it is able to make such accurate predictions in data sets which seem to not have enough features to make such accurate predictions. With this said, care should be taken to avoid including proxy variables that can be used to extract variables which are unethical to consider in decision making in your use case. An example would be an insurance company including a variable closely correlated with race and or disability status, such as residential postal code in a model development task which will be used to build models that determine insurance premium pricing. This is unethical, and using Cerebros or any derivative work to facilitate such is prohibited and will be litigated without notice or opportunity to voluntarily settle, if discovered by Cerebros maintainers.    
5. Furthermore, an association however strong it may be does not imply causality, nor implies that it is ethical to apply the knowledge of such association in your business case. You are encouraged to use as conservative of judgment as possible in such, and if necessary consulting with the right subject matter experts to assist in making these determinations. Failure to do so is a violation of the license agreement.   
