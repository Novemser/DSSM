package strlet.transferLearning.inductive.supervised.boosting;

import java.util.Random;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.supervised.SingleSourceInstanceTransfer;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class TradaBoost extends SingleSourceInstanceTransfer implements
		Randomizable, WeightedInstancesHandler {

	private int seed = 1;
	private Classifier m_Classifier = new ZeroR();
	private int m_NumIterations = 10;

	/** a ZeroR model in case no model can be built from the data */
	protected Classifier m_ZeroR = null;

	/** Max num iterations tried to find classifier with non-zero error. */
	private static int MAX_NUM_RESAMPLING_ITERATIONS = 10;

	/** Array for storing the generated base classifiers. */
	protected Classifier[] m_Classifiers = null;

	/** Array for storing the weights for the votes. */
	protected double[] m_Betas = null;

	/** The number of successfully generated base classifiers. */
	private int m_IterationsPerformed = 0;

	@Override
	public void setSeed(int seed) {
		this.seed = seed;
	}

	@Override
	public int getSeed() {
		return seed;
	}

	/**
	 * Sets the number of bagging iterations
	 */
	public void setNumIterations(int numIterations) {
		m_NumIterations = numIterations;
	}

	/**
	 * Gets the number of bagging iterations
	 * 
	 * @return the maximum number of bagging iterations
	 */
	public int getNumIterations() {
		return m_NumIterations;
	}

	/**
	 * Set the base learner.
	 * 
	 * @param newClassifier
	 *            the classifier to use.
	 * @throws Exception
	 */
	public void setClassifier(Classifier newClassifier) throws Exception {

		m_Classifier = Classifier.makeCopy(newClassifier);
	}

	/**
	 * Get the classifier used as the base learner.
	 * 
	 * @return the classifier used as the classifier
	 */
	public Classifier getClassifier() {
		return m_Classifier;
	}

	@Override
	public void buildModel(Instances source, Instances target) throws Exception {
		testWithFail(source, target);
		source = new Instances(source);
		source.deleteWithMissingClass();
		target = new Instances(target);
		target.deleteWithMissingClass();

		// only class? -> build ZeroR model
		if (target.numAttributes() == 1) {
			System.err
					.println("Cannot build model (only class attribute present in data!), "
							+ "using ZeroR model instead!");
			m_ZeroR = new ZeroR();
			m_ZeroR.buildClassifier(target);
			return;
		}

		int sourceInstances = source.numInstances();
		if (sourceInstances < m_NumIterations)
			m_NumIterations = sourceInstances - 1;

		// m_NumClasses = target.numClasses();
		m_IterationsPerformed = 0;
		m_Betas = new double[m_NumIterations];
		m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);

		performBuild(source, target);

	}

	private void performBuild(Instances source, Instances target)
			throws Exception {

		Random rand = new Random(seed);
		double tmp = Math.log(source.numInstances() / (0.0 + m_NumIterations));
		double beta = 1 / (1 + Math.sqrt(2 * tmp));

		Class<Randomizable> rClass = Randomizable.class;
		if (rClass.isAssignableFrom(m_Classifier.getClass())) {
			for (int i = 0; i < m_Classifiers.length; ++i) {
				Classifier classifier = m_Classifiers[i];
				Randomizable randomizable = rClass.cast(classifier);
				randomizable.setSeed(rand.nextInt());
			}
		}

		// Do boostrap iterations
		for (m_IterationsPerformed = 0; m_IterationsPerformed < m_Classifiers.length; ++m_IterationsPerformed) {
			/*
			 * if (m_Debug) { System.err.println("Training classifier " +
			 * (m_IterationsPerformed + 1)); }
			 */
			Instances trainData = join(source, target);
			Classifier classifier = m_Classifiers[m_IterationsPerformed];

			double epsilon = 0;
			if (m_Classifier instanceof WeightedInstancesHandler) {
				epsilon = trainUsingWeights(classifier, trainData, target);
			} else {
				epsilon = trainUsingResampling(classifier, trainData, target,
						rand);
			}

			// Stop if error too small or error too big and ignore this model
			if (Utils.grOrEq(epsilon, 0.5) || Utils.eq(epsilon, 0)) {
				if (m_IterationsPerformed == 0) {
					// If we're the first we have to to use it
					m_IterationsPerformed = 1;
				}
				break;
			}

			// Determine the weight to assign to this model
			double reweight = epsilon / (1 - epsilon);
			m_Betas[m_IterationsPerformed] = Math.log(reweight);
			/*
			 * if (m_Debug) { System.err.println("\terror rate = " + epsilon +
			 * "  beta = " + m_Betas[m_IterationsPerformed]); }
			 */
			// Update instance weights
			double oldSumOfWeights = source.sumOfWeights();
			oldSumOfWeights += target.sumOfWeights();
			setWeights(source, beta);
			setWeights(target, reweight);
			double newSumOfWeights = source.sumOfWeights();
			newSumOfWeights += target.sumOfWeights();

			// Renormalize weights
			double d = oldSumOfWeights / newSumOfWeights;
			for (int index = 0; index < source.numInstances(); ++index) {
				Instance instance = source.instance(index);
				double oldWeight = instance.weight();
				double newWeight = oldWeight * d;
				instance.setWeight(newWeight);
			}
			for (int index = 0; index < target.numInstances(); ++index) {
				Instance instance = target.instance(index);
				double oldWeight = instance.weight();
				double newWeight = oldWeight * d;
				instance.setWeight(newWeight);
			}
		}

	}

	private double trainUsingWeights(Classifier classifier,
			Instances trainData, Instances testData) throws Exception {

		classifier.buildClassifier(trainData);
		Evaluation evaluation = new Evaluation(testData);
		evaluation.evaluateModel(classifier, testData);
		return evaluation.errorRate();

	}

	private double trainUsingResampling(Classifier classifier,
			Instances trainData, Instances testData, Random rand)
			throws Exception {

		for (int i = 0; i < MAX_NUM_RESAMPLING_ITERATIONS; ++i) {
			// Build the classifier
			Instances sample = trainData.resampleWithWeights(rand);
			classifier.buildClassifier(sample);

			// Evaluate the classifier
			Evaluation evaluation = new Evaluation(testData);
			evaluation.evaluateModel(classifier, testData);
			double epsilon = evaluation.errorRate();
			if (!Utils.eq(epsilon, 0))
				return epsilon;
		}

		return 0;

	}

	private Instances join(Instances source, Instances target) {

		int n = source.numInstances();
		int m = target.numInstances();
		Instances joint = new Instances(target, n + m);

		for (int index = 0; index < n; ++index) {
			Instance instance = source.instance(index);
			joint.add(instance);
		}
		for (int index = 0; index < m; ++index) {
			Instance instance = target.instance(index);
			joint.add(instance);
		}
		return joint;

	}

	/**
	 * Sets the weights for the next iteration.
	 * 
	 * @param training
	 *            the training instances
	 * @param reweight
	 *            the reweighting factor
	 * @throws Exception
	 *             if something goes wrong
	 */
	private void setWeights(Instances training, double reweight)
			throws Exception {

		for (int index = 0; index < training.numInstances(); ++index) {
			Instance instance = training.instance(index);
			Classifier classifier = m_Classifiers[m_IterationsPerformed];
			double observed = classifier.classifyInstance(instance);
			double actual = instance.classValue();
			if (!Utils.eq(observed, actual)) {
				instance.setWeight(instance.weight() * reweight);
			}
		}

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		// default model?
		if (m_ZeroR != null) {
			return m_ZeroR.distributionForInstance(instance);
		}

		if (m_IterationsPerformed == 0) {
			throw new Exception("No model built");
		} else if (m_IterationsPerformed == 1) {
			return m_Classifiers[0].distributionForInstance(instance);
		}

		double[] retVal = new double[instance.numClasses()];
		for (int t = (m_IterationsPerformed - 1) / 2; t < m_IterationsPerformed; ++t) {
			double beta = m_Betas[t];
			Classifier classifier = m_Classifiers[t];
			double[] dist = classifier.distributionForInstance(instance);
			for (int classIndex = 0; classIndex < retVal.length; ++classIndex) {
				double distribution = dist[classIndex];
				retVal[classIndex] -= distribution * beta;
			}
		}
		// for (int classIndex = 0; classIndex < retVal.length; ++classIndex) {
		// retVal[classIndex] = Math.exp(retVal[classIndex]);
		// }
		Utils.normalize(retVal);
		return retVal;

	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {
		TradaBoost dup = new TradaBoost();
		dup.seed = this.seed;
		dup.m_NumIterations = this.m_NumIterations;
		dup.m_IterationsPerformed = this.m_IterationsPerformed;
		dup.setClassifier(this.m_Classifier);
		if (this.m_ZeroR != null) {
			dup.m_ZeroR = Classifier.makeCopy(this.m_ZeroR);
		}
		if (m_Betas != null) {
			dup.m_Betas = new double[this.m_Betas.length];
			dup.m_Classifiers = new Classifier[this.m_Classifiers.length];
			for (int i = 0; i < this.m_Betas.length; ++i) {
				dup.m_Betas[i] = this.m_Betas[i];
				dup.m_Classifiers[i] = Classifier
						.makeCopy(this.m_Classifiers[i]);
			}
		}
		return dup;
	}

}
