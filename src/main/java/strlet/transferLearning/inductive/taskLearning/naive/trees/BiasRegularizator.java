package strlet.transferLearning.inductive.taskLearning.naive.trees;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import strlet.auxiliary.ThreadPool;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.taskLearning.SingleSourceModelTransfer;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * <!-- globalinfo-start --> Meta classifier for building and using a standard
 * classifier in a model transfer settings. Creates a bagged ensemble of the
 * classifier on the source examples alone, followed by a re-weighting of the
 * ensemble weights to minimize error on target examples.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * @author Noam Segev (nsegev@cs.technion.ac.il)
 */
public class BiasRegularizator extends SingleSourceModelTransfer implements
		Randomizable, WeightedInstancesHandler {

	private int seed = 1;
	private double[] m_Weights = null;
	private int m_NumIterations = 10;
	private Classifier baseClassifier = new ZeroR();
	private Classifier[] m_Classifiers = null;

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

		baseClassifier = Classifier.makeCopy(newClassifier);
	}

	/**
	 * Get the classifier used as the base learner.
	 * 
	 * @return the classifier used as the classifier
	 */
	public Classifier getClassifier() {
		return baseClassifier;
	}

	@Override
	protected void buildModel(Instances source) throws Exception {

		source = new Instances(source);
		source.deleteWithMissingClass();
		Queue<Future<Classifier>> futures = new LinkedList<Future<Classifier>>();
		m_Classifiers = Classifier.makeCopies(baseClassifier, m_NumIterations);
		m_Weights = new double[m_NumIterations];
		Random rand = new Random(getSeed());

		for (int i = 0; i < m_NumIterations; ++i) {
			m_Weights[i] = 1;
			Instances bagData = source.resampleWithWeights(rand);
			if (Randomizable.class.isAssignableFrom(baseClassifier.getClass())) {
				Randomizable.class.cast(m_Classifiers[i]).setSeed(
						rand.nextInt());
			}

			if (ThreadPool.poolSize() <= 1) {
				m_Classifiers[i].buildClassifier(bagData);
			} else {
				futures.add(ThreadPool.submit(new ClassifierBuilder(
						m_Classifiers[i], bagData)));
			}
		}
		while (!futures.isEmpty())
			futures.poll().get();

		Utils.normalize(m_Weights);

	}

	@Override
	protected void transferModel(Instances target) throws Exception {

		double[] targetAcc = new double[m_NumIterations];

		for (int i = 0; i < m_NumIterations; ++i) {
			targetAcc[i] = classifierAccuracy(m_Classifiers[i], target);
		}

		for (int i = 0; i < m_NumIterations; ++i) {
			targetAcc[i] *= targetAcc[i];
		}
		Utils.normalize(targetAcc);
		m_Weights = targetAcc;

	}

	private static double classifierAccuracy(Classifier classifier,
			Instances test) throws Exception {
		if (classifier == null)
			return 0;

		int[][] confusionMatrix = new int[test.numClasses()][test.numClasses()];

		for (int i = 0; i < test.numInstances(); ++i) {
			Instance instance = test.instance(i);
			int classification = (int) Math.round(classifier
					.classifyInstance(instance));
			int trueClass = (int) Math.round(instance.classValue());
			++confusionMatrix[trueClass][classification];
		}

		int totalCorrect = 0;
		for (int i = 0; i < confusionMatrix.length; ++i) {
			totalCorrect += confusionMatrix[i][i];
		}
		return (totalCorrect + 0.0) / test.numInstances();

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		double[] dist = new double[instance.numClasses()];
		for (int i = 0; i < dist.length; ++i) {
			dist[i] = 0;
		}

		if (instance.classAttribute().isNumeric()) {
			for (int i = 0; i < m_NumIterations; ++i) {
				dist[0] += m_Classifiers[i].classifyInstance(instance)
						* m_Weights[i];
			}
		} else {
			for (int i = 0; i < m_NumIterations; ++i) {
				double[] d = m_Classifiers[i].distributionForInstance(instance);
				for (int j = 0; j < d.length; ++j) {
					dist[j] = d[j] * m_Weights[i];
				}
			}
		}

		return dist;
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {
		BiasRegularizator other = new BiasRegularizator();
		other.setNumIterations(getNumIterations());
		other.setSeed(getSeed());
		other.setClassifier(getClassifier());
		if (m_Weights == null)
			return other;
		other.m_Classifiers = new Classifier[getNumIterations()];
		other.m_Weights = new double[getNumIterations()];
		for (int i = 0; i < m_Classifiers.length; i++) {
			other.m_Classifiers[i] = Classifier.makeCopy(m_Classifiers[i]);
			other.m_Weights[i] = m_Weights[i];
		}
		return other;
	}

	private class ClassifierBuilder implements Callable<Classifier> {

		private final Classifier m_Classifier;
		private final Instances m_Source;

		public ClassifierBuilder(Classifier classifier, Instances source) {
			m_Classifier = classifier;
			m_Source = source;
		}

		@Override
		public Classifier call() throws Exception {
			m_Classifier.buildClassifier(m_Source);
			return m_Classifier;
		}

	}

}
