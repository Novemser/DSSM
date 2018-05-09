package strlet.transferLearning.inductive.supervised.bagging;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import strlet.auxiliary.ThreadPool;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.supervised.SingleSourceInstanceTransfer;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class TrBagg extends SingleSourceInstanceTransfer implements
		Randomizable, WeightedInstancesHandler {

	private int seed = 1;
	private Classifier m_Classifier = new ZeroR();
	private int m_NumIterations = 10;

	/** a ZeroR model in case no model can be built from the data */
	protected Classifier m_ZeroR = null;

	/** Array for storing the generated base classifiers. */
	protected Classifier[] m_Classifiers = null;

	/** Array of flags, indicating active/disabled ensemble members */
	protected boolean[] m_inUse = null;

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
		m_Classifier.getCapabilities().testWithFail(source);
		m_Classifier.getCapabilities().testWithFail(target);

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
		} else {
			m_ZeroR = null;
		}

		learningPhase(source, target);
		filterPhase(target);
	}

	private void learningPhase(Instances source, Instances target)
			throws Exception {

		Instances data = new Instances(target, source.numInstances()
				+ target.numInstances());
		for (int index = 0; index < source.numInstances(); ++index) {
			data.add(source.instance(index));
		}
		for (int index = 0; index < target.numInstances(); ++index) {
			data.add(target.instance(index));
		}

		m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);
		m_inUse = new boolean[m_NumIterations];
		Random rand = new Random(getSeed());

		Queue<Future<Classifier>> futures = new LinkedList<Future<Classifier>>();
		for (int i = 0; i < m_NumIterations; ++i) {
			m_inUse[i] = false;
			Instances bagData = source.resampleWithWeights(rand);
			if (Randomizable.class.isAssignableFrom(m_Classifier.getClass())) {
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

		if (Randomizable.class.isAssignableFrom(m_Classifier.getClass())) {
			Randomizable.class.cast(m_Classifier).setSeed(rand.nextInt());
		}
		m_Classifier.buildClassifier(target);

	}

	private void filterPhase(Instances target) throws Exception {

		double[] errors = new double[m_NumIterations];
		for (int i = 0; i < m_NumIterations; ++i) {
			errors[i] = calcErr(m_Classifiers[i], target);
		}
		int[] sorted = Utils.sort(errors);
		m_inUse[sorted[0]] = true;

		double e = calcEpsilon(target);
		for (int i = 1; i < m_NumIterations; ++i) {
			m_inUse[sorted[i]] = true;
			double eTag = calcEpsilon(target);
			if (Utils.smOrEq(eTag, e)) {
				e = eTag;
			} else {
				m_inUse[sorted[i]] = false;
			}
		}
	}

	private double calcErr(Classifier classifier, Instances data)
			throws Exception {

		int wrong = 0;
		for (int index = 0; index < data.numInstances(); ++index) {
			Instance instance = data.instance(index);
			int expected = (int) Math.round(instance.classValue());
			double observed = classifier.classifyInstance(instance);
			if (!Utils.eq(expected, observed))
				++wrong;
		}
		return (wrong + 0.0) / data.numInstances();
	}

	private double calcEpsilon(Instances data) throws Exception {

		int wrong = 0;
		for (int index = 0; index < data.numInstances(); ++index) {
			Instance instance = data.instance(index);
			double expected = instance.classValue();

			int[] observed = new int[data.numClasses()];
			for (int i = 0; i < m_NumIterations; ++i) {
				if (m_inUse[i]) {
					Classifier c = m_Classifiers[i];
					double o = c.classifyInstance(instance);
					++observed[(int) Math.round(o)];
				}
			}
			if (!Utils.eq(expected, Utils.maxIndex(observed)))
				++wrong;
		}
		return (wrong + 0.0) / data.numInstances();
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (m_ZeroR != null) {
			return m_ZeroR.distributionForInstance(instance);
		}

		if (instance.classAttribute().isNumeric()) {
			double[] d = { 0 };
			d[0] = m_Classifier.classifyInstance(instance);
			int N = 1;
			for (int i = 0; i < m_NumIterations; ++i) {
				if (m_inUse[i]) {
					d[0] += m_Classifiers[i].classifyInstance(instance);
					++N;
				}
			}
			d[0] /= N;
			return d;
		} else {
			double[] dist = new double[instance.numClasses()];
			for (int i = 0; i < m_NumIterations; ++i) {
				if (m_inUse[i]) {
					double[] d = m_Classifiers[i]
							.distributionForInstance(instance);
					for (int j = 0; j < d.length; ++j)
						dist[j] += d[j];
				}
			}

			double[] d = m_Classifier.distributionForInstance(instance);
			for (int j = 0; j < d.length; ++j)
				dist[j] += d[j];

			Utils.normalize(dist);
			return dist;
		}
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {
		TrBagg other = new TrBagg();
		other.setSeed(getSeed());
		other.setClassifier(getClassifier());
		other.setNumIterations(getNumIterations());
		
		if (m_ZeroR != null) {
			other.m_ZeroR = Classifier.makeCopy(m_ZeroR);
		} else {
			other.m_ZeroR = null;
		}
		
		if (m_inUse != null) {
			other.m_Classifiers = new Classifier[m_NumIterations];
			other.m_inUse = new boolean[m_NumIterations];
			for (int i=0; i<m_NumIterations; ++i) {
				other.m_inUse[i] = m_inUse[i];
				other.m_Classifiers[i] = Classifier.makeCopy(m_Classifiers[i]);
			}
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
