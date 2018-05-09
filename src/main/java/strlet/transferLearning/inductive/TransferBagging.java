package strlet.transferLearning.inductive;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import strlet.auxiliary.ThreadPool;
import strlet.transferLearning.inductive.taskLearning.naive.TargetOnly;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class TransferBagging extends SingleSourceTransfer implements
		Randomizable, WeightedInstancesHandler {

	private int seed = 1;
	private int m_NumIterations = 10;
	protected SingleSourceTransfer m_Classifier;
	protected SingleSourceTransfer[] m_Classifiers = null;

	public TransferBagging() {
		m_Classifier = new TargetOnly();
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
	public void setClassifier(SingleSourceTransfer newClassifier)
			throws Exception {
		m_Classifier = newClassifier.makeDuplicate();
	}

	@Override
	public void setSeed(int seed) {
		this.seed = seed;
	}

	@Override
	public int getSeed() {
		return seed;
	}

	@Override
	public void buildModel(Instances source, Instances target) throws Exception {
		testWithFail(source, target);

		Queue<Future<SingleSourceTransfer>> futures = new LinkedList<Future<SingleSourceTransfer>>();
		m_Classifiers = new SingleSourceTransfer[m_NumIterations];
		Random rand = new Random(getSeed());
		for (int i = 0; i < m_NumIterations; ++i) {
			Instances bagData = source.resampleWithWeights(rand);
			m_Classifiers[i] = m_Classifier.makeDuplicate();
			if (Randomizable.class.isAssignableFrom(m_Classifier.getClass())) {
				Randomizable.class.cast(m_Classifiers[i]).setSeed(
						rand.nextInt());
			}

			if (ThreadPool.poolSize() <= 1) {
				m_Classifiers[i].buildModel(bagData, target);
			} else {
				futures.add(ThreadPool.submit(new ClassifierBuilder(
						m_Classifiers[i], bagData, target)));
			}
		}
		while (!futures.isEmpty())
			futures.poll().get();

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] dist = new double[instance.numClasses()];
		for (SingleSourceTransfer classifier : m_Classifiers) {
			double[] d = classifier.distributionForInstance(instance);
			for (int i = 0; i < dist.length; ++i) {
				dist[i] += d[i];
			}
		}
		Utils.normalize(dist);
		return dist;
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {

		TransferBagging tb = new TransferBagging();
		tb.setNumIterations(getNumIterations());
		tb.setSeed(getSeed());
		tb.setClassifier(m_Classifier);
		return tb;
	}

	private class ClassifierBuilder implements Callable<SingleSourceTransfer> {

		private final SingleSourceTransfer m_Classifier;
		private final Instances m_Source;
		private final Instances m_Target;

		public ClassifierBuilder(SingleSourceTransfer classifier,
				Instances source, Instances target) {
			m_Classifier = classifier;
			m_Source = source;
			m_Target = target;
		}

		@Override
		public SingleSourceTransfer call() throws Exception {
			m_Classifier.buildModel(m_Source, m_Target);
			return m_Classifier;
		}

	}

}
