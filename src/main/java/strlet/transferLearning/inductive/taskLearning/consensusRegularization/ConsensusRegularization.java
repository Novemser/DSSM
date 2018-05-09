package strlet.transferLearning.inductive.taskLearning.consensusRegularization;

import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import strlet.auxiliary.ThreadPool;
import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.taskLearning.SingleSourceModelTransfer;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Debug.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Optimization;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class ConsensusRegularization extends SingleSourceModelTransfer
		implements Randomizable, WeightedInstancesHandler {

	private int seed = 1;
	private int m_NumIterations = 10;

	/** The base classifier to use */
	protected Classifier m_Classifier = new ZeroR();
	/** Array for storing the generated base classifiers. */
	protected Classifier[] m_Classifiers = null;
	private double[] m_Weights = null;

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
	protected void buildModel(Instances source) throws Exception {

		source = new Instances(source);
		source.deleteWithMissingClass();
		int numIterations = 2 * m_NumIterations;
		m_Classifiers = Classifier.makeCopies(m_Classifier, numIterations);
		m_Weights = new double[numIterations];
		buildClassifiers(source, 0);

	}

	@Override
	protected void transferModel(Instances target) throws Exception {

		target = new Instances(target);
		target.deleteWithMissingClass();
		buildClassifiers(target, m_NumIterations);
		Utils.normalize(m_Weights);

		ConsensusOptimization opt = new ConsensusOptimization(target,
				m_Classifiers);
		double[] x = new double[m_Weights.length];
		for (int i = 0; i < x.length; ++i) {
			x[i] = m_Weights[i];
		}
		double[][] constraints = new double[2][x.length];
		for (int i = 0; i < x.length; ++i) {
			constraints[0][i] = 0;
			constraints[1][i] = 1;
		}

		// Find the minimum, up to 1000 (200*5) iterations
		opt.setMaxIteration(50);
		opt.setDebug(false);
		for (int i = 0; i < 20; ++i) {

			for (int j = 0; j < x.length; ++j) {
				if (x[j] < 0) {
					x[j] = 0;
				}
			}
			if (Utils.eq(Utils.sum(x), 0)) {
				break;
			} else {
				Utils.normalize(x);
			}

			x = opt.findArgmin(x, constraints);
			if (x == null) {
				x = opt.getVarbValues();
			} else {
				break;
			}
		}

		for (int j = 0; j < x.length; ++j) {
			if (x[j] < 0) {
				x[j] = 0;
			}
		}
		if (!Utils.eq(Utils.sum(x), 0)) {
			Utils.normalize(x);
			m_Weights = x;
		}

	}

	private void buildClassifiers(Instances source, int startLocation)
			throws Exception, InterruptedException, ExecutionException {

		Random rand = new Random(seed);
		Queue<Future<Classifier>> futures = new LinkedList<Future<Classifier>>();
		for (int i = startLocation; i < startLocation + m_NumIterations; ++i) {
			Instances bag = source.resampleWithWeights(rand);
			m_Weights[i] = 1.0;
			if (Randomizable.class.isAssignableFrom(m_Classifier.getClass())) {
				Randomizable.class.cast(m_Classifiers[i]).setSeed(
						rand.nextInt());
			}
			if (ThreadPool.poolSize() <= 1) {
				m_Classifiers[i].buildClassifier(bag);
			} else {
				futures.add(ThreadPool.submit(new ClassifierBuilder(
						m_Classifiers[i], bag)));
			}
		}
		while (!futures.isEmpty()) {
			futures.poll().get();
		}
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {

		ConsensusRegularization dup = new ConsensusRegularization();
		dup.setNumIterations(m_NumIterations);
		dup.setSeed(getSeed());
		dup.setClassifier(m_Classifier);
		if (m_Weights != null) {
			dup.m_Classifiers = new Classifier[m_Classifiers.length];
			dup.m_Weights = new double[m_Weights.length];
			for (int i = 0; i < m_Weights.length; ++i) {
				dup.m_Classifiers[i] = Classifier.makeCopy(m_Classifiers[i]);
				dup.m_Weights[i] = m_Weights[i];
			}
		}
		return dup;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] dist = new double[instance.numClasses()];
		for (int i = 0; i < m_Weights.length; ++i) {
			double[] d = m_Classifiers[i].distributionForInstance(instance);
			double w = m_Weights[i];
			for (int j = 0; j < dist.length; ++j) {
				dist[j] += w * d[j];
			}
		}
		Utils.normalize(dist);
		return dist;
	}

	private class ClassifierBuilder implements Callable<Classifier> {
		private final Classifier m_Classifier;
		private final Instances m_Data;

		public ClassifierBuilder(Classifier classifier, Instances data) {
			m_Classifier = classifier;
			m_Data = data;
		}

		@Override
		public Classifier call() throws Exception {
			m_Classifier.buildClassifier(m_Data);
			return m_Classifier;
		}
	}

	private class ConsensusOptimization extends Optimization {

		private final int m_NumClasses;
		private final double[][][] m_Distributions;

		public ConsensusOptimization(Instances data, Classifier[] classifiers)
				throws Exception {

			int numInstances = data.numInstances();
			int numIterations = classifiers.length;
			m_NumClasses = data.numClasses();

			double[][][] distributions = new double[numInstances][numIterations][];
			for (int index = 0; index < numInstances; ++index) {
				Instance instance = data.instance(index);
				for (int j = 0; j < numIterations; ++j) {
					Classifier clr = classifiers[j];
					distributions[index][j] = clr
							.distributionForInstance(instance);
				}
			}
			m_Distributions = distributions;

		}

		@Override
		public String getRevision() {
			return null;
		}

		@Override
		// For a single instance i, and a single class c, the classifier c2 has
		// probability (p_c2) of m_Distributions[i][c2][c].
		// Therefore, for a single instance i, and a single class c,
		// the objective is - sum(x_c2 * p_c2) * ln (sum(x_c2 * p_c2)) +
		// (sum(x_c2)-1)^2
		protected double objectiveFunction(double[] x) throws Exception {

			double[] instSum = new double[m_Distributions.length];
			for (int index = 0; index < m_Distributions.length; ++index) {
				// Calculate entropy for a single instance
				double[][] distributions = m_Distributions[index];
				double[] innerSum = new double[m_NumClasses];
				for (int c = 0; c < m_NumClasses; ++c) {
					// Calculate entropy for a single class
					double[] innerProd = new double[distributions.length];
					for (int i = 0; i < innerProd.length; ++i) {
						double w = x[i];
						double p = distributions[i][c];
						innerProd[i] = w * p;
					}
					double sum = Utils.sum(innerProd);
					innerSum[c] = sum * log(sum);
				}
				instSum[index] = Utils.sum(innerSum);
			}

			double consensus = Utils.sum(instSum);
			double reg = (Utils.sum(x) - 1) * (Utils.sum(x) - 1);
			reg *= m_NumClasses * instSum.length;
			// System.out.println(reg - consensus);
			return reg - consensus;

		}

		@Override
		protected double[] evaluateGradient(double[] x) throws Exception {

			Operation op = new FirstDerivative();
			double[][] partials = calcPartials(x, op);
			double reg = 2 * (Utils.sum(x) - 1);
			return sumRow(partials, reg);

		}

		@Override
		protected double[] evaluateHessian(double[] x, int rowIndex)
				throws Exception {

			Operation op = new SecondDerivative(rowIndex);
			double[][] partials = calcPartials(x, op);
			return sumRow(partials, 2);

		}

		private double[][] calcPartials(double[] x, Operation op) {

			double[][] partials = new double[x.length][m_Distributions.length];

			// Calculate each single instance
			for (int index = 0; index < m_Distributions.length; ++index) {
				double[][] distributions = m_Distributions[index];
				double[][] innerSum = new double[x.length][m_NumClasses];
				// Calculate for a single class
				for (int c = 0; c < m_NumClasses; ++c) {
					// The sum of (p_c2 * w_c2)
					double[] innerProd = new double[distributions.length];
					for (int i = 0; i < innerProd.length; ++i) {
						double w = x[i];
						double p = distributions[i][c];
						innerProd[i] = w * p;
					}
					double sum = Utils.sum(innerProd);
					// The derivative of each w_c2 for class c
					for (int i = 0; i < innerProd.length; ++i) {
						innerSum[i][c] = op.op(distributions, i, c, sum);
						// In case of division by zero
						if (Double.isNaN(innerSum[i][c])) {
							innerSum[i][c] = 1.0;
						}
					}
				}
				// The derivative of each w_c2 for instance i
				for (int i = 0; i < x.length; ++i) {
					partials[i][index] = Utils.sum(innerSum[i]);
				}
			}

			return partials;

		}

		private double[] sumRow(double[][] partials, double reg) {

			reg *= m_NumClasses * m_Distributions.length;
			double[] sum = new double[m_Distributions[0].length];
			for (int i = 0; i < sum.length; ++i) {
				sum[i] = reg - Utils.sum(partials[i]);
			}
			return sum;

		}

		private double log(double x) {

			if (x <= 0 || x >= 1) {
				return 0;
			}
			return x * Math.log(x);

		}

		private abstract class Operation {
			public abstract double op(double[][] dist, int iterationIndex,
					int classIndex, double sum);
		}

		private class FirstDerivative extends Operation {
			@Override
			public double op(double[][] dist, int iterationIndex,
					int classIndex, double sum) {

				double p = dist[iterationIndex][classIndex];
				return p + p * log(sum);

			}
		}

		private class SecondDerivative extends Operation {

			private final int m_RowIndex;

			public SecondDerivative(int rowIndex) {
				m_RowIndex = rowIndex;
			}

			@Override
			public double op(double[][] dist, int iterationIndex,
					int classIndex, double sum) {

				double p_i = dist[iterationIndex][classIndex];
				double p_r = dist[m_RowIndex][classIndex];
				return (p_i * p_r) / sum;

			}

		}

	}

}
