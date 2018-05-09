package strlet.auxiliary;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;

public class ParallelBagging extends Bagging {

	private static final long serialVersionUID = 4206439301558156137L;

	@Override
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		if (m_Classifier == null) {
			throw new Exception("A base classifier has not been specified!");
		}
		m_Classifiers = Classifier.makeCopies(m_Classifier, m_NumIterations);

		if (m_CalcOutOfBag && (m_BagSizePercent != 100)) {
			throw new IllegalArgumentException("Bag size needs to be 100% if "
					+ "out-of-bag error is to be calculated!");
		}

		int bagSize = data.numInstances() * m_BagSizePercent / 100;
		Random random = new Random(m_Seed);

		boolean[][] inBag = null;
		if (m_CalcOutOfBag)
			inBag = new boolean[m_Classifiers.length][];

		ClassifierBuilder[] builders = new ClassifierBuilder[m_Classifiers.length];
		for (int j = 0; j < m_Classifiers.length; ++j) {
			Instances bagData = null;

			// create the in-bag dataset
			if (m_CalcOutOfBag) {
				inBag[j] = new boolean[data.numInstances()];
				bagData = resampleWithWeights(data, random, inBag[j]);
//				 bagData = data.resampleWithWeights(random, inBag[j]);
			} else {
				bagData = data.resampleWithWeights(random);
				if (bagSize < data.numInstances()) {
					bagData.randomize(random);
					Instances newBagData = new Instances(bagData, 0, bagSize);
					bagData = newBagData;
				}
			}

			if (Randomizable.class.isAssignableFrom(m_Classifier.getClass())) {
				Randomizable.class.cast(m_Classifiers[j]).setSeed(
						random.nextInt());
			}

			builders[j] = new ClassifierBuilder(m_Classifiers[j], bagData);
		}

		if (ThreadPool.poolSize() <= 1) {
			for (ClassifierBuilder builder : builders)
				builder.call();
		} else {
			Queue<Future<Classifier>> futures = new LinkedList<Future<Classifier>>();
			for (ClassifierBuilder builder : builders)
				futures.add(ThreadPool.submit(builder));
			while (!futures.isEmpty())
				futures.poll().get();
		}

		// calc OOB error?
		if (getCalcOutOfBag()) {
			double outOfBagCount = 0.0;
			double errorSum = 0.0;
			boolean numeric = data.classAttribute().isNumeric();

			for (int i = 0; i < data.numInstances(); i++) {
				double vote;
				double[] votes;
				if (numeric)
					votes = new double[1];
				else
					votes = new double[data.numClasses()];

				// determine predictions for instance
				int voteCount = 0;
				for (int j = 0; j < m_Classifiers.length; j++) {
					if (inBag[j][i])
						continue;

					voteCount++;
					// double pred =
					// m_Classifiers[j].classifyInstance(data.instance(i));
					if (numeric) {
						// votes[0] += pred;
						votes[0] = m_Classifiers[j].classifyInstance(data
								.instance(i));
					} else {
						// votes[(int) pred]++;
						double[] newProbs = m_Classifiers[j]
								.distributionForInstance(data.instance(i));
						// average the probability estimates
						for (int k = 0; k < newProbs.length; k++) {
							votes[k] += newProbs[k];
						}
					}
				}

				// "vote"
				if (numeric) {
					vote = votes[0];
					if (voteCount > 0) {
						vote /= voteCount; // average
					}
				} else {
					if (Utils.eq(Utils.sum(votes), 0)) {
					} else {
						Utils.normalize(votes);
					}
					vote = Utils.maxIndex(votes); // predicted class
				}

				// error for instance
				outOfBagCount += data.instance(i).weight();
				if (numeric) {
					errorSum += StrictMath.abs(vote
							- data.instance(i).classValue())
							* data.instance(i).weight();
				} else {
					if (vote != data.instance(i).classValue())
						errorSum += data.instance(i).weight();
				}
			}

			m_OutOfBagError = errorSum / outOfBagCount;
		} else {
			m_OutOfBagError = 0;
		}
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
}
