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

public class UnderBagging extends Bagging {

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

		if (m_CalcOutOfBag) {
			throw new IllegalArgumentException(
					"Underbagging does not support out-of-bag error!");
		}

		ClassifierBuilder[] builders = new ClassifierBuilder[m_Classifiers.length];
		Random rand = new Random(m_Seed);
		for (int j = 0; j < m_Classifiers.length; ++j) {
			builders[j] = new ClassifierBuilder(data, m_Classifiers[j]);
			builders[j].setSeed(rand.nextInt());
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

		m_OutOfBagError = 0;
	}

	private class ClassifierBuilder implements Callable<Classifier> {

		private final Instances unbalancedData;
		private int seed = 1;
		private final Classifier baseClassifier;

		public ClassifierBuilder(Instances unbalancedData,
				Classifier baseClassifier) {
			this.unbalancedData = unbalancedData;
			this.baseClassifier = baseClassifier;
		}

		@Override
		public Classifier call() throws Exception {

			BalancedBag bag = new BalancedBag(unbalancedData, seed);
			if (Randomizable.class.isInstance(baseClassifier)) {
				Randomizable.class.cast(baseClassifier).setSeed(seed);
			}
			baseClassifier.buildClassifier(bag.getBag());
			return baseClassifier;
		}

		public void setSeed(int seed) {
			this.seed = seed;
		}

	}

}
