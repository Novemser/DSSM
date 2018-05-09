package strlet.transferLearning;

import weka.core.Attribute;
import weka.core.Instance;

/**
 * Abstract utility class for handling settings common to
 * any transfer classifier.  
 *
 * @author Noam Segev (nsegev@cs.technion.ac.il)
 */
public abstract class TransferClassifier {

	/**
	 * Classifies the given test instance. The instance has to belong to a
	 * dataset when it's being classified. Note that a classifier MUST implement
	 * either this or distributionForInstance().
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return the predicted most likely class for the instance or
	 *         Instance.missingValue() if no prediction is made
	 * @exception Exception
	 *                if an error occurred during the prediction
	 */
	public double classifyInstance(Instance instance) throws Exception {

		double[] dist = distributionForInstance(instance);
		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}
		switch (instance.classAttribute().type()) {
		case Attribute.NOMINAL:
			int maxIndex = 0;
			for (int i = 1; i < dist.length; i++) {
				if (dist[i] > dist[maxIndex]) {
					maxIndex = i;
				}
			}
			if (dist[maxIndex] > 0) {
				return maxIndex;
			} else {
				return Instance.missingValue();
			}
		case Attribute.NUMERIC:
			return dist[0];
		default:
			return Instance.missingValue();
		}
	}

	/**
	 * Predicts the class memberships for a given instance. If an instance is
	 * unclassified, the returned array elements must be all zero. If the class
	 * is numeric, the array must consist of only one element, which contains
	 * the predicted value. Note that a classifier MUST implement either this or
	 * classifyInstance().
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return an array containing the estimated membership probabilities of the
	 *         test instance in each class or the numeric prediction
	 * @exception Exception
	 *                if distribution could not be computed successfully
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {

		double[] dist = new double[instance.numClasses()];
		switch (instance.classAttribute().type()) {
		case Attribute.NOMINAL:
			double classification = classifyInstance(instance);
			if (Instance.isMissingValue(classification)) {
				return dist;
			} else {
				dist[(int) Math.round(classification)] = 1.0;
			}
			return dist;
		case Attribute.NUMERIC:
			double v = classifyInstance(instance);
			if (Instance.isMissingValue(v)) {
				return dist;
			} else {
				dist[0] = v;
			}
			return dist;
		default:
			return dist;
		}
	}

}
