package strlet.transferLearning.inductive;

import strlet.transferLearning.TransferClassifier;
import weka.core.Instances;

/**
 * Abstract utility class for handling settings common to
 * transfer classifiers with a single source domain and a single target domains.  
 *
 * @author Noam Segev (nsegev@cs.technion.ac.il)
 */
public abstract class SingleSourceTransfer extends TransferClassifier {

	/**
	 * Generates a classifier. Must initialize all fields of the classifier that
	 * are not being set via options (ie. multiple calls of buildModel must
	 * always lead to the same result). Must not change the datasets in any way.
	 *
	 * @param source
	 *            set of source instances serving as training data
	 * @param target
	 *            set of target instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */
	public abstract void buildModel(Instances source, Instances target)
			throws Exception;

	/**
	 * Duplicates the current classifier. The current classifier and its
	 * duplicate must be able to give the same results or train the same model
	 * using the {@link #buildModel(Instances, Instances) buildModel} method.
	 * 
	 * @return An exact duplicate of the current model.
	 * @throws Exception
	 *             if the duplication failed for some reason
	 */
	public abstract SingleSourceTransfer makeDuplicate() throws Exception;

	/**
	 * tests the given data by calling the test method and throws an exception
	 * if the test fails.
	 *
	 * @param source
	 *            set of source instances serving as training data
	 * @param target
	 *            set of target instances serving as training data
	 * @throws Exception
	 *             in case the data doesn't pass the tests
	 * @see #test(Instances, Instances)
	 */
	protected void testWithFail(Instances source, Instances target)
			throws Exception {
		if (source == null) {
			throw new Exception("No source data provided");
		}
		if (target == null) {
			throw new Exception("No source data provided");
		}
	}

}
