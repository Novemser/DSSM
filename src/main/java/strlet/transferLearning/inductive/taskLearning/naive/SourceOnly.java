package strlet.transferLearning.inductive.taskLearning.naive;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import strlet.transferLearning.inductive.taskLearning.SingleSourceModelTransfer;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;

/**
<!-- globalinfo-start -->
* Meta classifier for building and using a standard classifier in a transfer learning settings.
* Trains the classifier on the source examples alone.
* <p/>
<!-- globalinfo-end -->
*
* @author Noam Segev (nsegev@cs.technion.ac.il)
*/
public class SourceOnly extends SingleSourceModelTransfer {

	/** The base classifier to use */
	private Classifier baseClassifier = new ZeroR();
	
	@Override
	protected void buildModel(Instances source) throws Exception {
		testWithFail(source, source);
		source = new Instances(source);
		source.deleteWithMissingClass();
		baseClassifier.buildClassifier(source);
	}
	
	@Override
	protected void transferModel(Instances target) throws Exception {
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		return baseClassifier.distributionForInstance(instance);
	}

	/**
	   * Set the base learner.
	   *
	   * @param baseClassifier the classifier to use.
	   */
	public void setBaseClassifier(Classifier baseClassifier) throws Exception {
		this.baseClassifier = Classifier.makeCopy(baseClassifier);
	}

	@Override
	public SingleSourceTransfer makeDuplicate() throws Exception {

		SourceOnly other = new SourceOnly();
		other.baseClassifier = Classifier.makeCopy(baseClassifier);
		return other;

	}

}
