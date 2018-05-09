package strlet.transferLearning.inductive.taskLearning;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Attribute;
import weka.core.Instances;

public abstract class SingleSourceModelTransfer extends SingleSourceTransfer {

	/**
	 * Generates a classifier. Must initialize all fields of the classifier that
	 * are not being set via options (ie. multiple calls of buildModel must
	 * always lead to the same result). Must not change the dataset in any way.
	 * 
	 * @param source
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if the classifier has not been generated successfully
	 */
	protected abstract void buildModel(Instances source) throws Exception;

	/**
	 * Transforms the previously generated classifier. Must come after a call to
	 * the {@link #buildModel(Instances) buildModel} method. initialize all
	 * fields of the classifier that are not being set via options (ie. multiple
	 * calls of buildModel followed by transferModel must always lead to the
	 * same result). Must not change the dataset in any way.
	 * 
	 * @param target
	 *            set of instances serving as training data
	 * @exception Exception
	 *                if no classifier has been generated before or if the
	 *                classifier transfer was not successful
	 */
	protected abstract void transferModel(Instances target) throws Exception;

	@Override
	public final void buildModel(Instances source, Instances target)
			throws Exception {
		testWithFail(source, target);
		buildModel(source);
		transferModel(target);
	}

	@Override
	protected void testWithFail(Instances source, Instances target)
			throws Exception {

		super.testWithFail(source, target);
		if (source.numAttributes() != target.numAttributes()) {
			throw new Exception(
					"Source and target domains must have the same structure");
		}
		for (int i = 0; i < source.numAttributes(); ++i) {
			Attribute attrS = source.attribute(i);
			Attribute attrT = target.attribute(i);

			if (!attrS.name().equals(attrT.name())) {
				throw new Exception(
						"Source and target domains must have the same structure");
			}
			if (attrS.isNominal() && attrT.isNominal()) {
				if (attrS.numValues() != attrT.numValues()) {
					throw new Exception(
							"Source and target domains must have the same structure");
				}
				for (int j = 0; j < attrS.numValues(); ++j) {
					if (!attrS.value(j).equals(attrT.value(j))) {
						throw new Exception(
								"Source and target domains must have the same structure");
					}
				}
				return;
			}
			if (attrS.type() != attrT.type()) {
				throw new Exception(
						"Source and target domains must have the same structure");
			}
		}
	}

}
