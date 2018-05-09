package strlet.transferLearning.inductive.supervised;

import strlet.transferLearning.inductive.SingleSourceTransfer;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * Abstract utility class for handling settings common to
 * feature transfer classifiers with a single source domain and a single target domains.  

 * In the feature transfer paradigm, both domains share the same 
 * set of features and labels and often a feature transformation metric 
 * is deduced during the learning stage.
 *
 * @author Noam Segev (nsegev@cs.technion.ac.il)
 */
public abstract class SingleSourceFeatureTransfer extends SingleSourceTransfer {

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
